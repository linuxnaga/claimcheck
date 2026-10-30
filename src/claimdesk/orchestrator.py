"""The orchestrator (plain Python, not an agent): Intake, then three specialists in parallel, then the hard
rules, then the Adjudicator for open questions only, then the letter and its checks.
"""
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

from claimcheck import config
from claimcheck.tracing import log_event, observe

from .agents import model_letter_check, run_adjudicator, run_letter, run_specialist
from .critic import bad_citations, check_letter
from .rules import apply_rules, empty_facts, resolve_open_issue
from .tools import claim_summary

SYSTEMS = ("full", "no_adjudicator")
SPECIALISTS = ("policy", "coding", "fraud")


def facts_from(claim, runs):
    """Turn the specialists' findings into facts for the rules. Every id must be backed by a tool result."""
    f = empty_facts()
    lines = {l["no"]: l for l in claim["lines"]}
    notes = []
    if "intake" in runs:
        rep = runs["intake"].finding
        f["complete"], f["missing_documents"] = rep.complete, rep.missing_documents
    if "policy" in runs:
        p, pol = runs["policy"].finding, runs["policy"]
        f["excluded_rule"], f["wait_rule"] = p.excluded_rule, p.wait_rule
        # a rule that rejects must be backed by the tool result it depends on
        waits = {"R-WAIT-PED": "pre_existing", "R-WAIT-MAT": "maternity"}
        if p.wait_rule and not any(r["wait_type"].replace("ped", "pre_existing") == waits[p.wait_rule]
                                   and not r["covered"] for r in pol.results("waiting_period_checker")):
            notes.append(f"policy said {p.wait_rule} but no waiting_period_checker result shows covered false")
            f["wait_rule"] = None
        if p.excluded_rule == "R-NETWORK-PREAUTH" and not any(not r["network"] for r in pol.results("get_hospital")):
            notes.append("policy said R-NETWORK-PREAUTH but no get_hospital result shows network false")
            f["excluded_rule"] = None
        f["covered_lines"] = [n for n in p.covered_lines if n in lines]
        if len(f["covered_lines"]) != len(p.covered_lines):
            notes.append("policy named lines that are not on the claim")
        # only non_medical lines may be left out (R-NONMED); a medical line left out is put back
        dropped = [n for n, l in lines.items() if n not in f["covered_lines"] and l["category"] != "non_medical"]
        if dropped:
            notes.append(f"policy left out medical lines {dropped}; they are covered unless coding finds them invalid")
            f["covered_lines"] = sorted(f["covered_lines"] + dropped)
    if "coding" in runs:
        c = runs["coding"].finding
        f["invalid_lines"] = [n for n in c.invalid_lines if lines.get(n, {}).get("category") == "procedure"]
        f["long_stay"] = c.long_stay
    if "fraud" in runs:
        fr = runs["fraud"].finding
        matches = {m["past_claim_id"]: m for r in runs["fraud"].results("find_similar_claims") for m in r["matches"]}
        for said in (fr.duplicate_of, fr.near_duplicate_of):
            m = matches.get(str(said).strip().upper()) if said else None
            if said and m is None:
                notes.append(f"fraud named {said}, which its tool never returned")
            elif m and m["exact"]:
                f["duplicate_of"] = m["past_claim_id"]
            elif m and not f["near_duplicate_of"]:
                f["near_duplicate_of"] = m["past_claim_id"]
        if f["duplicate_of"]:
            f["near_duplicate_of"] = None
        f["inflated"], f["watchlist"] = fr.inflated, fr.watchlist
    return f, notes


def _finish(claim, system, result, runs, t0, control=None, letter=None, letter_problems=(), warning=None,
            flagged=False, facts=None, notes=()):
    agents = [r.to_dict() for r in runs]
    tokens = sum(a["tokens"] for a in agents)
    out = dict(claim_id=claim["claim_id"], system=system, decision=result["decision"], payable=result["payable"],
               needs_human=result["needs_human"] or bool(control) or bool(letter_problems),
               human_reasons=result.get("human_reasons", []), reasons=result["reasons"], calc=result.get("calc"),
               open_issue=result.get("open_issue"), control=control,
               letter=letter.model_dump() if letter else None, letter_problems=list(letter_problems),
               letter_warning=warning, injection_flag=flagged, facts=facts, notes=list(notes),
               bad_citations=sorted({c for r in runs for c in bad_citations(r)}), agents=agents, tokens=tokens,
               llm_calls=sum(a["llm_calls"] for a in agents), json_retries=sum(a["json_retries"] for a in agents),
               cost_usd=round(tokens / 1e6 * config.PRICE_PER_1M_TOKENS, 6), latency_s=round(time.time() - t0, 2))
    log_event("claimdesk_case", {k: out[k] for k in ("claim_id", "system", "decision", "payable", "needs_human",
                                                     "control", "tokens", "latency_s", "agents")})
    return out


@observe("claimdesk_decide")
def decide_claim(claim, system="full", disable=(), write_letter=True):
    """Decide one claim. system: 'full' or 'no_adjudicator'. disable: specialists to leave out (ablation)."""
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}")
    t0 = time.time()
    summary, flagged = claim_summary(claim)
    runs = {}
    pend = {"decision": "pend", "payable": None, "needs_human": True, "reasons": [], "calc": None}

    runs["intake"] = run_specialist("intake", claim, summary)
    if runs["intake"].finding is None:
        return _finish(claim, system, pend, list(runs.values()), t0, control="specialist_failed:intake",
                       flagged=flagged)
    if not runs["intake"].finding.complete:
        facts, notes = facts_from(claim, runs)
        result = apply_rules(claim, facts)
        return _letter_and_finish(claim, system, result, runs, t0, write_letter, flagged, facts, notes)

    roles = [r for r in SPECIALISTS if r not in disable]
    with ThreadPoolExecutor(max_workers=len(roles) or 1) as pool:
        for role, run in zip(roles, pool.map(lambda r: run_specialist(r, claim, summary), roles)):
            runs[role] = run
    failed = [r for r in roles if runs[r].finding is None]
    if failed:
        return _finish(claim, system, pend, list(runs.values()), t0,
                       control="specialist_failed:" + ",".join(failed), flagged=flagged)

    facts, notes = facts_from(claim, runs)
    result = apply_rules(claim, facts)
    if result["open_issue"]:
        choice, reason = "pend", ""
        if system == "full":
            match = next((m for r in runs["fraud"].results("find_similar_claims") for m in r["matches"]
                          if m["past_claim_id"] == result["open_issue"]["past_claim_id"]), None)
            findings = {k: runs[k].finding.model_dump() for k in roles}
            runs["adjudicator"] = adj = run_adjudicator(claim, result["open_issue"], findings, match)
            if adj.finding is not None:
                choice, reason = adj.finding.decision, adj.finding.reason
        result = resolve_open_issue(result, choice, reason)
    return _letter_and_finish(claim, system, result, runs, t0, write_letter, flagged, facts, notes)


def _letter_and_finish(claim, system, result, runs, t0, write_letter, flagged, facts, notes):
    letter, problems, warning = None, [], None
    result = dict(result, claim_id=claim["claim_id"])
    if write_letter:
        run = run_letter(result)
        if run.finding is not None:
            problems = check_letter(result, run.finding)
            if problems:                      # one revision, then a human sees the problems
                runs["letter_first"] = run
                run = run_letter(result, problems)
                problems = check_letter(result, run.finding) if run.finding else ["letter writer failed"]
        else:
            problems = ["letter writer failed"]
        runs["letter"] = run
        letter = run.finding
        if letter is not None and not problems and config.USE_LLM_CRITIC:
            warning = model_letter_check(result, letter, run.usage)
    return _finish(claim, system, result, list(runs.values()), t0, letter=letter, letter_problems=problems,
                   warning=warning, flagged=flagged, facts=facts, notes=notes)


# ---------- case store: drafts wait for a human --------------------------------------
def _db():
    config.ensure_dirs()
    con = sqlite3.connect(config.DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS cd_cases (case_id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id TEXT, "
                "decision TEXT, payable REAL, needs_human INTEGER, letter TEXT, status TEXT, reviewer TEXT, "
                "note TEXT, final_payable REAL)")
    return con


def create_decision_draft(result):
    """Save a decision as DRAFT. Only review_case (called by the app for a human) can confirm it."""
    letter = result["letter"]["body"] if result.get("letter") else ""
    with _db() as con:
        cur = con.execute("INSERT INTO cd_cases (claim_id, decision, payable, needs_human, letter, status) "
                          "VALUES (?,?,?,?,?,?)", (result["claim_id"], result["decision"], result["payable"],
                                                   int(result["needs_human"]), letter, "DRAFT"))
        return cur.lastrowid


def list_cases(status="DRAFT"):
    cols = ("case_id", "claim_id", "decision", "payable", "needs_human", "letter", "status", "reviewer", "note",
            "final_payable")
    with _db() as con:
        rows = con.execute(f"SELECT {', '.join(cols)} FROM cd_cases WHERE status=? ORDER BY payable DESC",
                           (status,)).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def review_case(case_id, reviewer, action, note="", amount=None):
    """action: CONFIRM, CHANGE_AMOUNT (with amount) or REJECT_DRAFT (with a note)."""
    if action not in ("CONFIRM", "CHANGE_AMOUNT", "REJECT_DRAFT"):
        raise ValueError("action must be CONFIRM, CHANGE_AMOUNT or REJECT_DRAFT")
    if action == "CHANGE_AMOUNT" and (amount is None or amount < 0):
        raise ValueError("CHANGE_AMOUNT needs an amount of zero or more")
    if action == "REJECT_DRAFT" and not note.strip():
        raise ValueError("say why the draft is rejected")
    with _db() as con:
        row = con.execute("SELECT status, payable FROM cd_cases WHERE case_id=?", (case_id,)).fetchone()
        if not row or row[0] != "DRAFT":
            raise ValueError("case not found or already reviewed")
        status = "REJECTED_DRAFT" if action == "REJECT_DRAFT" else "CONFIRMED"
        final = amount if action == "CHANGE_AMOUNT" else row[1]
        con.execute("UPDATE cd_cases SET status=?, reviewer=?, note=?, final_payable=? WHERE case_id=?",
                    (status, reviewer, note[:300], final, case_id))
    return True
