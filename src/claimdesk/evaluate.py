"""Runs a system on the frozen test claims and scores it (specification section 10)."""
import json
import statistics
from collections import Counter, defaultdict

from claimcheck import config
from claimcheck.llm import llm

from . import data
from .baselines import single_agent, single_prompt
from .orchestrator import decide_claim
from .pacer import PACER
from .tools import billed_total

SYSTEMS = {
    "single_prompt": single_prompt,                                            # A
    "single_agent": single_agent,                                              # B
    "no_adjudicator": lambda c: decide_claim(c, system="no_adjudicator"),      # C
    "full": lambda c: decide_claim(c, system="full"),                          # D
    "full_no_coding": lambda c: decide_claim(c, disable=("coding",)),
    "full_no_fraud": lambda c: decide_claim(c, disable=("fraud",)),
}
CUT_RULES = {"R-NONMED", "R-ROOM-CAP", "R-PACKAGE", "R-CODE-PAIR"}


def cited_rules(res):
    out = set()
    for r in res.get("reasons") or []:
        for c in [r.get("rule")] + (r.get("cite") if isinstance(r.get("cite"), list) else [r.get("cite")]):
            if c:
                out.add(str(c))
    return out


def score(claim, gold, res):
    d_ok = res["decision"] == gold["decision"]
    if gold["decision"] in ("approve", "part_pay"):
        amount_ok = res["payable"] is not None and abs(float(res["payable"]) - gold["payable"]) <= 1
    elif gold["decision"] == "reject":
        amount_ok = d_ok and not res["payable"]
    else:
        amount_ok = None
    cited = cited_rules(res)
    if gold["decision"] in ("reject", "pend"):
        need = {r for r in gold["key_rules"] if r not in ("R-COST-SHARE",)}
    else:
        need = set(gold["key_rules"]) & CUT_RULES
    citation_ok = need <= cited and not res.get("bad_citations")
    total = billed_total(claim)
    return {
        "correct": d_ok and amount_ok is not False,
        "decision_ok": d_ok,
        "amount_ok": amount_ok,
        "citation_ok": citation_ok,
        "human_ok": res["needs_human"] if gold["needs_human"] else None,
        "false_flag": res["needs_human"] if not gold["needs_human"] else None,
        "fraud_caught": res["needs_human"] if gold["fraud_signal"] else None,
        "injection_followed": (res["decision"] == "approve" or (res["payable"] or 0) >= total - 1)
        if gold["case_type"] == "injection" else None,
        "failure": None if d_ok and amount_ok is not False else classify(gold, res),
    }


def classify(gold, res):
    """Failure taxonomy (section 10). Uses the facts the system worked from when it has them."""
    ctrl = res.get("control") or ""
    if ctrl:
        errors = " ".join(a.get("error") or "" for a in res.get("agents", []))
        return "endpoint_error" if "endpoint" in errors or ctrl == "endpoint_error" else \
            "json_failure" if "json" in errors or ctrl == "json_failed" else "specialist_failed"
    f, g = res.get("facts"), gold["facts"]
    if f is None:
        return "decision_wrong" if res["decision"] != gold["decision"] else "amount_wrong"
    if f["complete"] != g["complete"]:
        return "intake_miss"
    if f["excluded_rule"] != g["excluded_rule"] or f["wait_rule"] != g["wait_rule"]:
        return "policy_rule_miss"
    if f["duplicate_of"] != g["duplicate_of"] or bool(f["near_duplicate_of"]) != bool(g["near_duplicate_of"]):
        return "fraud_miss" if (g["duplicate_of"] or g["near_duplicate_of"]) else "fraud_false_alarm"
    if sorted(f["invalid_lines"]) != sorted(g["invalid_lines"]) or f["long_stay"] != g["long_stay"]:
        return "coding_miss"
    if f["covered_lines"] is not None and sorted(set(f["covered_lines"]) - set(f["invalid_lines"])) != \
            sorted(set(g["covered_lines"]) - set(g["invalid_lines"])):
        return "policy_lines_wrong"
    if g["near_duplicate_of"]:
        return "adjudicator_wrong_call"
    return "arithmetic"


def run_eval(system, run_no=1, limit=None, types=None, ids=None, progress=True, tag="claimdesk", split="test"):
    config.ensure_dirs()
    fn = SYSTEMS[system]
    golds = data.test_gold(split)
    claims = [c for c in data.test_claims(split)
              if (types is None or golds[c["claim_id"]]["case_type"] in types) and (ids is None or c["claim_id"] in ids)]
    claims = claims[:limit] if limit else claims
    out_path = config.RESULTS_DIR / f"{tag}_{system}_run{run_no}.jsonl"
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for n, c in enumerate(claims, 1):
            gold = golds[c["claim_id"]]
            hits0, waited0 = llm.rate_limit_hits, PACER.waited
            try:
                res = fn(c)
            except Exception as e:           # one bad claim never stops a run
                res = {"claim_id": c["claim_id"], "decision": "pend", "payable": None, "needs_human": True,
                       "reasons": [], "control": f"crash: {str(e)[:200]}", "agents": [], "tokens": 0,
                       "llm_calls": 0, "json_retries": 0, "latency_s": 0, "cost_usd": 0, "bad_citations": []}
            row = {"claim_id": c["claim_id"], "case_type": gold["case_type"], "gold": {k: gold[k] for k in
                   ("decision", "payable", "needs_human", "key_rules")}, "gold_facts": gold["facts"],
                   **score(c, gold, res),
                   "decision": res["decision"], "payable": res["payable"], "needs_human": res["needs_human"],
                   "control": res.get("control"), "tokens": res["tokens"], "llm_calls": res["llm_calls"],
                   "json_retries": res["json_retries"], "latency_s": res["latency_s"], "cost_usd": res["cost_usd"],
                   "rate_limit_hits": llm.rate_limit_hits - hits0, "paced_s": round(PACER.waited - waited0, 2),
                   "result": res}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            if progress and (n % 10 == 0 or n == len(claims)):
                print(f"  {system} run {run_no}: {n}/{len(claims)}, accuracy so far "
                      f"{sum(r['correct'] for r in rows) / n:.2f}", flush=True)
    summary = summarize(rows)
    (config.RESULTS_DIR / f"{tag}_{system}_run{run_no}_summary.json").write_text(json.dumps(summary, indent=1))
    return rows, summary


def agent_correct(role, finding, g):
    """Is one agent's own finding right? Judged before the evidence checks correct anything."""
    if finding is None:
        return False
    f = finding
    if role == "intake":
        return f["complete"] == g["complete"]
    if role == "policy":
        return (f["excluded_rule"] == g["excluded_rule"] and f["wait_rule"] == g["wait_rule"]
                and set(f["covered_lines"]) == set(g["covered_lines"]))
    if role == "coding":
        return set(f["invalid_lines"]) == set(g["invalid_lines"]) and f["long_stay"] == g["long_stay"]
    if role == "fraud":
        return (f["duplicate_of"] == g["duplicate_of"] and f["near_duplicate_of"] == g["near_duplicate_of"]
                and f["inflated"] == g["inflated"] and f["watchlist"] == g["watchlist"])
    if role == "adjudicator":
        return f["decision"] == ("pend" if g["near_same_procedure"] else "pay")
    return None


def scorecard(rows):
    """Per agent: how often its own finding was right, and what it cost."""
    acc, tokens, secs = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        for a in r["result"].get("agents", []):
            ok = agent_correct(a["role"], a["finding"], r["gold_facts"])
            if ok is not None:
                acc[a["role"]].append(ok)
            tokens[a["role"]].append(a["tokens"])
            secs[a["role"]].append(a["seconds"])
    return {role: {"runs": len(tokens[role]),
                   "finding_accuracy": round(sum(acc[role]) / len(acc[role]), 3) if acc[role] else None,
                   "avg_tokens": round(statistics.mean(tokens[role])), "avg_seconds": round(statistics.mean(secs[role]), 2)}
            for role in tokens}


def _rate(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return round(sum(bool(v) for v in vals) / len(vals), 3) if vals else None


def summarize(rows):
    n = len(rows)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["case_type"]].append(r["correct"])
    lat = sorted(r["latency_s"] for r in rows)
    dis = [r["correct"] for r in rows if r["case_type"] == "disagreement"]
    return {
        "n": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "decision_accuracy": _rate(rows, "decision_ok"),
        "amount_accuracy": _rate(rows, "amount_ok"),
        "citation_accuracy": _rate(rows, "citation_ok"),
        "fraud_recall": _rate(rows, "fraud_caught"),
        "false_flag_rate": _rate(rows, "false_flag"),
        "human_routing": _rate(rows, "human_ok"),
        "injection_followed": sum(bool(r["injection_followed"]) for r in rows),
        "disagreement_accuracy": round(sum(dis) / len(dis), 3) if dis else None,
        "accuracy_by_type": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_type.items())},
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
        "avg_llm_calls": round(statistics.mean(r["llm_calls"] for r in rows), 2),
        "avg_latency_s": round(statistics.mean(lat), 2),
        "p95_latency_s": lat[int(0.95 * (n - 1))],
        "avg_cost_usd": round(statistics.mean(r["cost_usd"] for r in rows), 6),
        "json_retries": sum(r["json_retries"] for r in rows),
        "rate_limit_hits": sum(r.get("rate_limit_hits", 0) for r in rows),
        "avg_paced_s": round(statistics.mean(r.get("paced_s", 0) for r in rows), 2),
        "controls": dict(Counter(r["control"].split(":")[0] for r in rows if r["control"])),
        "failure_taxonomy": dict(Counter(r["failure"] for r in rows if r["failure"])),
        "agent_scorecard": scorecard(rows),
    }


def resummarize(path, split="dev"):
    """Summary (with the agent scorecard) of a saved run file, e.g. one written before the scorecard existed."""
    golds = data.test_gold(split)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for r in rows:
        r.setdefault("gold_facts", golds[r["claim_id"]]["facts"])
    return summarize(rows)
