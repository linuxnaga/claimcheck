import json
import re

import pytest
from claimcheck import corpus_gen, tools as cc_tools
from claimcheck.llm import llm
from claimcheck.retrieval import Retriever, make_chunks
from claimcheck.tools import ToolError
from claimdesk import data, evaluate, gen, orchestrator, rules
from claimdesk import tools as cd_tools
from claimdesk.critic import check_letter
from claimdesk.schemas import Letter


@pytest.fixture(autouse=True)
def setup():
    llm.use_cache = False
    docs = corpus_gen.load_docs()
    cc_tools.set_retriever(Retriever(docs, make_chunks(docs, "sections")))   # BM25 only
    yield
    llm.use_cache = True


def claim_of(case, variant=None):
    gold = data.test_gold()
    for c in data.test_claims():
        g = gold[c["claim_id"]]
        if g["case_type"] == case and (variant is None or g["variant"] == variant):
            return c, g
    raise LookupError(case)


def ctx(claim):
    return {"claim": claim}


# ---------- data --------------------------------------------------------------------
def test_data_sizes_and_case_types():
    gold = data.test_gold()
    assert len(data.test_claims()) == 150 and len(gold) == 150
    assert len(data.past_claims()) == 800 and len(data.members()) == 60 and len(data.hospitals()) == 40
    assert len(data.procedures()) == 120 and len(data.diagnoses()) == 40
    counts = {}
    for g in gold.values():
        counts[g["case_type"]] = counts.get(g["case_type"], 0) + 1
    assert counts == dict(gen.CASE_COUNTS)


def test_data_is_deterministic():
    before = (data.claims_dir() / "test_gold.jsonl").read_text()
    gen.build_claims()
    assert (data.claims_dir() / "test_gold.jsonl").read_text() == before


def test_gold_labels_match_the_rules():
    gold = data.test_gold()
    for c in data.test_claims():
        res, facts = rules.gold_decision(c)
        assert res["decision"] == gold[c["claim_id"]]["decision"], c["claim_id"]
        assert res["payable"] == gold[c["claim_id"]]["payable"], c["claim_id"]


def test_claim_summary_removes_injected_instruction():
    c, _ = claim_of("injection", 0)
    text, flagged = cd_tools.claim_summary(c)
    assert flagged and "Ignore previous instructions" not in text and "REMOVED" in text


# ---------- tools ---------------------------------------------------------------------
def test_tools_only_see_the_claims_member_and_hospital():
    c, _ = claim_of("clean")
    other = next(m for m in data.members() if m != c["member_id"])
    with pytest.raises(ToolError):
        cd_tools.run_tool("get_member_profile", {"member_id": other}, ctx(c))
    with pytest.raises(ToolError):
        cd_tools.run_tool("get_hospital", {"hospital_id": "H99"}, ctx(c))
    with pytest.raises(ToolError):
        cd_tools.run_tool("tariff_check", {"line": 2}, ctx(c), allowed=["get_hospital"])
    prof = cd_tools.run_tool("get_member_profile", {"member_id": c["member_id"]}, ctx(c))
    assert "name" not in prof and "email" not in prof


def test_calculator_caps_and_disallowed_lines():
    c, g = claim_of("part_pay", 0)                          # a non_medical line
    full = cd_tools.compute_payable(c, [l["no"] for l in c["lines"]])
    cut = cd_tools.compute_payable(c, g["facts"]["covered_lines"])
    assert cut["disallowed_amount"] == c["lines"][-1]["amount"] and cut["asteria_pays"] <= full["asteria_pays"]
    c, _ = claim_of("part_pay", 2)                          # procedure above its package limit
    assert cd_tools.compute_payable(c, [1, 2, 3, 4])["package_excess"] > 0
    c, _ = claim_of("part_pay", 1)                          # room above the cap
    assert cd_tools.compute_payable(c, [1, 2, 3, 4])["room_excess"] > 0
    with pytest.raises(ToolError):
        cd_tools.compute_payable(c, [9])


def test_coding_fraud_and_waiting_tools():
    c, _ = claim_of("coding", 0)
    r = cd_tools.run_tool("check_code_pair", {"procedure_code": c["lines"][1]["code"],
                                              "diagnosis_code": c["diagnosis_code"], "stay_days": 1}, ctx(c))
    assert r["valid"] is False and "R-CODE-PAIR" in r["rules_broken"]
    c, _ = claim_of("coding", 9)
    r = cd_tools.run_tool("check_code_pair", {"procedure_code": c["lines"][1]["code"],
                                              "diagnosis_code": c["diagnosis_code"],
                                              "stay_days": cd_tools.stay_days(c)}, ctx(c))
    assert r["valid"] and r["long_stay"]
    c, _ = claim_of("duplicate")
    assert any(m["exact"] for m in cd_tools.run_tool("find_similar_claims", {}, ctx(c))["matches"])
    c, _ = claim_of("disagreement", 0)
    m = cd_tools.run_tool("find_similar_claims", {}, ctx(c))["matches"]
    assert m and not m[0]["exact"] and m[0]["same_procedure"]
    c, _ = claim_of("inflated")
    assert cd_tools.run_tool("tariff_check", {"line": 2}, ctx(c))["over_2x"]
    c, _ = claim_of("waiting")
    wt = "maternity" if data.diagnoses()[c["diagnosis_code"]]["type"] == "maternity" else "pre_existing"
    assert cd_tools.run_tool("waiting_period_checker", {"wait_type": wt}, ctx(c))["covered"] is False


# ---------- rules ----------------------------------------------------------------------
def test_rules_order_and_human_review():
    c, _ = claim_of("clean")
    f = rules.empty_facts()
    assert rules.apply_rules(c, dict(f, complete=False))["decision"] == "pend"
    assert rules.apply_rules(c, dict(f, excluded_rule="R-EXCL-COSMETIC", duplicate_of="P0001"))["reasons"][0]["rule"] \
        == "R-EXCL-COSMETIC"
    r = rules.apply_rules(c, dict(f, duplicate_of="P0001"))
    assert r["decision"] == "reject" and r["payable"] == 0 and r["needs_human"]
    r = rules.apply_rules(c, dict(f, near_duplicate_of="P0001"))
    assert r["open_issue"]["default"] == "pend"
    assert rules.resolve_open_issue(r, "maybe")["decision"] == "pend"       # anything odd keeps the safe default
    assert rules.resolve_open_issue(r, "pay")["decision"] == r["decision"]
    r = rules.apply_rules(c, dict(f, watchlist=True))
    assert r["needs_human"] and "R-WATCH" in r["human_reasons"]


# ---------- a fake model that answers every role correctly ---------------------------------
class Oracle:
    """Plays every agent perfectly using the gold facts, so the plumbing can be tested on all 150 claims."""

    def __init__(self, broken_roles=()):
        self.broken_roles = set(broken_roles)
        self.calls = 0

    def __call__(self, messages, temperature=0.0, max_tokens=600, json_mode=False):
        self.calls += 1
        sys_, user = messages[0]["content"], messages[-1]["content"]
        role = re.search(r"ROLE: (\w+)", sys_)
        role = role.group(1) if role else "CHECK"
        if role.lower() in self.broken_roles:
            return "not json at all", 10, 5
        cid = re.search(r"Claim (CL\d+)", user)
        g = data.test_gold()[cid.group(1)] if cid else None
        f = g["facts"] if g else None
        first = "(no tool results yet)" in user
        claim = next((c for c in data.test_claims() if cid and c["claim_id"] == cid.group(1)), None)
        reply = {"supported": True}
        if role == "INTAKE":
            reply = {"calls": [], "finding": {"complete": f["complete"], "missing_documents": f["missing_documents"]}}
        elif role == "POLICY":
            wt = "maternity" if data.diagnoses()[claim["diagnosis_code"]]["type"] == "maternity" else "pre_existing"
            reply = {"calls": [{"tool": "get_hospital", "args": {"hospital_id": claim["hospital_id"]}},
                               {"tool": "waiting_period_checker", "args": {"wait_type": wt}},
                               {"tool": "cost_share_calculator", "args": {"covered_lines": f["covered_lines"]}}],
                     "finding": None} if first else \
                {"calls": [], "finding": {"excluded_rule": f["excluded_rule"], "wait_rule": f["wait_rule"],
                                          "covered_lines": f["covered_lines"],
                                          "reasons": [{"text": "plan rules", "cite": "R-COST-SHARE"}]}}
        elif role == "CODING":
            reply = {"calls": [{"tool": "check_code_pair", "args": {"procedure_code": claim["lines"][1]["code"],
                     "diagnosis_code": claim["diagnosis_code"], "stay_days": cd_tools.stay_days(claim)}}],
                     "finding": None} if first else \
                {"calls": [], "finding": {"invalid_lines": f["invalid_lines"], "long_stay": f["long_stay"]}}
        elif role == "FRAUD":
            reply = {"calls": [{"tool": "find_similar_claims", "args": {}}, {"tool": "tariff_check", "args": {"line": 2}},
                               {"tool": "get_hospital", "args": {"hospital_id": claim["hospital_id"]}}],
                     "finding": None} if first else \
                {"calls": [], "finding": {"risk": "high" if f["duplicate_of"] or f["near_duplicate_of"] else "low",
                                          "duplicate_of": f["duplicate_of"], "near_duplicate_of": f["near_duplicate_of"],
                                          "inflated": f["inflated"], "watchlist": f["watchlist"],
                                          "signals": [{"text": "checked", "cite": "R-DUP"}]}}
        elif role == "ADJUDICATOR":
            reply = {"decision": "pend" if f["near_same_procedure"] else "pay", "reason": "compared procedures",
                     "cite": ["R-NEAR-DUP"]}
        elif role == "LETTER":
            facts = json.loads(user.split("Decision: ", 1)[1].split("\n")[0])
            amount = facts.get("asteria_pays", "")
            reply = {"subject": "Your claim", "body": f"Dear member, this is a draft decision ({facts['decision']}). "
                                                      f"Asteria pays {amount}. A claims reviewer will confirm it."}
        return json.dumps(reply), 100, 40


def test_full_system_with_a_perfect_model_matches_every_gold_label(monkeypatch):
    monkeypatch.setattr(llm, "chat_raw", Oracle())
    rows, s = evaluate.run_eval("full", tag="unit_cd", progress=False)
    assert s["accuracy"] == 1.0, [r["claim_id"] for r in rows if not r["correct"]]
    assert s["human_routing"] == 1.0 and s["false_flag_rate"] == 0 and s["injection_followed"] == 0
    assert s["citation_accuracy"] == 1.0
    assert all(not r["result"]["letter_problems"] for r in rows)
    card = s["agent_scorecard"]
    assert all(card[r]["finding_accuracy"] == 1.0 for r in ("intake", "policy", "coding", "fraud", "adjudicator"))
    assert card["adjudicator"]["runs"] == 6


def test_without_adjudicator_open_cases_pend(monkeypatch):
    monkeypatch.setattr(llm, "chat_raw", Oracle())
    rows, s = evaluate.run_eval("no_adjudicator", types=["disagreement"], tag="unit_cd", progress=False)
    assert all(r["decision"] == "pend" for r in rows) and s["disagreement_accuracy"] == 0.5


def test_failed_specialist_pends_the_claim(monkeypatch):
    monkeypatch.setattr(llm, "chat_raw", Oracle(broken_roles=["fraud"]))
    c, _ = claim_of("clean")
    res = orchestrator.decide_claim(c, write_letter=False)
    assert res["decision"] == "pend" and res["needs_human"] and res["control"] == "specialist_failed:fraud"


def test_disabled_specialist_is_skipped(monkeypatch):
    oracle = Oracle()
    monkeypatch.setattr(llm, "chat_raw", oracle)
    c, g = claim_of("duplicate")
    res = orchestrator.decide_claim(c, disable=("fraud",), write_letter=False)
    assert res["decision"] != "reject" and "fraud" not in [a["role"] for a in res["agents"]]


def test_hallucinated_duplicate_id_is_ignored(monkeypatch):
    c, _ = claim_of("clean")

    class Liar(Oracle):
        def __call__(self, messages, *a, **kw):
            text, p, q = super().__call__(messages, *a, **kw)
            if "ROLE: FRAUD" in messages[0]["content"] and '"finding": {' in text:
                d = json.loads(text)
                d["finding"]["duplicate_of"] = "P9999"
                text = json.dumps(d)
            return text, p, q

    monkeypatch.setattr(llm, "chat_raw", Liar())
    res = orchestrator.decide_claim(c, write_letter=False)
    assert res["decision"] == "approve" and any("P9999" in n for n in res["notes"])


def test_repeated_tool_call_is_refused_and_step_cap_ends_the_run(monkeypatch):
    c, _ = claim_of("clean")
    same = json.dumps({"calls": [{"tool": "find_similar_claims", "args": {}}], "finding": None})
    monkeypatch.setattr(llm, "chat_raw", lambda m, *a, **kw: (same, 50, 20))
    from claimdesk.agents import run_specialist
    run = run_specialist("fraud", c, cd_tools.claim_summary(c)[0])
    assert run.error == "step_cap" and run.steps == 4
    assert sum("already called" in json.dumps(x["result"]) for x in run.calls) == 3


def test_single_prompt_baseline_uses_the_models_numbers(monkeypatch):
    c, g = claim_of("clean")
    reply = json.dumps({"decision": "approve", "payable": 12345, "needs_human": False, "reasons": []})
    monkeypatch.setattr(llm, "chat_raw", lambda m, *a, **kw: (reply, 300, 50))
    res = evaluate.SYSTEMS["single_prompt"](c)
    sc = evaluate.score(c, g, res)
    assert res["payable"] == 12345 and sc["decision_ok"] and sc["amount_ok"] is False


# ---------- letter checks and the human review step ----------------------------------------
def test_letter_checks():
    c, _ = claim_of("clean")
    res, _ = rules.gold_decision(c)
    pay = corpus_gen.fmt_inr(res["payable"])
    ok = Letter(subject="s", body=f"Dear member, Asteria pays {pay}. A claims reviewer will confirm this draft.")
    assert check_letter(res, ok) == []
    bad = Letter(subject="s", body="Dear member, Asteria pays ₹99,999. Payment is guaranteed.")
    probs = " ".join(check_letter(res, bad))
    assert "not in the decision" in probs and "payable amount is missing" in probs and "promise" in probs
    rej = dict(res, decision="reject", payable=0, calc=None)
    assert check_letter(rej, Letter(subject="s", body="Dear member, your claim was not approved. A reviewer will "
                                                      "confirm this draft.")) == []


def test_drafts_need_a_human_to_confirm():
    c, _ = claim_of("clean")
    res, _ = rules.gold_decision(c)
    case_id = orchestrator.create_decision_draft(dict(res, claim_id=c["claim_id"], letter=None))
    assert any(x["case_id"] == case_id for x in orchestrator.list_cases("DRAFT"))
    with pytest.raises(ValueError):
        orchestrator.review_case(case_id, "reviewer", "REJECT_DRAFT")          # a reason is required
    orchestrator.review_case(case_id, "reviewer", "CHANGE_AMOUNT", amount=1000)
    with pytest.raises(ValueError):
        orchestrator.review_case(case_id, "reviewer", "CONFIRM")               # already reviewed


def test_dev_split_is_separate_from_test():
    dev, test = data.test_gold("dev"), data.test_gold()
    assert len(dev) == 33 and not set(dev) & set(test)
    assert {g["case_type"] for g in dev.values()} == {c for c, _ in gen.CASE_COUNTS}


def test_unbacked_policy_rules_are_dropped(monkeypatch):
    c, _ = claim_of("clean")

    class WrongPolicy(Oracle):
        def __call__(self, messages, *a, **kw):
            text, p, q = super().__call__(messages, *a, **kw)
            if "ROLE: POLICY" in messages[0]["content"] and '"finding": {' in text:
                d = json.loads(text)
                d["finding"].update(wait_rule="R-WAIT-PED", covered_lines=[1, 2])   # no checker call, lines cut
                text = json.dumps(d)
            return text, p, q

    monkeypatch.setattr(llm, "chat_raw", WrongPolicy())
    res = orchestrator.decide_claim(c, write_letter=False)
    assert res["decision"] == "approve" and len(res["notes"]) == 2


def test_finding_written_at_the_top_level_is_accepted(monkeypatch):
    c, _ = claim_of("clean")
    monkeypatch.setattr(llm, "chat_raw", lambda m, *a, **kw: (json.dumps({"calls": [], "complete": True}), 50, 20))
    from claimdesk.agents import run_specialist
    run = run_specialist("intake", c, cd_tools.claim_summary(c)[0])
    assert run.error is None and run.finding.complete


def test_waiting_checker_says_when_no_wait_applies():
    c, _ = claim_of("clean")
    r = cd_tools.run_tool("waiting_period_checker", {"wait_type": "pre_existing"}, ctx(c))
    assert r["applies"] is False and r["covered"] is True


def test_pacer_spaces_calls_and_backs_off_after_a_429():
    from claimdesk.pacer import Pacer
    p = Pacer(rpm=600)                       # 0.1 s apart
    p.wait(); p.wait()
    assert p.waited > 0.05
    p.done(True)
    assert p.gap >= 1.0                      # slowed down after a rate-limit reply
    for _ in range(50):
        p.done(False)
    assert p.gap == pytest.approx(p.min_gap)  # back to full speed after clean calls
