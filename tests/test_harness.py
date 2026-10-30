from claimcheck import evalset
from claimcheck.harness import run_eval


def test_oracle_answers_score_100_percent():
    for it in evalset.load_eval(types=["fact", "version", "cost_calc", "waiting_calc", "member_fact", "injection_doc"]):
        g = it["gold"]
        res = {"answer": {"status": "answered", "value": g["value"], "citations": g["doc_ids"], "answer_text": ""},
               "status": "answered", "control": None, "retrieved": g["doc_ids"], "tool_errors": 0}
        assert evalset.score_item(it, res)["correct"], it["id"]


def test_within_one_rupee_is_correct():
    it = evalset.load_eval(types=["cost_calc"])[0]
    res = {"answer": {"status": "answered", "value": it["gold"]["value"] + 1, "citations": [], "answer_text": ""},
           "control": None, "retrieved": [], "tool_errors": 0}
    assert evalset.score_item(it, res)["correct"]


def test_poisoned_answer_scores_wrong():
    it = evalset.load_eval(types=["injection_doc"])[0]
    res = {"answer": {"status": "answered", "value": 99999, "citations": [], "answer_text": ""},
           "control": None, "retrieved": [], "tool_errors": 0}
    assert not evalset.score_item(it, res)["correct"]


def test_harness_runs_on_guard_only_items():
    rows, s = run_eval("unit", "agent", types=["injection_user", "clarify"], progress=False)
    assert s["n"] == 10 and s["accuracy_by_type"] == {"injection_user": 1.0, "clarify": 1.0}


def test_guard_catches_8_of_10_refusals():
    rows, s = run_eval("unit_refuse", "agent", types=["refuse"], limit=8, progress=False)
    assert s["accuracy"] == 1.0
