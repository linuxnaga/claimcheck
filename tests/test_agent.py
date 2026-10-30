import json
import pytest
from claimcheck import agent, config, corpus_gen, tools
from claimcheck.llm import llm
from claimcheck.retrieval import Retriever, make_chunks


class Scripted:
    """Fake model: returns queued replies for planner / writer / critic calls."""
    def __init__(self, planner=(), writer=(), tokens=(100, 50)):
        self.planner, self.writer, self.tokens = list(planner), list(writer), tokens
        self.calls = 0
        self.planner_prompts = []

    def __call__(self, messages, temperature=0.0, max_tokens=600, json_mode=False):
        self.calls += 1
        sys = messages[0]["content"]
        if "PLANNER" in sys:
            self.planner_prompts.append(messages[1]["content"])
            reply = self.planner.pop(0) if self.planner else json.dumps({"action": "final"})
        elif "WRITER" in sys:
            reply = self.writer.pop(0)
        else:
            reply = json.dumps({"supported": True})
        return reply if isinstance(reply, str) else json.dumps(reply), *self.tokens


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    llm.use_cache = False
    docs = corpus_gen.load_docs()
    tools.set_retriever(Retriever(docs, make_chunks(docs, "sections")))   # BM25 only
    monkeypatch.setattr(config, "TOKEN_BUDGET", 12000)
    yield
    llm.use_cache = True


DOC = "BEN-SILVER-B-2025-deductible"
SEARCH = {"action": "search_plan_docs",
          "args": {"query": "annual deductible", "tier": "silver", "zone": "B", "topic": "deductible"}}
FINAL = {"action": "final"}
Q = "What is the annual deductible on the Silver plan in zone B?"


def value(doc):
    return next(f for f in corpus_gen.load_facts() if f["doc_id"] == doc)["value"]


def good_answer(doc=DOC):
    return {"status": "answered", "answer_text": "ok", "value": value(doc), "unit": "INR_per_year",
            "citations": [doc]}


def run(fake, monkeypatch, q=Q, user=None):
    monkeypatch.setattr(llm, "chat_raw", fake)
    return agent.run_agent(q, user_id=user)


def test_happy_path(monkeypatch):
    res = run(Scripted([SEARCH, FINAL], [good_answer()]), monkeypatch)
    assert res["status"] == "answered" and res["control"] is None
    assert res["tool_calls"] == 1 and res["answer"]["citations"] == [DOC]


def test_malformed_json_is_retried(monkeypatch):
    res = run(Scripted(["not json at all", SEARCH, FINAL], [good_answer()]), monkeypatch)
    assert res["json_retries"] >= 1 and res["status"] == "answered"


def test_malformed_json_three_times_escalates(monkeypatch):
    res = run(Scripted(["bad", "bad", "bad"], []), monkeypatch)
    assert res["status"] == "escalated" and res["control"] == "json_failed"


def test_loop_detected(monkeypatch):
    res = run(Scripted([SEARCH, SEARCH, SEARCH], [good_answer()]), monkeypatch)
    assert res["control"] == "loop_detected"
    kinds = [t["type"] for t in res["trace"]]
    assert kinds.index("repeat_blocked") < kinds.index("loop_detected")   # first repeat is only refused


def test_single_repeat_is_refused_not_fatal(monkeypatch):
    res = run(Scripted([SEARCH, SEARCH, FINAL], [good_answer()]), monkeypatch)
    assert res["status"] == "answered" and res["control"] is None
    assert any(t["type"] == "repeat_blocked" for t in res["trace"])


def test_critic_rejects_invented_citation(monkeypatch):
    bad = dict(good_answer(), citations=["BEN-FAKE-Z-2025-x"])
    res = run(Scripted([SEARCH, FINAL, FINAL], [bad, good_answer()]), monkeypatch)
    assert res["status"] == "answered" and res["answer"]["citations"] == [DOC]
    assert any(t["type"] == "critic" and t["problems"] for t in res["trace"])


def test_wrong_number_is_caught(monkeypatch):
    bad = dict(good_answer(), value=77777)
    res = run(Scripted([SEARCH, FINAL] + [FINAL] * 6, [bad] * 5), monkeypatch)
    assert res["control"] == "critic_failed" and res["status"] == "escalated"
    assert "helpdesk" in res["answer"]["answer_text"]


def test_superseded_version_is_rejected():
    """Second line of defence: even if a 2024 document reaches the writer, the critic rejects citing it."""
    old = "BEN-SILVER-B-2024-deductible"
    state = agent.State(Q, None, [])
    state.retrieved[old] = {"doc_id": old, "tier": "silver", "zone": "B", "year": "2024", "status": "SUPERSEDED",
                            "text": "The member pays the first ₹%s" % value(old)}
    problems = agent.rule_critic(agent.Answer(**good_answer(old)), state)
    assert any("superseded" in p for p in problems)
    state.question = Q.rstrip("?") + " in 2024?"
    assert not any("superseded" in p for p in agent.rule_critic(agent.Answer(**good_answer(old)), state))

def test_step_cap_escalates(monkeypatch):
    plans = [{"action": "search_plan_docs", "args": {"query": f"q{i}"}} for i in range(15)]
    res = run(Scripted(plans, [good_answer()]), monkeypatch)
    assert res["control"] == "step_cap" and res["steps"] == config.MAX_STEPS
    assert res["status"] == "escalated"


def test_token_budget(monkeypatch):
    monkeypatch.setattr(config, "TOKEN_BUDGET", 500)
    plans = [{"action": "search_plan_docs", "args": {"query": f"q{i}"}} for i in range(15)]
    res = run(Scripted(plans, [good_answer()], tokens=(400, 200)), monkeypatch)
    assert res["control"] == "budget_exceeded"


def test_tool_error_is_reported_to_planner(monkeypatch):
    bad = {"action": "search_plan_docs", "args": {"query": "x", "zone": "Mars"}}
    fake = Scripted([bad, SEARCH, FINAL], [good_answer()])
    res = run(fake, monkeypatch)
    assert res["tool_errors"] == 1 and res["status"] == "answered"


def test_poisoned_doc_is_neutralised(monkeypatch):
    res = run(Scripted([SEARCH, FINAL], [good_answer()]), monkeypatch)
    assert "NOTICE-INJ1" in res["retrieved"] and res["injection_flags"] >= 1


def test_injected_value_is_rejected(monkeypatch):
    bad = dict(good_answer(), value=99999, citations=["NOTICE-INJ1"])
    res = run(Scripted([SEARCH, FINAL] + [FINAL] * 6, [bad] * 5), monkeypatch)
    assert res["answer"]["value"] != 99999


def test_cost_needs_calculator(monkeypatch):
    q = "I have a ₹1,20,000 hospital bill with 3 days in a ₹9,000 per day room. How much will I pay?"
    guess = {"status": "answered", "answer_text": "About ₹17,500", "value": 17500, "unit": "INR",
             "citations": [DOC]}
    profile = {"action": "get_member_profile", "args": {"member_id": "M003"}}
    calc = {"action": "cost_share_calculator",
            "args": {"member_id": "M003", "bill_amount": 120000, "room_days": 3, "room_rate_charged": 9000}}
    r = tools.run_tool("cost_share_calculator", calc["args"], {"user_id": "M003"})
    right = {"status": "answered", "answer_text": "estimate", "value": r["member_pays"], "unit": "INR",
             "breakdown": {k: r[k] for k in ("room_excess", "deductible_part", "copay", "member_pays", "asteria_pays")},
             "citations": r["sources"]}
    res = run(Scripted([profile, SEARCH, FINAL, calc, FINAL], [guess, right]), monkeypatch, q=q, user="M003")
    first_critic = next(t for t in res["trace"] if t["type"] == "critic")
    assert any("cost_share_calculator" in p for p in first_critic["problems"])
    assert res["status"] == "answered" and res["answer"]["value"] == r["member_pays"]


def test_no_outcome_promise(monkeypatch):
    promise = dict(good_answer(), answer_text="The deductible is ₹17,500 and your claim will be approved.")
    res = run(Scripted([SEARCH, FINAL, FINAL], [promise, good_answer()]), monkeypatch)
    first_critic = next(t for t in res["trace"] if t["type"] == "critic")
    assert any("promise" in p for p in first_critic["problems"])


def test_appeal_creates_draft(monkeypatch):
    call = {"action": "create_appeal_draft", "args": {"member_id": "M003", "claim_id": "C0031",
                                                      "reason": "itemised bill now available"}}
    reply = {"status": "draft_created", "answer_text": "Draft appeal saved; a claims reviewer must confirm it."}
    res = run(Scripted([call, FINAL], [reply]), monkeypatch,
              q="Please appeal my rejected claim C0031, the bill is itemised now.", user="M003")
    assert res["status"] == "draft_created" and res["case_id"] and res["answer"]["case_id"] == res["case_id"]
    assert any(d["case_id"] == res["case_id"] for d in tools.list_draft_appeals())


def test_guard_blocks_without_calling_model(monkeypatch):
    fake = Scripted()
    res = run(fake, monkeypatch, q="Should I have the knee surgery now or wait?")
    assert res["status"] == "refused" and fake.calls == 0


def test_guard_asks_back_without_model(monkeypatch):
    fake = Scripted()
    res = run(fake, monkeypatch, q="What is my room rent cap?")
    assert res["status"] == "clarify" and res["answer"]["follow_up_question"] and fake.calls == 0


def test_memory_keeps_turns(monkeypatch, tmp_path):
    mem = agent.Memory(tmp_path / "m.db")
    monkeypatch.setattr(llm, "chat_raw", Scripted([SEARCH, FINAL], [good_answer()]))
    agent.run_agent(Q, session_id="s1", memory=mem)
    assert len(mem.last("s1")) == 1
    doc_c = "BEN-SILVER-C-2025-deductible"
    search_c = {"action": "search_plan_docs", "args": dict(SEARCH["args"], zone="C")}
    fake = Scripted([search_c, FINAL], [good_answer(doc_c)])
    monkeypatch.setattr(llm, "chat_raw", fake)
    res = agent.run_agent("And for zone C?", session_id="s1", memory=mem)
    assert Q in fake.planner_prompts[0]          # the planner sees the previous turn
    assert res["status"] == "answered" and len(mem.last("s1")) == 2


def test_indian_numbers_are_parsed():
    assert agent.numbers_in("covered up to ₹1,20,000 per year and 15%") == {120000.0, 15.0}


def test_endpoint_failure_is_not_called_json_failure(monkeypatch):
    from claimcheck.llm import EndpointError

    def down(*a, **k):
        raise EndpointError("endpoint call failed: timeout")
    res = run(down, monkeypatch)
    assert res["status"] == "escalated" and res["control"] == "endpoint_error"


def test_unasked_2024_filter_is_removed(monkeypatch):
    wrong_year = {"action": "search_plan_docs", "args": dict(SEARCH["args"], year="2024")}
    res = run(Scripted([wrong_year, FINAL], [good_answer()]), monkeypatch)
    assert any(t["type"] == "year_filter_removed" for t in res["trace"])
    assert res["status"] == "answered" and DOC in res["retrieved"]
