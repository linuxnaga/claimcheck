"""The two baselines for the ablation: A = one prompt with no tools, B = one agent with all nine tools.

Both see the same rules as ClaimDesk and answer with a DirectDecision; their amounts come from the model.
"""
import json
import time

from claimcheck import config, tools as cc_tools
from claimcheck.corpus_gen import fmt_inr
from claimcheck.llm import EndpointError, LLMError

from . import data
from .agents import STEP_FORMAT, AgentRun, Spec, _tools_text, chat_json, run_specialist
from .schemas import DirectDecision
from .tools import TOOLS, claim_summary, get_hospital, get_member_profile, HospitalArgs, MemberArgs

RULES_TEXT = "\n".join(f"- {k}: {v}" for k, v in data.RULES.items())
DECISION_TEXT = f"""Decision:
- reject: an exclusion (R-EXCL-COSMETIC, R-NETWORK-PREAUTH), a waiting period not served (R-WAIT-PED, R-WAIT-MAT) or a duplicate (R-DUP). payable 0.
- pend: a required document is missing (R-DOCS), or a near duplicate for the SAME procedure (R-NEAR-DUP). payable null.
- part_pay: something is cut (non_medical lines, room rent above the cap, a procedure above its package limit, a procedure not allowed for the diagnosis).
- approve: every line is paid in full before the deductible and co-payment.
payable = what Asteria pays after the room rent cap, package limits, deductible and co-payment.
needs_human is true for reject, pend, payable above {fmt_inr(data.HUMAN_REVIEW_ABOVE)}, a watch-list hospital, a procedure above twice its package limit, a long stay or a near duplicate.
Answer: {{"decision": "approve|part_pay|reject|pend", "payable": number or null, "needs_human": true/false, "reasons": [{{"text": "short", "cite": "rule id"}}]}}"""

SINGLE_PROMPT_SYSTEM = f"""ROLE: SINGLE PROMPT claims officer of Asteria Health (India).
Decide the claim from the information given. Do the sums yourself. Reply with ONLY one JSON object.
Rules:
{RULES_TEXT}
{DECISION_TEXT}"""

SINGLE_AGENT_SYSTEM = f"""ROLE: SINGLE AGENT claims officer of Asteria Health (India).
You decide the whole claim yourself with these tools:
{_tools_text(list(TOOLS))}
{STEP_FORMAT}
Rules:
{RULES_TEXT}
{DECISION_TEXT.replace("Answer:", "Finding:")}
Use cost_share_calculator for every amount (give it every covered line); do not do sums yourself."""

SINGLE_AGENT = Spec("single_agent", SINGLE_AGENT_SYSTEM, list(TOOLS), DirectDecision, 10, 15000)


def _result(claim, system, d, runs, t0, control=None):
    agents = [r.to_dict() for r in runs]
    tokens = sum(a["tokens"] for a in agents)
    return dict(claim_id=claim["claim_id"], system=system,
                decision=d.decision if d else "pend", payable=d.payable if d else None,
                needs_human=d.needs_human if d else True,
                reasons=[r.model_dump() for r in d.reasons] if d else [], control=control,
                bad_citations=[], agents=agents, tokens=tokens, llm_calls=sum(a["llm_calls"] for a in agents),
                json_retries=sum(a["json_retries"] for a in agents),
                cost_usd=round(tokens / 1e6 * config.PRICE_PER_1M_TOKENS, 6), latency_s=round(time.time() - t0, 2))


def single_prompt(claim):
    """System A: one retrieval of plan documents and one prompt. No tools, the model does the sums."""
    t0 = time.time()
    run = AgentRun("single_prompt")
    ctx = {"claim": claim}
    summary, _ = claim_summary(claim)
    m = get_member_profile(MemberArgs(member_id=claim["member_id"]), ctx)
    h = get_hospital(HospitalArgs(hospital_id=claim["hospital_id"]), ctx)
    docs = []
    try:
        res = cc_tools.search_plan_docs(cc_tools.SearchArgs(query="deductible co-payment room rent cap waiting period",
                                                            tier=m["tier"], zone=m["zone"]), ctx)
        docs = res["documents"]
    except Exception:
        pass
    user = (f"{summary}\n\nMember: {json.dumps(m)}\nHospital: {json.dumps(h)}\n\nPlan documents:\n"
            + "\n".join(f'<doc id="{d["doc_id"]}">\n{d["text"]}\n</doc>' for d in docs))
    control = None
    try:
        run.finding = chat_json([{"role": "system", "content": SINGLE_PROMPT_SYSTEM}, {"role": "user", "content": user}],
                                DirectDecision, run.usage, max_tokens=500)
    except LLMError as e:
        control = "endpoint_error" if isinstance(e, EndpointError) else "json_failed"
    run.steps, run.seconds = 1, time.time() - t0
    return _result(claim, "single_prompt", run.finding, [run], t0, control)


def single_agent(claim):
    """System B: one agent loop with all nine tools (ClaimCheck style)."""
    t0 = time.time()
    summary, _ = claim_summary(claim)
    run = run_specialist(SINGLE_AGENT, claim, summary)
    return _result(claim, "single_agent", run.finding, [run], t0, run.error)
