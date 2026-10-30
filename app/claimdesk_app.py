"""ClaimDesk reviewer app (Streamlit). Run: python -m streamlit run app/claimdesk_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from claimcheck.corpus_gen import fmt_inr
from claimdesk import data
from claimdesk.orchestrator import create_decision_draft, decide_claim, list_cases, review_case
from claimdesk.tools import claim_summary

st.set_page_config(page_title="ClaimDesk", page_icon="🗂️", layout="wide")
st.title("🗂️ ClaimDesk")
st.caption("A team of agents prepares each claim decision; a claims reviewer confirms it. All data is fictional.")


def rupees(x):
    return "-" if x is None else fmt_inr(x)


submit, queue = st.tabs(["Submit claim", "Reviewer queue"])

with submit:
    split = st.radio("Claims", ["dev", "test"], horizontal=True,
                     help="Use dev claims for demos; test claims are for the official evaluation.")
    claims = {c["claim_id"]: c for c in data.test_claims(split)}
    gold = data.test_gold(split)
    cid = st.selectbox("Claim", list(claims), format_func=lambda c: f"{c} ({gold[c]['case_type']})")
    claim = claims[cid]
    st.code(claim_summary(claim)[0], language=None)
    if st.button("Decide with ClaimDesk", type="primary"):
        with st.spinner("Intake, then Policy, Coding and Fraud in parallel..."):
            st.session_state["result"] = decide_claim(claim)
    res = st.session_state.get("result")
    if res and res["claim_id"] == cid:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Decision", res["decision"].replace("_", " "))
        c2.metric("Asteria pays", rupees(res["payable"]))
        c3.metric("Needs a human", "yes" if res["needs_human"] else "no")
        c4.metric("Tokens / time", f"{res['tokens']:,} / {res['latency_s']} s")
        if res["control"]:
            st.warning(f"Stopped early: {res['control']}. The claim is pended for a human.")
        st.subheader("Reasons")
        for r in res["reasons"]:
            st.write(f"- {r['text']}  `{', '.join(map(str, r['cite']))}`")
        if res.get("open_issue"):
            adj = next((a for a in res["agents"] if a["role"] == "adjudicator"), None)
            st.info(f"Open question: close to past claim {res['open_issue']['past_claim_id']}. "
                    f"Adjudicator: {adj['finding'] if adj and adj['finding'] else 'no answer, so the claim pends'}")
        if res["notes"]:
            st.warning("Evidence checks: " + "; ".join(res["notes"]))
        st.subheader("What each agent did")
        for a in res["agents"]:
            with st.expander(f"{a['role']}: {a['steps']} steps, {a['tokens']} tokens, {a['seconds']} s"
                             + (f", error {a['error']}" if a["error"] else "")):
                for c in a["calls"]:
                    st.write(f"`{c['tool']}` {c['args']}")
                    st.json(c["result"], expanded=False)
                st.write("Finding:")
                st.json(a["finding"])
        if res["letter"]:
            st.subheader("Draft letter")
            st.text(res["letter"]["subject"] + "\n\n" + res["letter"]["body"])
            if res["letter_problems"]:
                st.error("Letter checks failed: " + "; ".join(res["letter_problems"]))
        st.caption(f"Gold label: {gold[cid]['decision']} {rupees(gold[cid]['payable'])}")
        if st.button("Send to the reviewer queue"):
            st.success(f"Saved as draft case {create_decision_draft(res)}.")

with queue:
    reviewer = st.text_input("Reviewer name", "Claims reviewer")
    cases = list_cases("DRAFT")
    if not cases:
        st.write("No drafts waiting.")
    for case in cases:
        with st.expander(f"Case {case['case_id']}: claim {case['claim_id']}, {case['decision']}, "
                         f"{rupees(case['payable'])}" + (" (needs a human)" if case["needs_human"] else "")):
            st.text(case["letter"] or "(no letter)")
            note = st.text_input("Note", key=f"n{case['case_id']}")
            amount = st.number_input("Changed amount (₹)", min_value=0, step=100, key=f"a{case['case_id']}")
            b1, b2, b3 = st.columns(3)
            try:
                if b1.button("Confirm", key=f"c{case['case_id']}"):
                    review_case(case["case_id"], reviewer, "CONFIRM", note)
                    st.rerun()
                if b2.button("Change amount", key=f"m{case['case_id']}"):
                    review_case(case["case_id"], reviewer, "CHANGE_AMOUNT", note, amount)
                    st.rerun()
                if b3.button("Reject draft", key=f"r{case['case_id']}"):
                    review_case(case["case_id"], reviewer, "REJECT_DRAFT", note)
                    st.rerun()
            except ValueError as e:
                st.error(str(e))
