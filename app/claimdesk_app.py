"""ClaimDesk reviewer app (Streamlit). Run: python -m streamlit run app/claimdesk_app.py"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from claimcheck.corpus_gen import fmt_inr
from claimdesk import data
from claimdesk.pdf_claims import CATEGORIES, check_claim, normalise, read_claim_pdf
from claimdesk.orchestrator import create_decision_draft, decide_claim, list_cases, review_case
from claimdesk.tools import claim_summary

st.set_page_config(page_title="ClaimDesk", page_icon="🗂️", layout="wide")
st.title("🗂️ ClaimDesk")
st.caption("A team of agents prepares each claim decision; a claims reviewer confirms it. All data is fictional.")


def rupees(x):
    return "-" if x is None else fmt_inr(x)


submit, queue = st.tabs(["Submit claim", "Reviewer queue"])

def show_result(res, gold_row):
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
    if gold_row:
        st.caption(f"Gold label: {gold_row['decision']} {rupees(gold_row['payable'])}")
    if st.button("Send to the reviewer queue"):
        st.success(f"Saved as draft case {create_decision_draft(res)}.")


def review_extracted(claim, key):
    """Lets the user correct what was read from the PDF. Returns the corrected claim."""
    c = dict(claim)
    a, b, d = st.columns(3)
    c["claim_id"] = a.text_input("Claim ID", c["claim_id"], key=f"{key}id")
    c["member_id"] = b.text_input("Member ID", c["member_id"], key=f"{key}m")
    c["hospital_id"] = d.text_input("Hospital ID", c["hospital_id"], key=f"{key}h")
    c["admission_date"] = a.text_input("Admission date", c.get("admission_date") or "", key=f"{key}ad")
    c["discharge_date"] = b.text_input("Discharge date", c.get("discharge_date") or "", key=f"{key}dd")
    c["diagnosis_code"] = d.text_input("Diagnosis code", c["diagnosis_code"], key=f"{key}dx")
    types = ["emergency", "planned"]
    c["admission_type"] = a.selectbox("Admission type", types, key=f"{key}t",
                                      index=types.index(c["admission_type"]) if c.get("admission_type") in types else 0)
    c["preauth"] = b.checkbox("Pre-authorised", c["preauth"], key=f"{key}p")
    c["documents"] = d.multiselect("Documents received", list(data.REQUIRED_DOCS), c["documents"], key=f"{key}docs")
    cols = ["category", "description", "code", "days", "rate", "amount"]
    rows = st.data_editor([{k: l.get(k) for k in cols} for l in c["lines"]], num_rows="dynamic", key=f"{key}lines",
                          column_config={"category": st.column_config.SelectboxColumn(options=list(CATEGORIES))})
    c["lines"] = [edited_line(r) for r in rows if r.get("category")]
    return normalise(c)


def edited_line(r):
    """One edited bill row: code only on a procedure line, days and rate only on a room line, no empty cells."""
    only = {"code": "procedure", "days": "room", "rate": "room"}
    return {k: v for k, v in r.items()
            if v not in (None, "") and v == v and only.get(k, r["category"]) == r["category"]}


with submit:
    source = st.radio("Claim source", ["Upload PDF", "Dev claims", "Test claims"], horizontal=True,
                      help="Dev claims are for demos; test claims are for the official evaluation.")
    claim, gold_row = None, None
    if source == "Upload PDF":
        up = st.file_uploader("Claim PDF (claim form, itemised bill and discharge summary pages)", type="pdf")
        st.caption("Sample PDFs: run `python scripts/make_claim_pdfs.py`, then upload one from data/claims/pdf/dev/.")
        if up is not None:
            raw = up.getvalue()
            key = "pdf" + hashlib.sha1(raw).hexdigest()[:10]
            if st.session_state.get("pdf_key") != key:
                with st.spinner("Reading the PDF..."):
                    st.session_state["pdf_read"] = read_claim_pdf(raw)
                st.session_state["pdf_key"] = key
            read = st.session_state["pdf_read"]
            if read["claim"] is None:
                st.error("; ".join(read["problems"]))
            else:
                st.success("Read with the claim form rules." if read["method"] == "form" else
                           "Not in the claim form layout: read by the model. Check every value below.")
                with st.expander("What was read (correct anything wrong)", expanded=read["method"] == "model"
                                 or bool(read["problems"])):
                    claim = review_extracted(read["claim"], key)
                with st.expander("PDF text"):
                    st.text(read["text"][:5000])
                problems = check_claim(claim)
                if problems:
                    st.error("Fix before deciding: " + "; ".join(problems))
                    claim = None
                if claim:      # a packet made from a dev or test claim still shows its gold label
                    gold_row = {**data.test_gold("test"), **data.test_gold("dev")}.get(claim["claim_id"])
    else:
        split = "dev" if source == "Dev claims" else "test"
        claims = {c["claim_id"]: c for c in data.test_claims(split)}
        gold = data.test_gold(split)
        cid = st.selectbox("Claim", list(claims), format_func=lambda c: f"{c} ({gold[c]['case_type']})")
        claim, gold_row = claims[cid], gold[cid]
    if claim:
        st.code(claim_summary(claim)[0], language=None)
        if st.button("Decide with ClaimDesk", type="primary"):
            with st.spinner("Intake, then Policy, Coding and Fraud in parallel..."):
                st.session_state["result"] = decide_claim(claim)
                st.session_state["result_claim"] = claim
        res = st.session_state.get("result")
        if res and st.session_state.get("result_claim") == claim:
            show_result(res, gold_row)

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
