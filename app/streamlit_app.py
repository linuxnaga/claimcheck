"""ClaimCheck web app (Streamlit). Run: streamlit run app/streamlit_app.py"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from claimcheck import config, tools
from claimcheck.agent import Memory, run_agent
from claimcheck.corpus_gen import TIERS, fmt_inr, load_members
from claimcheck.ratelimit import RateLimiter
from claimcheck.tracing import mask_pii

st.set_page_config(page_title="ClaimCheck", page_icon="🩺")

REVIEWER = "Claims reviewer (staff)"
GUEST = "Guest (not signed in)"
BREAKDOWN_LABELS = [("room_excess", "Room charges above the cap"), ("deductible_part", "Deductible"),
                    ("copay", "Co-payment"), ("member_pays", "You pay"), ("asteria_pays", "Asteria pays")]


@st.cache_resource
def get_limiter():
    return RateLimiter(config.RATE_LIMIT_PER_MIN)


@st.cache_resource
def get_memory():
    return Memory()


@st.cache_resource
def warm_up():
    from claimcheck import retrieval
    from claimcheck.llm import llm
    if config.EMBED_MODEL and not retrieval.index_ready(config.CHUNKING):
        retrieval.build_index(config.CHUNKING, llm.embed)   # first start only (needs the endpoint)
    tools.get_retriever()   # loads the search index once
    return True


def show_breakdown(b):
    rows = [(label, fmt_inr(b[k])) for k, label in BREAKDOWN_LABELS if k in b and b[k] is not None]
    if rows:
        st.table({"Item": [r[0] for r in rows], "Amount": [r[1] for r in rows]})
        st.caption("Estimate, not a claim decision.")


st.title("🩺 ClaimCheck")
st.caption("Answers come from Asteria Health plan documents and always show their sources. "
           "ClaimCheck cannot give medical or legal advice or share other members' data. All data is fictional.")

members = {m["member_id"]: m for m in load_members()}
choice = st.sidebar.selectbox(
    "Sign in as (demo users)", [GUEST] + list(members) + [REVIEWER],
    format_func=lambda x: x if x in (GUEST, REVIEWER) else
    f"{x} - {members[x]['name']} ({TIERS[members[x]['tier']]}, zone {members[x]['zone']})")
user_id = choice if choice in members else None
st.sidebar.info("Demo data is fictional. Do not type real personal or medical data.")

# ---------------- claims reviewer view: the human-in-the-loop boundary ----------------
if choice == REVIEWER:
    st.subheader("Draft appeals waiting for review")
    drafts = tools.list_draft_appeals()
    if not drafts:
        st.info("No draft appeals.")
    for d in drafts:
        with st.container(border=True):
            st.warning(f"DRAFT appeal #{d['case_id']} · member {d['member_id']} · claim {d['claim_id']}")
            st.write(d["reason"])
            c1, c2 = st.columns(2)
            if c1.button("Confirm and submit", key=f"ok{d['case_id']}"):
                tools.review_appeal(d["case_id"], "reviewer", "SUBMITTED")
                st.rerun()
            if c2.button("Reject", key=f"no{d['case_id']}"):
                tools.review_appeal(d["case_id"], "reviewer", "REJECTED")
                st.rerun()
    st.stop()

# ---------------- member / guest chat ----------------------------------------------------
if "sid" not in st.session_state:
    st.session_state.sid = uuid.uuid4().hex
    st.session_state.chat = []

try:
    warm_up()
except Exception as e:
    st.error(f"Search index not ready: {e}. Run scripts/build_data.py and scripts/build_index.py first.")
    st.stop()

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["text"])
        if m.get("res"):
            r = m["res"]
            if r["answer"].get("breakdown"):
                show_breakdown(r["answer"]["breakdown"])
            if r["answer"]["citations"]:
                st.caption("Sources: " + ", ".join(r["answer"]["citations"]))
            if r["status"] == "draft_created":
                st.warning(f"Appeal #{r['case_id']} is a DRAFT. A claims reviewer must confirm it before it is submitted.")
            with st.expander("How I got this (trace)"):
                st.write(f"Status: {r['status']} | stopped by: {r['control'] or 'nothing'} | steps: {r['steps']} | "
                         f"tool calls: {r['tool_calls']} | tokens: {r['tokens']} | {r['latency_s']} s")
                st.json(r["trace"])

question = st.chat_input("Ask about your health plan")
if question:
    question = question.strip()[:500]
    st.session_state.chat.append({"role": "user", "text": mask_pii(question)})
    if not get_limiter().allow(st.session_state.sid):
        st.session_state.chat.append({"role": "assistant",
                                      "text": "Too many requests. Please wait a minute and try again."})
    else:
        with st.spinner("Checking the plan documents..."):
            res = run_agent(question, user_id=user_id, session_id=st.session_state.sid, memory=get_memory())
        st.session_state.chat.append({"role": "assistant", "text": res["answer"]["answer_text"], "res": res})
    st.rerun()
