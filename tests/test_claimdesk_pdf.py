import json
import sys
from pathlib import Path

import pytest
from claimcheck.llm import llm
from claimdesk import data, orchestrator
from claimdesk import pdf_claims as P
from claimdesk.tools import claim_summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_claim_pdfs import letter_pdf  # noqa: E402


@pytest.fixture(autouse=True)
def no_cache():
    llm.use_cache = False
    yield
    llm.use_cache = True


def gold_claim(case):
    gold = data.test_gold("dev")
    return next(c for c in data.test_claims("dev") if gold[c["claim_id"]]["case_type"] == case)


def test_every_claim_survives_the_round_trip_through_a_pdf():
    for split in ("dev", "test"):
        for c in data.test_claims(split):
            r = P.read_claim_pdf(P.claim_to_pdf(c), use_model=False)
            assert r["method"] == "form" and r["problems"] == [] and r["claim"] == c, c["claim_id"]


def test_documents_are_the_pages_found():
    c = gold_claim("missing_info")
    assert set(c["documents"]) != set(data.REQUIRED_DOCS)
    r = P.read_claim_pdf(P.claim_to_pdf(c), use_model=False)
    assert r["claim"]["documents"] == c["documents"] and r["problems"] == []   # Intake, not the reader, pends it


def test_injected_bill_text_gets_the_same_guard_as_a_json_claim():
    c = gold_claim("injection")
    r = P.read_claim_pdf(P.claim_to_pdf(c), use_model=False)
    assert r["claim"]["lines"] == c["lines"]
    assert claim_summary(r["claim"]) == claim_summary(c)       # the same guard runs as for a JSON claim


def test_checks_catch_wrong_values():
    c = json.loads(json.dumps(data.test_claims("dev")[0]))
    c.update(member_id="M999", discharge_date="2020-01-01")
    room = next(l for l in c["lines"] if l["category"] == "room")
    room["amount"] += 500
    problems = " | ".join(P.check_claim(P.normalise(c)))
    assert "unknown member" in problems and "before the admission" in problems and "days x" in problems


def test_unreadable_and_empty_pdfs_are_refused():
    assert "could not read" in P.read_claim_pdf(b"not a pdf")["problems"][0]
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    assert "no text" in P.read_claim_pdf(bytes(pdf.output()))["problems"][0]


def test_other_layouts_are_read_by_the_model(monkeypatch):
    c = data.test_claims("dev")[0]
    reply = {k: c[k] for k in ("claim_id", "member_id", "hospital_id", "admission_type", "preauth",
                               "admission_date", "discharge_date", "diagnosis_code")}
    reply["documents"] = []                     # wrong on purpose: the page check must not override a letter
    reply["lines"] = [{k: v for k, v in l.items() if k != "no"} for l in c["lines"]]
    seen = {}

    def fake(messages, *a, **kw):
        seen["prompt"] = messages[-1]["content"]
        return json.dumps(reply), 100, 100
    monkeypatch.setattr(llm, "chat_raw", fake)
    r = P.read_claim_pdf(letter_pdf(c))
    assert r["method"] == "model" and r["problems"] == []
    assert c["member_id"] in seen["prompt"]      # the PDF text reached the model as data
    assert {k: v for k, v in r["claim"].items() if k != "documents"} == {k: v for k, v in c.items() if k != "documents"}


def test_a_claim_without_an_id_gets_a_stable_one():
    c = {k: v for k, v in data.test_claims("dev")[0].items() if k != "claim_id"}
    a, b = P.normalise(c), P.normalise(c)
    assert a["claim_id"] == b["claim_id"] and a["claim_id"].startswith("UP")


def test_a_pdf_claim_is_decided_like_the_json_claim(monkeypatch):
    from test_claimdesk import Oracle
    monkeypatch.setattr(llm, "chat_raw", Oracle())
    gold = data.test_gold()                     # the fake model knows the test split
    c = next(c for c in data.test_claims() if gold[c["claim_id"]]["case_type"] == "part_pay")
    from_pdf = P.read_claim_pdf(P.claim_to_pdf(c), use_model=False)["claim"]
    a, b = orchestrator.decide_claim(from_pdf), orchestrator.decide_claim(c)
    assert (a["decision"], a["payable"]) == (b["decision"], b["payable"])
