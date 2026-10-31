"""Writes claim PDFs to upload in the ClaimDesk app.

  python scripts/make_claim_pdfs.py                  # dev claims, claim form layout -> data/claims/pdf/dev/
  python scripts/make_claim_pdfs.py --split test     # test claims (official evaluation only)
  python scripts/make_claim_pdfs.py --free-layout    # also a few hospital-letter PDFs, read by the model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimdesk import data
from claimdesk.pdf_claims import claim_to_pdf, rs


def letter_pdf(claim):
    """The same claim written as a hospital covering letter: no form fields, so the app needs the model to read it."""
    from fpdf import FPDF

    m, h = data.members()[claim["member_id"]], data.hospitals()[claim["hospital_id"]]
    dx = data.diagnoses()[claim["diagnosis_code"]]
    pdf = FPDF(format="A4")
    pdf.add_page()
    pdf.set_font("Helvetica", "", 11)
    charges = "; ".join(
        f"room rent for {l['days']} days at {rs(l['rate'])} per day ({rs(l['amount'])})" if l["category"] == "room" else
        f"procedure code {l['code']}, {l['description']} ({rs(l['amount'])})" if l["category"] == "procedure" else
        f"{l['category'].replace('_', '-')} charges: {l['description']} ({rs(l['amount'])})" for l in claim["lines"])
    docs = " and ".join(d.replace("_", " ") for d in claim["documents"]) or "no supporting documents"
    text = (f"To the Claims Team, Asteria Health\n\n"
            f"{h['name']} ({h['city']}, our hospital id {claim['hospital_id']}) submits claim reference "
            f"{claim['claim_id']} for {m['name']}, member number {claim['member_id']}. The patient was admitted on "
            f"{claim['admission_date']} as a{'n emergency' if claim['admission_type'] == 'emergency' else ' planned'} "
            f"case and discharged on {claim['discharge_date']}. The admission was "
            f"{'pre-authorised' if claim['preauth'] else 'not pre-authorised'}. Diagnosis: {dx['name']}, "
            f"code {dx['code']}.\n\nCharges: {charges}.\n\nEnclosed: {docs}.\n\nBilling department")
    pdf.multi_cell(180, 7, text)
    return bytes(pdf.output())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--free-layout", action="store_true")
    a = ap.parse_args()
    out = data.claims_dir() / "pdf" / a.split
    out.mkdir(parents=True, exist_ok=True)
    claims = data.test_claims(a.split)
    for c in claims:
        (out / f"{c['claim_id']}.pdf").write_bytes(claim_to_pdf(c))
    print(f"wrote {len(claims)} claim form PDFs to {out}")
    if a.free_layout:
        for c in claims[:5]:
            (out / f"{c['claim_id']}_letter.pdf").write_bytes(letter_pdf(c))
        print(f"wrote 5 hospital-letter PDFs ({claims[0]['claim_id']}_letter.pdf ...)")


if __name__ == "__main__":
    main()
