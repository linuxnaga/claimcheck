# Changelog
## Unreleased: ClaimDesk (multi-agent claims adjudication)
- New `claimdesk` package: 6 agents (Intake, Policy, Coding, Fraud, Adjudicator, Letter writer) and a code
  orchestrator that runs Policy, Coding and Fraud in parallel under one shared limit of 3 model calls at a time.
- Seeded data: 60 members, 40 hospitals, 40 diagnoses, 120 procedure codes (fictional), 800 past claims,
  150 test claims (11 case types) and 33 dev claims, all with gold labels from the hard rules.
- Hard rules in code decide the amount; rejections must be backed by a tool result; medical lines can only be
  left out by Coding. Drafts wait for a human (Confirm, Change amount, Reject draft).
- Claims can be uploaded as PDF (`pdf_claims.py`, new dependencies `pypdf` and `fpdf2`). The claim form layout is
  read by rules; any other layout by the model. Documents received are the pages found. Every value is checked
  against the tables and shown for correction before deciding; scanned (image-only) PDFs are refused.
- Baselines A (one prompt) and B (one agent, all tools) for the ablation; reviewer app `app/claimdesk_app.py`.
- 19 new tests, including a fake model that plays every role correctly on all 150 claims.

## 1.0.0
- ClaimCheck built from the PolicyPilot starter: Asteria Health corpus (600 documents, amounts in ₹),
  15 members, 8 claims, 98-item evaluation set.
- Five tools: search_plan_docs, get_member_profile, cost_share_calculator, waiting_period_checker,
  create_appeal_draft (DRAFT only; a claims reviewer confirms in the app).
- Guard for medical, diagnosis, outcome-promise, other-member, legal, fraud and injection requests.
