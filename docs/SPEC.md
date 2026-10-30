# ClaimCheck specification

**Status: DRAFT** (change to APPROVED on Day 2, then commit)

Team: [team name] · Started from the PolicyPilot starter (say so in the report).

The full, living specification is the shared doc: https://claude.ai/artifact/9GQuvGvNvzp8DTvxEsRuCt
On Day 2, export it as Markdown and paste it over this file so the approved version is in Git.

## What ClaimCheck does
Answers health-plan benefit questions for members of the fictional insurer **Asteria Health** (India, amounts in ₹),
citing a plan document for every number. It estimates what a member pays on a bill (cost-share calculator),
checks waiting periods, and drafts claim appeals that a claims reviewer must confirm.

## What it refuses
Medical advice, diagnosis, promises about claim outcomes, other members' data, legal advice, fraud help,
off-topic requests, and prompt injection.

Three refusal examples added by the team:
1. [ ]
2. [ ]
3. [ ]

## When it asks back
A benefit question with no plan tier or zone and nobody signed in; a cost question with no bill amount.

## Correct answer
Code in `src/claimcheck/evalset.py` scores every answer. A number is correct when it equals the gold value
(within ₹1 for rupee amounts) and the status is `answered`. Refusal, clarification and appeal items are
scored on their status.

## Evaluation set (frozen as tag `eval-v1`)
| Type | Count |
| --- | --- |
| fact | 37 |
| cost_calc | 12 |
| waiting_calc | 6 |
| member_fact | 6 |
| version | 8 |
| injection_doc | 6 |
| injection_user | 5 |
| refuse | 10 |
| clarify | 5 |
| appeal | 3 |
| **Total** | **98** |

## Numeric targets (set on Day 4, after the baseline)
| Metric | Target |
| --- | --- |
| Overall accuracy | [ ] |
| Accuracy on cost_calc + waiting_calc | [ ] |
| Accuracy on refuse + injection | [ ] |
| Citation accuracy | [ ] |
| recall@5, best setup | [ ] |
| p95 latency | [ ] |
