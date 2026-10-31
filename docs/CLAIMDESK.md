# ClaimDesk: build notes

ClaimDesk decides a submitted Asteria Health claim with a team of agents and a human reviewer.
Specification: https://claude.ai/artifact/NWvSYPJBZAhQ3ANFM2TVq8

## How one claim is decided

1. **Intake** (agent) checks the required documents. Missing → **pend** at once (no other agent runs).
2. **Policy, Coding and Fraud** (agents) run in parallel threads. At most 3 model calls run at once across all agents
   (`CD_CONCURRENCY`, the endpoint is rate limited). Each specialist replies with tool calls (up to 3 per step) or
   its finding, validated by Pydantic.
3. **Evidence checks** (code, `orchestrator.facts_from`): a finding is only used where a tool result backs it.
   - A waiting-period rejection needs a `waiting_period_checker` result with `covered: false`.
   - A network rejection needs a `get_hospital` result with `network: false`.
   - A duplicate or near-duplicate id must be one that `find_similar_claims` returned; its `exact` flag decides which.
   - Policy may only leave out `non_medical` lines; a medical line it leaves out is put back (members are never cut
     without a rule). Covering a non-medical line by mistake is NOT corrected: that error stays visible in the scores.
4. **Hard rules** (code, `rules.apply_rules`) decide the amount and most decisions.
5. **Adjudicator** (agent) runs only for an open question (a near duplicate): pay or pend. Anything else → pend.
6. **Letter writer** (agent) drafts the member letter; code checks every ₹ amount, the payable amount, the decision
   wording and the "draft" note; one revision, then a human sees the problems. The model letter check only warns.
7. The draft waits for a human: Confirm, Change amount, or Reject draft (with a note).

A specialist that fails (bad JSON twice, endpoint error, step cap, token budget) makes the claim **pend** for a
human; ClaimDesk never guesses.

## Claims uploaded as PDF

`src/claimdesk/pdf_claims.py` turns an uploaded PDF into the same claim dict the JSON claims use, so every agent,
rule and check runs unchanged.

1. A packet has a **claim form** page (ids, dates, diagnosis, one `Line N | category | amount | description` row
   per charge) and, when sent, an **ITEMISED BILL** page and a **DISCHARGE SUMMARY** page. The documents received
   are the pages found, so a packet without its bill pends at Intake like any other.
2. The claim form layout is read with rules (no model call). Any other layout (a hospital letter, say) goes to the
   model, which copies the fields into a Pydantic schema; the PDF text is marked as data, never instructions.
3. Every value is checked against the member, hospital, diagnosis and procedure tables (and room days x rate).
   The app shows what was read in an editable table and will not decide until the problems are fixed.
4. Bill descriptions still go through the same injection guard before any agent sees them.

Checks: all 183 dev and test claims survive PDF -> claim exactly; 5 hospital-letter PDFs read by the live model
had every id, date, code and amount right (only the descriptions were reworded). Scanned PDFs (images, no text)
are refused: OCR is not supported.

## Why the gold labels are trustworthy

`gen.py` builds each claim for a case type, computes its gold decision with the same tools and rules
(`rules.gold_decision`), and re-generates the claim if the result does not fit the case type. A test feeds a fake
model that plays every role correctly; the full system then matches all 150 gold labels. So any error in a live
run comes from the model, not from the plumbing.

## Dev and test

- `dev` (33 claims, CL2001-CL2033): tune prompts here.
- `test` (150 claims, CL1001-CL1150): official runs only. Tag `claims-eval-v1` before the first official run;
  any later change to the test claims is `claims-eval-v2`.

The first live check (12 claims) ran before the dev split existed, on the old test claims. Adding the dev split
changed the generator's random sequence, so all 150 test claims were regenerated: the current test set has not
been seen by anyone tuning prompts.

## Changes made after live runs

| Seen on the endpoint | Change |
| --- | --- |
| Agents repeated tool calls instead of finishing | The prompt lists "Already done (never repeat)" calls, as in ClaimCheck |
| Fraud ran out of tokens (2,396 of 2,000) | Budgets set to about calls × 800 (Intake 2,000, Coding 2,500, Fraud 3,000) |
| Adjudicator held an honest follow-up although the tool said `same_procedure: false` | Its input states both procedure codes side by side |
| Letter subject was the placeholder "short", then an invented claim id | Schema hint rewritten; the claim id is passed in |
| Intake said complete with the itemised bill missing | Prompt checks each document one by one, with an example |
| Policy rejected for a waiting period it never checked, and cut medicines as non-medical | Evidence checks (above) |
| Dev run 1 (69.7%): Policy rejected 9 payable claims with R-WAIT-PED. It called the waiting checker for diagnoses unrelated to the member's condition, and the tool answered "not covered" from the enrolment date alone | The tool now returns `applies` and says when no waiting period applies (a tool design fix, not a prompt trick) |
| Intake put its finding's fields at the top level, so the reply was read as empty | The step reader accepts a top-level finding |
| Adjudicator wrote "same procedure code" when the codes differed | Its input carries the tool's `same_procedure` flag |
| Hallucinated doc ids in citations (e.g. `BEN-...-nomeds`) | Prompt: cite a doc id only when copied from a tool's `sources` |

## Dev results

| Run | Accuracy | Notes |
| --- | --- | --- |
| Dev run 1 | 69.7% | 9 policy waiting-period errors, 1 adjudicator error; fraud recall 100%; 0 injections followed; 7,132 tokens and 40.5 s per claim |
| Dev run 2 | 90.9% | After the waiting-tool fix. 3 errors: 2 missed exclusions (Policy), 1 long stay (Coding). Disagreements 3/3, false flags 0%, injections followed 0. p95 72.1 s. Cache was on, so tokens are understated |

| Dev run 3 | 90.9% | Same as run 2 with the adaptive pacer, cache off: 7,879 tokens, 10.6 calls, 41.1 s average, p95 65.1 s; still 69 HTTP 429 replies |

## Single agent (B) against ClaimDesk (D), dev set, cache off

| Metric | B: one agent, 9 tools | D: ClaimDesk |
| --- | --- | --- |
| Accuracy | 27.3% | **90.9%** |
| Payable amount within ₹1 | 37.9% | 89.7% |
| Fraud cases sent to a human | 8.3% | 100% |
| Required human reviews routed | 13.0% | 91.3% |
| Honest claims wrongly flagged | 20% | 0% |
| Injections followed | 0 | 0 |
| Tokens per claim | 6,250 | 7,879 |
| Model calls per claim | 4.1 | 10.6 |
| Latency average / p95 | 13.3 s / 28.7 s | 41.1 s / 65.1 s |

Why B fails: with all nine tools in one prompt it took the short path every time. Over 33 claims it never called
`find_similar_claims`, `tariff_check` or `waiting_period_checker`, and never chose "pend" (19 part-pay,
11 approve, 3 reject). It missed every duplicate, waiting period and missing document. ClaimDesk gives each check
to an agent whose whole job is that check, so the check always happens. The cost is 26% more tokens and about
3× the latency.

## The pacer

Dev run 1 (no pacer, cache off) against dev run 3 (pacer at 30 requests a minute, cache off): average latency
40.5 s → 41.1 s, p95 74.6 s → 65.1 s. The pacer trims the worst waits but 69 replies were still HTTP 429, so the
endpoint's real limit is below 30 requests a minute. Next step: try `CD_RPM=15` on dev before the official runs.

Agent scorecard (each agent's own finding against the gold facts, before evidence checks):

| Agent | Dev run 1 | Dev run 2 |
| --- | --- | --- |
| Intake | 97% | 100% |
| Policy | 53% | 87% |
| Coding | 97% | 97% |
| Fraud | 93% | 93% |
| Adjudicator | 1 of 2 | 3 of 3 |

Re-print the scorecard of any saved run: `python scripts/summarize_claimdesk.py data/results/check_dev_full_run3.jsonl`.
