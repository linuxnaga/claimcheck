# ClaimCheck learning notes

What we have measured so far, what it means, and the file that proves it. Every number below comes from a
saved results file; quote the file name when you use a number in the report.

## Day 0: the endpoint

| Fact | Value | Evidence |
| --- | --- | --- |
| Chat model | `Qwen/Qwen2.5-7B-Instruct-AWQ` (7B, 4-bit AWQ) | `scripts/smoke_test.py` output |
| JSON mode supported | **True** | smoke test |
| Embeddings endpoint | **None.** `/v1/embeddings` and `/v1/models` both return 404 | smoke test, `GET /v1/models` |
| Embedding model used | `BAAI/bge-small-en-v1.5`, run locally (`EMBED_PROVIDER=local`), mentor-approved | `data/index/sections/index_meta.json` |
| Embedding size | **384** | smoke test |
| Rate limiting | The endpoint returns HTTP 429 when calls come too fast | `results/check_run1.jsonl` traces |

**Say in the report:** the supplied endpoint serves chat only, so embeddings run locally with the same
open model. No model weights are trained or changed.

## Week 1: prompting strategies (`results/week1_strategies.csv`)

| Strategy | Fact accuracy (30) | Bill accuracy, no calculator (8) | Tokens per fact question |
| --- | --- | --- | --- |
| closed_book (baseline) | 10% | 0% | 190 |
| fewshot_json | **100%** | 12.5% | 692 |
| cot_sc (3 samples, vote) | **100%** | 12.5% | 1,036 |
| decompose | **100%** | 0% | 480 |

All 4 strategies returned valid JSON on the first try for every question.

**What it means**
1. Given the right document, the 7B model reads plan numbers perfectly: every strategy scores 100% on facts.
   So for facts, the bottleneck is finding the right document (Week 2), not reading it.
2. Without a document, the model invents round numbers: 6 of the 27 wrong closed-book answers were ₹50,000 and
   another 6 were ₹5,000. This is why every answer must cite a retrieved document.
3. Multi-step money sums are where the model breaks: at best 1 of 8 bills right, whatever the prompt.
   Chain-of-thought costs 1.5x the tokens of few-shot and does not help. **This is the evidence for the
   cost-share calculator tool.**
4. **Pick fewshot_json** as the writer style: as accurate as cot_sc for 2/3 of the tokens.

### How our 8B model breaks (from `results/week1_failures.csv`)

| Failure pattern | Example question | Model value | Correct value | How many |
| --- | --- | --- | --- | --- |
| Invents a plausible number with no document | "What is the yearly dental limit on the Platinum plan in zone E?" (closed_book) | ₹50,000 | ₹18,000 | 27 of 30 facts wrong; 12 of them were ₹50,000 or ₹5,000 |
| Claims it applied the rules, but the total is made up | ₹90,000 bill, 3 days at ₹8,000 (M001, fewshot): "after meeting the deductible and applying the room rent cap" | ₹10,000 | ₹17,750 | 22 of 24 bill answers (3 strategies × 8) |
| Misreads year-to-date usage | ₹75,000 bill (M015, fewshot): "with the deductible and cost-sharing already met" | ₹28,000 | ₹21,675 (only ₹12,500 of a ₹20,000 deductible was met) | Seen in the replayed answers |
| Only one-step sums come out right | ₹45,000 bill, no room, zero deductible, 4% co-pay (M007) | ₹1,800 | ₹1,800 (correct) | The only bill fewshot and cot_sc got right |

What we expected but did **not** see: picking the distracting second number from the Conditions section. With
the right document, the model never did this in 90 fact answers.

## Week 2: retrieval (`results/retrieval_experiment.csv`)

45 questions (37 fact, 8 version). Recall without oracle filters:

| Chunking | Keyword @5 | Dense @5 | Hybrid @1 | Hybrid @5 | Hybrid + rerank @1 |
| --- | --- | --- | --- | --- | --- |
| whole | 0.956 | 1.0 | 0.644 | 1.0 | 0.600 |
| whole_noheader | 0.956 | 0.978 | 0.733 | 1.0 | 0.600 |
| words60 | 0.822 | 1.0 | 0.733 | 1.0 | 0.600 |
| **sections** | 0.844 | 1.0 | **0.911** | 1.0 | 0.600 |

With oracle filters (correct tier, zone, topic, year) every setup scores 1.0 at every k.

**The three documented decisions**

| Decision | Choice | Evidence | Reason |
| --- | --- | --- | --- |
| Chunking | sections | hybrid recall@1 0.911, the best of all 4 | One chunk per numbered section keeps each entitlement sentence on its own |
| Embedding model | bge-small-en-v1.5, local | dense recall@5 1.0 vs keyword 0.844 (sections) | The only option (no endpoint); next we would test a larger bge model |
| Reranker | **Keep it off for top-1**, rely on filters | recall@1 falls to 0.600 with MiniLM in every chunking | Documents differ only by tier, zone or year; a general web-search reranker cannot tell them apart |

**Why keyword search ranks first so badly (0.02–0.04 at k=1):** the 50 FAQs are short and repeat the benefit's
name, so BM25 puts an FAQ first for 33 of 37 fact questions. FAQs are tagged tier "ALL", so the planner's filters
remove them. **This is the evidence for a planner that extracts tier and zone.**

## Week 3: official evaluation (`results/agent_run*_summary.json`, `results/single_rag_run*_summary.json`)

98 items, eval-v1, 3 runs each, `USE_CACHE=0`, Qwen2.5-7B at temperature 0, local bge-small embeddings.
Run on 2026-10-30, 17:29 to 18:35. Every metric except latency had **zero spread** across the 3 runs: at
temperature 0 the endpoint answers the same way every time.

### Ablation: single prompt against the full agent

| Metric | Single RAG (1 search, 1 call) | Full agent | What to say |
| --- | --- | --- | --- |
| Accuracy overall | 65.3% | **99.0%** | +33.7 points |
| Accuracy on fact | 97.3% | 97.3% | Retrieval plus the writer is enough for plain facts |
| Accuracy on cost_calc / waiting_calc | 0% / 16.7% | **100% / 100%** | The single prompt cannot run the calculator or the waiting-period checker |
| Accuracy on member_fact | 16.7% | **100%** | It cannot look up the profile, so it guesses the member's plan |
| Accuracy on version | 0% | **100%** | It only searches current documents; the agent's planner passes year 2024 |
| Accuracy on appeal | 0% | **100%** | No tool to create a draft |
| Accuracy on refuse / injection / clarify | 100% | 100% | The code guard runs in both, so safety does not depend on the agent |
| Citation accuracy | 57.3% | 98.7% | |
| Injected ₹99,999 returned | 0 | 0 | |
| Average tokens per question | 962 | 3,163 | The agent costs 3.3x the tokens |
| Average / p95 latency | 2.7 s / 2.5 s | 10.6 s / 44.2 s | The price of the extra steps (see latency below) |

### Targets from the specification

| Target | Result | Met? |
| --- | --- | --- |
| Overall accuracy ≥ 75% | 99.0% | Yes |
| cost_calc + waiting_calc ≥ 85% | 100% | Yes |
| refuse + injection 100% | 100% | Yes |
| Injected value returned = 0 | 0 | Yes |
| Citation accuracy ≥ 85% | 98.7% | Yes |
| recall@5 ≥ 90% (best setup, no filters) | 100% | Yes |
| Task completion ≥ 90% | 98.0% | Yes |
| p95 latency ≤ 30 s | 44.2 s | **No** |

**Latency:** 17 of 98 agent questions took over 30 s, but they did the same work as the fast ones (2 steps,
about 3,600 tokens). The time is spent waiting on the shared endpoint (HTTP 429 back-off and queueing), not
on extra agent steps. Median latency is 4.5 s.

### Failure taxonomy (checked by hand)

Agent, every run: 1 wrong answer out of 98.

| Category | Count | Real example | What we did |
| --- | --- | --- | --- |
| control: critic_failed | 1 | fact-002 (Silver zone E deductible): the answer ₹15,000 was right, the code critic passed it, but the model support check rejected it 3 times ("without specifying the unit") and the agent escalated | Kept as a known limit (fixing it mid-evaluation would void the runs). Proposed fix: when the code critic passes, the model check may warn but not block |
| control: loop_detected (answer still correct) | 1 | version-006: a repeated search was refused, the run recovered | None needed; counts against task completion only |
| retrieval_miss, reasoning_error, tool_call_error, hallucination | 0 | | |

Single prompt, every run: 34 wrong. The rule labels said 26 retrieval_miss and 8 reasoning_error; reading all
34 rows, we corrected them to the real cause:

| Corrected category | Count | Real example |
| --- | --- | --- |
| No tool: needs the calculator, the checker or the appeal tool (12 cost, 5 waiting, 3 appeal) | 20 | cost_calc-001: "you will not have to pay anything" on a ₹75,000 bill (correct: ₹21,675) |
| Hallucination: guesses the member's plan with no profile lookup (member_fact) | 5 | member_fact-002: answers for Silver zone B; the member is on a different plan (₹6,500 given, ₹10,500 correct) |
| retrieval_miss: 2024 documents are never searched | 8 | version-002: gives the 2025 value (36 months) as the 2024 one (24 months) |
| reasoning_error | 1 | fact-014: the right document was retrieved but the model said no document matched |

## Before the official runs: the agent on the real model

A 20-question mixed check (`results/check_run1.jsonl`) scored 14/20. Every failure was fixed and re-checked:

| What went wrong with Qwen | Fix (all in code, not prompts alone) | Re-check |
| --- | --- | --- |
| HTTP 429 from the endpoint after ~4 s of retries | Exponential back-off up to ~1 minute, honours `Retry-After` | 0 errors on re-run |
| Starter bug: any error switched JSON mode off for good | Only a 400 Bad Request disables JSON mode | `test_rate_limit_retries_and_keeps_json_mode` |
| Endpoint outages labelled "invalid JSON" | Separate `endpoint_error` control label | `test_endpoint_failure_is_not_called_json_failure` |
| Valid JSON followed by extra text was rejected | Keep the first complete JSON object | `test_extract_json_ignores_trailing_text...` |
| Planner asked for the profile again and again | Prompt states the known tier/zone and calls already made; first repeat is refused with a note | `test_single_repeat_is_refused_not_fatal` |
| Planner added an unasked 2024 year filter | Code drops it and logs `year_filter_removed` | `test_unasked_2024_filter_is_removed`; 6/6 poisoned-doc items right |
| Writer refused coverage and 2024 questions as "medical" or "superseded" | Clearer writer rules | 8/8 version items right (`results/check_versions_run1.jsonl`) |

Cost on the real model: about 2,300 tokens and 7 s per question; p95 latency 18 s.

## Compensating for a small model (the graded section): evidence so far

| What we did | Why the 7B model needed it | Evidence |
| --- | --- | --- |
| Calculator tool does every sum | 0–12.5% bill accuracy without it (Week 1); 0% single prompt vs 100% agent (Week 3) | `week1_strategies.csv`; agent and single_rag summaries |
| Retrieval with citations | 10% fact accuracy closed-book, 100% with the right document | Week 1 table |
| Planner extracts tier and zone as filters | Unfiltered keyword top-1 is an FAQ 33 of 37 times | Week 2 diagnosis |
| Code guard for refusals and injection | 100% of refuse, injection_user and clarify items, with no model call | `check_run1` summary |
| Sanitiser plus number-checking critic | Injected ₹99,999 never returned (0 of 6) | `check_versions_run1` summary |
| Code fixes for planner slips (repeats, unasked 2024 filter) | Seen in real traces | Week 3 preview table |

## Viva practice: questions you can now answer with numbers

- *Why a calculator tool?* Week 1: even the best prompt got 1 of 8 bills right; Week 3: single prompt 0%, agent 100%.
- *Is the agent worth 3.3x the tokens?* 65.3% → 99.0% accuracy; on member, cost, version and appeal questions it is the only thing that works.
- *Why is p95 latency 44 s?* Endpoint waiting (429 back-off), not extra steps: slow questions used the same 2 steps.
- *Why hybrid search?* Sections: keyword recall@1 0.022, dense 0.822, hybrid 0.911.
- *Did the reranker help?* No: recall@1 dropped to 0.600. Explain the near-twin documents.
- *Why does the planner extract filters?* With oracle filters recall is 1.0; without, an FAQ wins.
- *Which prompt style and why?* fewshot_json: same 100% fact accuracy as cot_sc at 2/3 of the tokens.
