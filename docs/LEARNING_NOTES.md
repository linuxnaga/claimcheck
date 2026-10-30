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

## Week 3 preview: the agent on the real model

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
| Calculator tool does every sum | 0–12.5% bill accuracy without it, 100% (2/2) with it | Week 1 table; `check_run1` cost_calc |
| Retrieval with citations | 10% fact accuracy closed-book, 100% with the right document | Week 1 table |
| Planner extracts tier and zone as filters | Unfiltered keyword top-1 is an FAQ 33 of 37 times | Week 2 diagnosis |
| Code guard for refusals and injection | 100% of refuse, injection_user and clarify items, with no model call | `check_run1` summary |
| Sanitiser plus number-checking critic | Injected ₹99,999 never returned (0 of 6) | `check_versions_run1` summary |
| Code fixes for planner slips (repeats, unasked 2024 filter) | Seen in real traces | Week 3 preview table |

## Viva practice: questions you can now answer with numbers

- *Why a calculator tool?* Week 1: even the best prompt got 1 of 8 bills right.
- *Why hybrid search?* Sections: keyword recall@1 0.022, dense 0.822, hybrid 0.911.
- *Did the reranker help?* No: recall@1 dropped to 0.600. Explain the near-twin documents.
- *Why does the planner extract filters?* With oracle filters recall is 1.0; without, an FAQ wins.
- *Which prompt style and why?* fewshot_json: same 100% fact accuracy as cot_sc at 2/3 of the tokens.
