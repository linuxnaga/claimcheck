# 12-page report outline (page budget in brackets)
Every table must come from a saved results file; put the file name in the caption.

1. Summary and specification (1.5): what ClaimCheck does, `docs/SPEC.md`, the eval-set table.
2. Prompting strategies and how the model breaks (1.5): `results/week1_strategies.csv` (fact and cost rows);
   three real failure patterns from `results/week1_failures.csv`.
3. Retrieval (2): `results/retrieval_experiment.csv`; the three documented decisions (chunking, embedding
   model, reranker); recall@5 next to end-to-end fact accuracy.
4. Agent design (2): the loop diagram, the five tools, the cost-share rule, the error-handling table.
5. Evaluation and ablation (2): `agent_run*_summary.json` mean ± spread, single_rag vs agent table,
   failure taxonomy checked by hand.
6. Observability, responsible AI, deployment (1.5): traces, `results/security_report.md`, red-team log,
   green CI run, rate limit and cache screenshots.
7. What we did to compensate for a small model (1, GRADED): code guard, tier/zone filters and normalisers,
   calculator and waiting-period tools, JSON schema retries, deterministic critic, sanitiser, caps and cache.
8. Limits, future work, team (0.5).
