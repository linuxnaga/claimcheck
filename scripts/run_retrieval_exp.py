"""Week 2: chunking x retrieval mode experiment. Reports recall@k separately from answer quality.

Scored on the 37 fact and 8 version questions. Writes results/retrieval_experiment.csv.
"""
import argparse, csv
import _boot  # noqa
import numpy as np
from claimcheck import config, evalset, retrieval
from claimcheck.corpus_gen import load_docs
from claimcheck.llm import llm

ap = argparse.ArgumentParser()
ap.add_argument("--no-dense", action="store_true", help="BM25 only (no embedding calls)")
ap.add_argument("--no-rerank", action="store_true")
a = ap.parse_args()
items = [{"question": i["question"], "gold_doc_ids": i["gold"]["doc_ids"], "gold": i["gold"], "type": i["type"]}
         for i in evalset.load_eval(types=["fact", "version"])]
reranker = None
if not a.no_rerank:
    try:
        reranker = retrieval.Reranker()
    except Exception as e:
        print("reranker unavailable:", e)


def oracle_filters(it):
    """The correct tier, zone, topic and year, as a perfect planner would extract them."""
    g = it["gold"]
    f = {"tier": g["tier"], "zone": g["zone"], "topic": g["topic"]}
    if it["type"] == "version":
        f["year"] = "2024"
    else:
        f["status"] = "CURRENT"
    return f


docs = load_docs()
rows = []
for chunking in retrieval.CHUNKINGS:
    chunks = retrieval.make_chunks(docs, chunking)
    emb = None
    if not a.no_dense:
        p = retrieval.index_path(chunking)
        if not retrieval.index_ready(chunking):
            print(f"embedding {len(chunks)} chunks for '{chunking}' ...")
            retrieval.build_index(chunking, llm.embed)
        emb = np.load(p / "emb.npy")
    r = retrieval.Retriever(docs, chunks, emb, llm.embed if emb is not None else None, reranker)
    modes = [("bm25", False)] + ([("dense", False), ("hybrid", False)] if emb is not None else []) + \
            ([("hybrid", True)] if (reranker and emb is not None) else ([("bm25", True)] if reranker else []))
    for mode, rr in modes:
        for use_f in (False, True):
            rec = retrieval.recall_at_k(r, items, mode, filters_fn=oracle_filters if use_f else None, rerank=rr)
            row = {"chunking": chunking, "mode": mode + ("+rerank" if rr else ""),
                   "oracle_filters": use_f, **{f"recall@{k}": v for k, v in rec.items()}}
            rows.append(row)
            print(row)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "retrieval_experiment.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("saved results/retrieval_experiment.csv")
