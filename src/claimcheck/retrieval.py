"""Hybrid retrieval: chunking, BM25, dense vectors, rank fusion, optional reranker."""
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .corpus_gen import TIERS, ZONES, load_docs

CHUNKINGS = ["whole", "whole_noheader", "words60", "sections"]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def header(d):
    tier = TIERS.get(d["tier"], d["tier"])
    zone = f"Zone {d['zone']} ({ZONES[d['zone']]})" if d["zone"] in ZONES else "All zones"
    return f"[{d['doc_id']}] {d['title']} | {tier} | {zone} | {d['year']} {d['status']}"


def make_chunks(docs, strategy):
    chunks = []
    for d in docs:
        text, h = d["text"], header(d)
        if strategy == "whole":
            parts = [h + "\n" + text]
        elif strategy == "whole_noheader":
            parts = [text]
        elif strategy == "words60":
            words = text.split()
            parts = [h + "\n" + " ".join(words[i:i + 60]) for i in range(0, max(1, len(words) - 20), 40)]
        elif strategy == "sections":
            parts = [h + "\n" + s.strip() for s in re.split(r"\n(?=\d\. )", text) if s.strip()]
        else:
            raise ValueError(f"unknown chunking: {strategy}")
        for i, p in enumerate(parts):
            chunks.append({"chunk_id": f"{d['doc_id']}#{i}", "doc_id": d["doc_id"], "text": p})
    return chunks


class Reranker:
    """Cross-encoder reranker (runs on CPU or the Colab GPU)."""

    def __init__(self, model_name=None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name or config.RERANK_MODEL)

    def score(self, query, texts):
        return list(self.model.predict([(query, t) for t in texts]))


class Retriever:
    def __init__(self, docs, chunks, embeddings=None, embed_fn=None, reranker=None):
        self.docs = {d["doc_id"]: d for d in docs}
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.emb = embeddings
        self.embed_fn = embed_fn
        self.reranker = reranker

    def _mask(self, filters):
        ok = np.ones(len(self.chunks), dtype=bool)
        if not filters:
            return ok
        for i, c in enumerate(self.chunks):
            d = self.docs[c["doc_id"]]
            for field, wanted in filters.items():
                if wanted is None:
                    continue
                if field == "status" and d["status"] != wanted:
                    ok[i] = False
                elif field != "status" and d.get(field) != wanted:
                    ok[i] = False
        return ok

    @staticmethod
    def _rank(scores, mask, n):
        s = np.where(mask, scores, -np.inf)
        idx = np.argsort(-s)[:n]
        return [int(i) for i in idx if np.isfinite(s[i])]

    def search_chunks(self, query, mode="hybrid", n=30, filters=None):
        mask = self._mask(filters)
        rankings = []
        if mode in ("bm25", "hybrid"):
            rankings.append(self._rank(np.array(self.bm25.get_scores(tokenize(query))), mask, n))
        if mode in ("dense", "hybrid") and self.emb is not None and self.embed_fn is not None:
            qv = self.embed_fn([query])[0]
            rankings.append(self._rank(self.emb @ qv, mask, n))
        if not rankings:
            raise ValueError("no usable retrieval mode (dense needs embeddings)")
        if len(rankings) == 1:
            return rankings[0]
        fused = {}
        for r in rankings:  # reciprocal rank fusion
            for rank, idx in enumerate(r):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)
        return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])][:n]

    def search_docs(self, query, mode="hybrid", k=5, filters=None, rerank=False, pool=20):
        """Return a ranked list of doc_ids (best first)."""
        order = []
        for idx in self.search_chunks(query, mode, n=60, filters=filters):
            did = self.chunks[idx]["doc_id"]
            if did not in order:
                order.append(did)
        if rerank and self.reranker is not None and order:
            cand = order[:pool]
            scores = self.reranker.score(query, [self.docs[d]["text"][:700] for d in cand])
            cand = [d for d, _ in sorted(zip(cand, scores), key=lambda x: -x[1])]
            order = cand + order[pool:]
        return order[:k]


# ---- building / loading the index -------------------------------------------
def index_path(chunking):
    return config.INDEX_DIR / chunking


def embed_signature():
    """Which embedding model made an index. Vectors from different models must never be mixed."""
    return {"provider": config.EMBED_PROVIDER, "model": config.EMBED_MODEL}


def index_ready(chunking):
    """True when a saved index exists AND was built with the current embedding provider and model."""
    p = index_path(chunking)
    if not (p / "emb.npy").exists() or not (p / "index_meta.json").exists():
        return False
    meta = json.loads((p / "index_meta.json").read_text())
    return {k: meta.get(k) for k in ("provider", "model")} == embed_signature()


def build_index(chunking, embed_fn):
    docs = load_docs()
    chunks = make_chunks(docs, chunking)
    p = index_path(chunking)
    p.mkdir(parents=True, exist_ok=True)
    emb = embed_fn([c["text"] for c in chunks])
    np.save(p / "emb.npy", emb)
    (p / "chunk_ids.json").write_text(json.dumps([c["chunk_id"] for c in chunks]))
    (p / "index_meta.json").write_text(json.dumps({**embed_signature(), "dims": int(emb.shape[1]),
                                                   "chunks": len(chunks)}, indent=1))
    return len(chunks)


def load_retriever(chunking=None, embed_fn=None, use_rerank=True):
    chunking = chunking or config.CHUNKING
    docs = load_docs()
    chunks = make_chunks(docs, chunking)
    emb, p = None, index_path(chunking)
    if index_ready(chunking):
        emb = np.load(p / "emb.npy")
        if len(emb) != len(chunks):
            emb = None  # index is stale: rebuild it
    elif (p / "emb.npy").exists():
        print(f"[warn] index '{chunking}' was built with a different embedding model than "
              f"{embed_signature()}; using BM25 only. Rebuild: python scripts/build_index.py --chunking {chunking} --force")
    if emb is not None and embed_fn is None:
        from .llm import llm
        embed_fn = llm.embed
    rer = None
    if use_rerank:
        try:
            rer = Reranker()
        except Exception as e:  # sentence-transformers missing or no internet
            print(f"[warn] reranker disabled: {e}")
    return Retriever(docs, chunks, emb, embed_fn, rer)


def recall_at_k(retriever, items, mode, ks=(1, 3, 5, 10), filters_fn=None, rerank=False):
    """items: list of dict(question, gold_doc_ids). Returns {k: recall}."""
    hits = {k: 0 for k in ks}
    for it in items:
        f = filters_fn(it) if filters_fn else None
        ranked = retriever.search_docs(it["question"], mode=mode, k=max(ks), filters=f, rerank=rerank)
        for k in ks:
            if any(g in ranked[:k] for g in it["gold_doc_ids"]):
                hits[k] += 1
    return {k: round(hits[k] / max(1, len(items)), 3) for k in ks}
