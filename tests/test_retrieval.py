import hashlib, re
import numpy as np
from claimcheck import corpus_gen, evalset, retrieval


def fake_embed(texts):
    out = np.zeros((len(texts), 256), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def _retriever(chunking):
    docs = corpus_gen.load_docs()
    chunks = retrieval.make_chunks(docs, chunking)
    return retrieval.Retriever(docs, chunks, fake_embed([c["text"] for c in chunks]), fake_embed)


def test_chunkers_produce_chunks():
    docs = corpus_gen.load_docs()
    for s in retrieval.CHUNKINGS:
        assert len(retrieval.make_chunks(docs, s)) >= len(docs)


def test_header_names_tier_zone_year():
    docs = {d["doc_id"]: d for d in corpus_gen.load_docs()}
    h = retrieval.header(docs["BEN-SILVER-B-2025-deductible"])
    assert "Silver" in h and "Zone B" in h and "2025" in h


def test_filters_work():
    r = _retriever("sections")
    ids = r.search_docs("deductible", mode="bm25", k=5, filters={"tier": "gold", "zone": "C", "status": "CURRENT"})
    assert ids and all(r.docs[i]["tier"] == "gold" and r.docs[i]["zone"] == "C" for i in ids)


def test_hybrid_runs():
    r = _retriever("sections")
    assert r.search_docs("ICU cap gold zone C", mode="hybrid", k=3)


def test_bm25_recall_reasonable():
    r = _retriever("sections")
    items = [{"question": i["question"], "gold_doc_ids": i["gold"]["doc_ids"]} for i in evalset.load_eval(types=["fact"])]
    rec = retrieval.recall_at_k(r, items, "bm25")
    print("bm25 recall", rec)
    assert rec[10] > 0.3


def test_index_is_tied_to_its_embedding_model(monkeypatch):
    from claimcheck import config
    monkeypatch.setattr(config, "EMBED_PROVIDER", "local")
    monkeypatch.setattr(config, "EMBED_MODEL", "model-a")
    retrieval.build_index("whole", fake_embed)
    assert retrieval.index_ready("whole")
    monkeypatch.setattr(config, "EMBED_MODEL", "model-b")       # switching models makes the index unusable
    assert not retrieval.index_ready("whole")
    assert retrieval.load_retriever("whole", use_rerank=False).emb is None
