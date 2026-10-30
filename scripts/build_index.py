"""Step: embed all chunks with the embeddings endpoint and save them (needs the endpoint)."""
import argparse
import _boot  # noqa
from claimcheck import retrieval
from claimcheck.llm import llm

ap = argparse.ArgumentParser()
ap.add_argument("--chunking", default="sections", choices=retrieval.CHUNKINGS)
ap.add_argument("--force", action="store_true", help="embed again even if the index exists")
a = ap.parse_args()
if retrieval.index_ready(a.chunking) and not a.force:
    raise SystemExit(f"index '{a.chunking}' already exists (use --force to rebuild). Nothing to do.")
n = retrieval.build_index(a.chunking, llm.embed)
print(f"built index '{a.chunking}' with {n} chunks")
