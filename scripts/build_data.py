"""Step: build the corpus, members, claims and the evaluation set (no model needed)."""
from collections import Counter

import _boot  # noqa
from claimcheck import corpus_gen, evalset

metas, facts = corpus_gen.build_all()
corpus_gen.write_members()
items = evalset.build_eval()
print(f"documents: {len(metas)}  benefit facts: {len(facts)}  members: {len(corpus_gen.MEMBERS)}  eval items: {len(items)}")
print(dict(Counter(m["kind"] for m in metas)))
print(dict(Counter(i["type"] for i in items)))
