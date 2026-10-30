from collections import Counter
from claimcheck import corpus_gen, evalset
from claimcheck.corpus_gen import TIERS, TOPICS, ZONES, doc_id_for


def test_corpus_size_and_unique_ids():
    metas = corpus_gen.load_meta()
    assert len(metas) == 600
    assert len({m["doc_id"] for m in metas}) == len(metas)
    assert Counter(m["kind"] for m in metas) == {"benefit": 520, "faq": 50, "procedure": 24, "poisoned": 6}


def test_at_least_9_topics_change_between_years():
    facts = {f["doc_id"]: f["value"] for f in corpus_gen.load_facts()}
    for tier in TIERS:
        for zone in ZONES:
            changed = sum(facts[doc_id_for(t, tier, zone, "2025")] != facts[doc_id_for(t, tier, zone, "2024")]
                          for t in TOPICS)
            assert changed >= 9, (tier, zone)


def test_zone_e_copay_is_5_points_below_zone_a():
    for tier in TIERS:
        assert corpus_gen.value_2025("copay_pct", tier, "A") - corpus_gen.value_2025("copay_pct", tier, "E") == 5


def test_indian_rupee_format():
    assert corpus_gen.fmt_inr(120000) == "₹1,20,000"
    assert corpus_gen.fmt_inr(32400) == "₹32,400"
    assert corpus_gen.fmt_inr(500) == "₹500"


def test_members_cover_every_tier_and_zone():
    ms = corpus_gen.load_members()
    assert len(ms) == 15
    assert {m["tier"] for m in ms} == set(TIERS) and {m["zone"] for m in ms} == set(ZONES)


def test_eval_set_size_and_types():
    items = evalset.load_eval()
    assert len(items) == 98
    assert Counter(i["type"] for i in items) == {
        "fact": 37, "cost_calc": 12, "waiting_calc": 6, "member_fact": 6, "version": 8, "injection_doc": 6,
        "injection_user": 5, "refuse": 10, "clarify": 5, "appeal": 3}


def test_fact_questions_cover_all_topics_tiers_zones():
    fs = evalset.load_eval(types=["fact"])
    assert {i["gold"]["topic"] for i in fs} == set(TOPICS)
    assert {i["gold"]["tier"] for i in fs} == set(TIERS)
    assert {i["gold"]["zone"] for i in fs} == set(ZONES)


def test_waiting_items_have_both_answers():
    assert sorted(i["gold"]["value"] for i in evalset.load_eval(types=["waiting_calc"])) == [0, 0, 0, 1, 1, 1]


def test_gold_docs_exist():
    ids = {m["doc_id"] for m in corpus_gen.load_meta()}
    for i in evalset.load_eval():
        for d in i["gold"].get("doc_ids", []):
            assert d in ids


def test_build_is_deterministic():
    a = [i["question"] for i in evalset.load_eval()]
    evalset.build_eval()
    assert a == [i["question"] for i in evalset.load_eval()]
