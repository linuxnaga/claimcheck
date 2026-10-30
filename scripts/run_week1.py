"""Week 1: compare prompting strategies on 30 fact questions and 8 cost questions.

The strategies get the CORRECT documents as context, so the table measures how well the model reads.
Cost questions get the four plan documents and the member's usage, but NO calculator: this shows how
often the model gets multi-step sums wrong.
"""
import argparse, csv
import _boot  # noqa
from claimcheck import config, evalset
from claimcheck.corpus_gen import load_members
from claimcheck.llm import Usage
from claimcheck.strategies import STRATEGIES, answer_with, doc_block
from claimcheck.tools import essential_text

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30, help="number of fact questions")
ap.add_argument("--n-cost", type=int, default=8, help="number of cost questions")
a = ap.parse_args()
members = {m["member_id"]: m for m in load_members()}
items = evalset.load_eval(types=["fact"], limit=a.n) + evalset.load_eval(types=["cost_calc"], limit=a.n_cost)


def context(it):
    blocks = [doc_block(d, essential_text((config.POLICY_DIR / f"{d}.txt").read_text(encoding="utf-8")))
              for d in it["gold"]["doc_ids"]]
    if it["user_id"]:
        m = members[it["user_id"]]
        blocks.append(f"Member usage this year: deductible already met ₹{m['deductible_met_ytd']}, "
                      f"cost-sharing already paid ₹{m['oop_used_ytd']}.")
    return "\n".join(blocks)


rows, fails = [], []
for s in STRATEGIES:
    for kind in ("fact", "cost_calc"):
        part = [i for i in items if i["type"] == kind]
        if not part:
            continue
        ok = valid = tokens = 0
        for it in part:
            u = Usage()
            ans = answer_with(s, it["question"], context(it), u)
            tokens += u.total
            if ans is None:
                fails.append((s, kind, it["question"], it["gold"]["value"], "INVALID JSON"))
                continue
            valid += 1
            good = evalset.value_ok(ans.value, it["gold"]) or (it["gold"]["value"] == 0 and ans.value is None)
            ok += bool(good)
            if not good:
                fails.append((s, kind, it["question"], it["gold"]["value"], ans.value))
        rows.append({"strategy": s, "question_type": kind, "n": len(part), "accuracy": round(ok / len(part), 3),
                     "valid_json_first_try": round(valid / len(part), 3), "avg_tokens": round(tokens / len(part))})
        print(rows[-1])
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "week1_strategies.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
with open(config.RESULTS_DIR / "week1_failures.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["strategy", "question_type", "question", "gold", "model_value"]); w.writerows(fails)
print("saved results/week1_strategies.csv and week1_failures.csv  -> write how the model breaks in your report")
