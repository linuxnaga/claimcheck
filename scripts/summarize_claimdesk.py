"""Print the summary and agent scorecard of saved ClaimDesk run files.
  python scripts/summarize_claimdesk.py data/results/check_dev_full_run1.jsonl [--split dev]"""
import _boot  # noqa: F401
import argparse
import json

from claimdesk.evaluate import resummarize

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    for p in a.paths:
        s = resummarize(p, a.split)
        print(p, f"accuracy {s['accuracy']}, tokens {s['avg_tokens']}, p95 {s['p95_latency_s']} s")
        print(json.dumps(s["agent_scorecard"], indent=1))
