"""Run the evaluation set. Examples:
   python scripts/run_eval.py --name agent --limit 10        (small test)
   python scripts/run_eval.py --name agent --run 1           (full 98 items)
   python scripts/run_eval.py --name single_rag --mode single (ablation)
"""
import argparse, json
import _boot  # noqa
from claimcheck.harness import run_eval

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="agent")
ap.add_argument("--mode", default="agent", choices=["agent", "single"])
ap.add_argument("--run", type=int, default=1)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--types", nargs="*", default=None)
a = ap.parse_args()
_, s = run_eval(a.name, a.mode, a.types, a.limit, a.run)
print(json.dumps(s, indent=1))
from claimcheck.tracing import flush; flush()
