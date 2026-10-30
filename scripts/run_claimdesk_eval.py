"""Run ClaimDesk systems on the frozen test claims.

  python scripts/run_claimdesk_eval.py --system full --tag check            # tune on the 33 dev claims
  python scripts/run_claimdesk_eval.py --system full --split test --runs 3 --no-cache   # official runs only
Systems: single_prompt (A), single_agent (B), no_adjudicator (C), full (D), full_no_coding, full_no_fraud."""
import _boot  # noqa: F401
import argparse
import json

from claimcheck import tracing
from claimcheck.llm import llm
from claimdesk.evaluate import SYSTEMS, run_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(SYSTEMS), required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--first-run", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--types", nargs="*", help="case types to include, e.g. clean disagreement")
    ap.add_argument("--ids", nargs="*", help="claim ids to include")
    ap.add_argument("--split", choices=["dev", "test"], default="dev",
                    help="dev (33 claims) for tuning; test (150, frozen) only for official runs")
    ap.add_argument("--tag", default="claimdesk", help="file name prefix; use 'check' for informal runs")
    ap.add_argument("--no-cache", action="store_true", help="call the model every time (official runs)")
    a = ap.parse_args()
    if a.no_cache:
        llm.use_cache = False
    for run_no in range(a.first_run, a.first_run + a.runs):
        _, summary = run_eval(a.system, run_no, a.limit, a.types, a.ids, tag=f"{a.tag}_{a.split}", split=a.split)
        print(json.dumps(summary, indent=1))
    tracing.flush()
