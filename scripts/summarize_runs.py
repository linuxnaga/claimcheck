"""After 3 runs of the same config, print mean and spread. Example: python scripts/summarize_runs.py agent"""
import glob, json, statistics, sys
import _boot  # noqa
from claimcheck import config

name = sys.argv[1] if len(sys.argv) > 1 else "agent"
files = sorted(glob.glob(str(config.RESULTS_DIR / f"{name}_run*_summary.json")))
runs = [json.load(open(f)) for f in files]
print(f"{len(runs)} runs found for '{name}'")
for key in ("accuracy", "citation_accuracy", "task_completion_rate", "avg_steps", "avg_tool_calls",
            "avg_tokens", "avg_latency_s", "p95_latency_s", "avg_cost_usd", "injected_value_returned"):
    vals = [r[key] for r in runs if r.get(key) is not None]
    if vals:
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0
        print(f"{key:24s} mean {statistics.mean(vals):.4f}   spread (std) {sd:.4f}")
for t in sorted({t for r in runs for t in r["accuracy_by_type"]}):
    vals = [r["accuracy_by_type"][t] for r in runs if t in r["accuracy_by_type"]]
    print(f"  accuracy[{t}] mean {statistics.mean(vals):.3f}")
print("failure taxonomy (last run):", runs[-1]["failure_taxonomy"] if runs else {})
