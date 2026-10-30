"""Runs the evaluation set and summarises it (accuracy, citations, cost, latency, failures)."""
import json
import statistics
from collections import Counter, defaultdict

from . import config
from .agent import run_agent, run_single_rag
from .evalset import load_eval, score_item


def _crash(e):
    return {"answer": {"status": "escalated", "answer_text": f"CRASH {e}", "citations": [], "value": None},
            "status": "escalated", "control": "crash", "retrieved": [], "tool_errors": 1, "tool_calls": 0,
            "steps": 0, "tokens": 0, "latency_s": 0, "cost_usd": 0, "json_retries": 0, "case_id": None,
            "injection_flags": 0, "trace": []}


def run_eval(name, mode="agent", types=None, limit=None, run_no=1, progress=True):
    config.ensure_dirs()
    items = load_eval(types, limit)
    out_path = config.RESULTS_DIR / f"{name}_run{run_no}.jsonl"
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for n, it in enumerate(items, 1):
            try:
                res = run_agent(it["question"], user_id=it["user_id"]) if mode == "agent" \
                    else run_single_rag(it["question"], user_id=it["user_id"])
            except Exception as e:   # never let one bad item stop the whole run
                res = _crash(e)
            sc = score_item(it, res)
            row = {"id": it["id"], "type": it["type"], "question": it["question"], "correct": sc["correct"],
                   "citation_ok": sc["citation_ok"], "failure": sc["failure"], "status": res["status"],
                   "control": res["control"], "steps": res["steps"], "tool_calls": res["tool_calls"],
                   "tool_errors": res["tool_errors"], "json_retries": res["json_retries"], "tokens": res["tokens"],
                   "latency_s": res["latency_s"], "cost_usd": res["cost_usd"], "answer": res["answer"],
                   "retrieved": res["retrieved"], "injection_flags": res["injection_flags"], "gold": it["gold"],
                   "trace": res["trace"]}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if progress and n % 10 == 0:
                print(f"  {n}/{len(items)} done, running accuracy {sum(r['correct'] for r in rows) / n:.2f}")
    summary = summarize(rows)
    (config.RESULTS_DIR / f"{name}_run{run_no}_summary.json").write_text(json.dumps(summary, indent=1))
    return rows, summary


def summarize(rows):
    n = len(rows)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["correct"])
    cit = [r["citation_ok"] for r in rows if r["citation_ok"] is not None]
    lat = sorted(r["latency_s"] for r in rows)
    injected = sum(1 for r in rows if r["type"] == "injection_doc" and r["answer"].get("value") is not None
                   and float(r["answer"]["value"]) == r["gold"]["bad_value"])
    return {
        "n": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "accuracy_by_type": {k: round(sum(v) / len(v), 3) for k, v in by_type.items()},
        "citation_accuracy": round(sum(cit) / len(cit), 3) if cit else None,
        "task_completion_rate": round(sum(r["control"] is None for r in rows) / n, 3),
        "injected_value_returned": injected,
        "avg_steps": round(statistics.mean(r["steps"] for r in rows), 2),
        "avg_tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 2),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
        "avg_latency_s": round(statistics.mean(lat), 2),
        "p95_latency_s": lat[int(0.95 * (n - 1))],
        "avg_cost_usd": round(statistics.mean(r["cost_usd"] for r in rows), 6),
        "total_tokens": sum(r["tokens"] for r in rows),
        "json_retries": sum(r["json_retries"] for r in rows),
        "failure_taxonomy": dict(Counter(r["failure"] for r in rows if r["failure"])),
    }
