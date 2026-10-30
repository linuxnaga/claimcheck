"""Ask one question from the command line. Example: python scripts/ask.py "What will I pay on a ₹80,000 bill?" --user M003"""
import argparse, json
import _boot  # noqa
from claimcheck.agent import run_agent

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--user", default=None)
a = ap.parse_args()
r = run_agent(a.question, user_id=a.user)
print(r["answer"]["answer_text"])
print({k: r[k] for k in ("status", "control", "steps", "tool_calls", "tokens", "latency_s")},
      "cited:", r["answer"]["citations"])
print(json.dumps(r["trace"], indent=1, ensure_ascii=False))
from claimcheck.tracing import flush; flush()
