"""Decide one test claim with ClaimDesk and print what each agent did.
  python scripts/decide_claim.py CL1001 [--system full|no_adjudicator] [--no-letter]"""
import _boot  # noqa: F401
import argparse
import json

from claimcheck.corpus_gen import fmt_inr
from claimdesk import data
from claimdesk.orchestrator import decide_claim

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("claim_id")
    ap.add_argument("--system", default="full")
    ap.add_argument("--no-letter", action="store_true")
    a = ap.parse_args()
    split = "dev" if a.claim_id.upper().startswith("CL2") else "test"
    claim = next(c for c in data.test_claims(split) if c["claim_id"] == a.claim_id.upper())
    gold = data.test_gold(split)[claim["claim_id"]]
    res = decide_claim(claim, system=a.system, write_letter=not a.no_letter)
    for ag in res["agents"]:
        print(f"\n[{ag['role']}] {ag['steps']} steps, {ag['tokens']} tokens, {ag['seconds']} s, error={ag['error']}")
        for c in ag["calls"]:
            print(f"   {c['tool']}({json.dumps(c['args'])}) -> {json.dumps(c['result'])[:160]}")
        print("   finding:", json.dumps(ag["finding"], ensure_ascii=False)[:400])
    pay = lambda x: "-" if x is None else fmt_inr(x)
    print(f"\nDecision {res['decision']} {pay(res['payable'])} needs_human={res['needs_human']} control={res['control']}")
    print(f"Gold     {gold['decision']} {pay(gold['payable'])} needs_human={gold['needs_human']} ({gold['case_type']})")
    print(f"Tokens {res['tokens']}, model calls {res['llm_calls']}, {res['latency_s']} s")
    if res["letter"]:
        print("\nLetter:", res["letter"]["subject"], "\n" + res["letter"]["body"])
        print("Letter problems:", res["letter_problems"], "| model warning:", res["letter_warning"])
