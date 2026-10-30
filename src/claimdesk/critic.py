"""Code checks on what the agents wrote. Deterministic: these are the checks that block."""
import json
import re

from claimcheck.agent import PROMISE, numbers_in

from . import data
from .agents import letter_facts

RUPEES = re.compile(r"₹\s?\d[\d,]*")


def check_letter(result, letter):
    """Problems with a letter, or [] when it is fine."""
    problems = []
    allowed = set()
    for v in letter_facts(result).values():
        if isinstance(v, str) and v.startswith("₹"):
            allowed |= numbers_in(v)
    stated = set().union(*(numbers_in(x) for x in RUPEES.findall(letter.body))) if RUPEES.search(letter.body) else set()
    extra = sorted(stated - allowed)
    if extra:
        problems.append(f"amounts not in the decision: {', '.join(str(int(x)) for x in extra)}")
    if result["decision"] in ("approve", "part_pay") and result["payable"] is not None \
            and float(result["payable"]) not in stated:
        problems.append("the payable amount is missing")
    said_paid = re.search(r"(?<!not )\b(approved|will be paid)\b", letter.body, re.I)
    if result["decision"] in ("reject", "pend") and said_paid:
        problems.append(f"the letter says approved or paid but the decision is {result['decision']}")
    if not re.search(r"\b(draft|review)", letter.body, re.I):
        problems.append("say that a claims reviewer will confirm this draft decision")
    if PROMISE.search(letter.body):
        problems.append("remove the promise")
    return problems


def bad_citations(run):
    """Citations in a finding that are neither a rule id nor an id seen in that agent's tool results."""
    if run.finding is None:
        return []
    seen = json.dumps([c["result"] for c in run.calls], ensure_ascii=False)
    found = []
    for key in ("reasons", "signals", "issues"):
        for item in getattr(run.finding, key, []) or []:
            cites = getattr(item, "cite", None) if hasattr(item, "cite") else getattr(item, "rule", None)
            for c in cites if isinstance(cites, list) else [cites]:
                for one in re.split(r"[,\s]+", str(c or "")):          # "BEN-A, BEN-B" counts as two
                    if one and one not in data.RULES and one not in seen:
                        found.append(one)
    return found
