"""Hard decision rules (specification section 8). Plain code: the model cannot override them.

`apply_rules` turns the facts the specialists found into a decision. `gold_facts` finds the same facts
straight from the tools, which is how the gold labels are made.
"""
from . import data
from .tools import WaitArgs, compute_payable, package_limit, similar_claims, stay_days, waiting_period_checker

DECISIONS = ("approve", "part_pay", "reject", "pend")


def empty_facts():
    return {"complete": True, "missing_documents": [], "excluded_rule": None, "wait_rule": None,
            "covered_lines": None, "invalid_lines": [], "long_stay": False, "duplicate_of": None,
            "near_duplicate_of": None, "near_same_procedure": None, "inflated": False, "watchlist": False}


def apply_rules(claim, facts):
    """Return the decision dict. `open_issue` is set when the Adjudicator has a judgement to make."""
    reasons, calc, open_issue = [], None, None

    def out(decision, payable, rule_texts):
        human = []
        if decision in ("reject", "pend"):
            human.append(decision)
        if payable and payable > data.HUMAN_REVIEW_ABOVE:
            human.append("R-HIGH-VALUE")
        for flag, rule in (("watchlist", "R-WATCH"), ("inflated", "R-TARIFF"), ("long_stay", "R-LOS")):
            if facts.get(flag):
                human.append(rule)
        if facts.get("near_duplicate_of"):
            human.append("R-NEAR-DUP")
        return {"decision": decision, "payable": payable, "needs_human": bool(human), "human_reasons": human,
                "reasons": rule_texts, "calc": calc, "open_issue": open_issue}

    if not facts["complete"]:
        missing = ", ".join(facts["missing_documents"]) or "a required document"
        return out("pend", None, [{"rule": "R-DOCS", "text": f"Waiting for: {missing}.", "cite": ["R-DOCS"]}])
    for key in ("excluded_rule", "wait_rule"):
        if facts[key]:
            return out("reject", 0, [{"rule": facts[key], "text": data.RULES[facts[key]], "cite": [facts[key]]}])
    if facts["duplicate_of"]:
        return out("reject", 0, [{"rule": "R-DUP", "text": f"Repeats claim {facts['duplicate_of']}.",
                                  "cite": ["R-DUP", facts["duplicate_of"]]}])

    covered = facts["covered_lines"]
    if covered is None:
        covered = [l["no"] for l in claim["lines"] if l["category"] != "non_medical"]
    covered = [n for n in covered if n not in set(facts["invalid_lines"])]
    calc = compute_payable(claim, covered)
    for l in claim["lines"]:
        if l["no"] in covered:
            continue
        rule = "R-CODE-PAIR" if l["no"] in facts["invalid_lines"] else "R-NONMED"
        reasons.append({"rule": rule, "text": f"Line {l['no']} ({l['category']}) is not covered.", "cite": [rule]})
    if calc["room_excess"]:
        reasons.append({"rule": "R-ROOM-CAP", "text": "Room rent above the daily cap is not paid.",
                        "cite": ["R-ROOM-CAP", calc["sources"][3]]})
    if calc["package_excess"]:
        reasons.append({"rule": "R-PACKAGE", "text": "The procedure is paid up to its package limit.",
                        "cite": ["R-PACKAGE"]})
    reasons.append({"rule": "R-COST-SHARE", "text": "Deductible and co-payment applied.",
                    "cite": ["R-COST-SHARE", calc["sources"][0], calc["sources"][1]]})
    part = calc["disallowed_amount"] or calc["room_excess"] or calc["package_excess"]
    decision = "part_pay" if part else "approve"
    if facts["near_duplicate_of"]:
        open_issue = {"kind": "near_duplicate", "past_claim_id": facts["near_duplicate_of"],
                      "default": "pend", "if_pay": decision}
    return out(decision, calc["asteria_pays"], reasons)


def resolve_open_issue(result, choice, reason=""):
    """Apply the Adjudicator's choice ('pay' or 'pend'). Anything else keeps the safe default."""
    issue = result["open_issue"]
    if issue is None:
        return result
    pay = choice == "pay"
    result = dict(result, decision=issue["if_pay"] if pay else issue["default"])
    if not pay:
        result["payable"] = None
    result["reasons"] = result["reasons"] + [{"rule": "R-NEAR-DUP", "cite": ["R-NEAR-DUP", issue["past_claim_id"]],
                                              "text": reason or f"Close to past claim {issue['past_claim_id']}."}]
    result["needs_human"] = True
    return result


def gold_facts(claim):
    """The true facts of a claim, found from the tools without any model."""
    f = empty_facts()
    missing = [d for d in data.REQUIRED_DOCS if d not in claim["documents"]]
    f["complete"], f["missing_documents"] = not missing, missing
    h = data.hospitals()[claim["hospital_id"]]
    dx = data.diagnoses()[claim["diagnosis_code"]]
    procs = data.procedures()
    if dx["type"] == "cosmetic":
        f["excluded_rule"] = "R-EXCL-COSMETIC"
    elif not h["network"] and claim["admission_type"] == "planned" and not claim["preauth"]:
        f["excluded_rule"] = "R-NETWORK-PREAUTH"
    ctx = {"claim": claim}
    if not waiting_period_checker(WaitArgs(wait_type="pre_existing"), ctx)["covered"]:
        f["wait_rule"] = "R-WAIT-PED"
    if not waiting_period_checker(WaitArgs(wait_type="maternity"), ctx)["covered"]:
        f["wait_rule"] = "R-WAIT-MAT"
    f["covered_lines"] = [l["no"] for l in claim["lines"] if l["category"] != "non_medical"]
    for l in claim["lines"]:
        if l["category"] == "procedure":
            p = procs[l["code"]]
            if claim["diagnosis_code"] not in p["diagnoses"]:
                f["invalid_lines"].append(l["no"])
            if stay_days(claim) > 2 * p["typical_stay"]:
                f["long_stay"] = True
            if l["amount"] > 2 * package_limit(p, h):
                f["inflated"] = True
    for s in similar_claims(claim):
        if s["exact"]:
            f["duplicate_of"] = s["past_claim_id"]
        elif f["near_duplicate_of"] is None:
            f["near_duplicate_of"], f["near_same_procedure"] = s["past_claim_id"], s["same_procedure"]
    f["watchlist"] = h["watchlist"]
    return f


def gold_decision(claim):
    facts = gold_facts(claim)
    res = apply_rules(claim, facts)
    if res["open_issue"]:
        res = resolve_open_issue(res, "pend" if facts["near_same_procedure"] else "pay")
    return res, facts

