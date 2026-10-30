"""The ClaimDesk tools. Plain Python: no model inside.

Every tool works on ONE claim, held in ctx["claim"], so an agent can never read another member's data.
"""
from datetime import date
from typing import List, Optional

from pydantic import BaseModel, ValidationError

from claimcheck import calculator, tools as cc_tools
from claimcheck.corpus_gen import CURRENT_YEAR, doc_id_for, fact_value, fmt_inr
from claimcheck.guard import sanitize_doc_text
from claimcheck.tools import ToolError

from . import data


# ---------- argument models -----------------------------------------------------
class MemberArgs(BaseModel):
    member_id: str


class HospitalArgs(BaseModel):
    hospital_id: str


class SearchArgs(BaseModel):
    query: str
    topic: Optional[str] = None


class WaitArgs(BaseModel):
    wait_type: str


class CostArgs(BaseModel):
    covered_lines: List[int]


class CodeArgs(BaseModel):
    code: str


class PairArgs(BaseModel):
    procedure_code: str
    diagnosis_code: str
    stay_days: int


class NoArgs(BaseModel):
    pass


class LineArgs(BaseModel):
    line: int


# ---------- helpers -------------------------------------------------------------------
def stay_days(claim):
    a, b = date.fromisoformat(claim["admission_date"]), date.fromisoformat(claim["discharge_date"])
    return max(1, (b - a).days)


def billed_total(claim):
    return sum(l["amount"] for l in claim["lines"])


def _member(ctx, member_id=None):
    claim = ctx["claim"]
    if member_id is not None and member_id.strip().upper() != claim["member_id"]:
        raise ToolError("access denied: you may only read the member on this claim")
    return data.members()[claim["member_id"]]


def _line(claim, no):
    for l in claim["lines"]:
        if l["no"] == no:
            return l
    raise ToolError(f"line {no} is not on this claim; lines are {[l['no'] for l in claim['lines']]}")


def package_limit(proc, hospital):
    return calculator.rupees(proc["package_rate"] * hospital["tariff_band"])


def claim_summary(claim):
    """The claim as text for a prompt. Bill descriptions are hospital text: suspicious lines are removed."""
    dx = data.diagnoses()[claim["diagnosis_code"]]
    procs = data.procedures()
    kind = {"general": "general", "ped": f"linked to the pre-existing condition '{dx['condition']}'",
            "maternity": "maternity", "cosmetic": "cosmetic"}[dx["type"]]
    out = [f"Claim {claim['claim_id']} | member {claim['member_id']} | hospital {claim['hospital_id']} | "
           f"{claim['admission_type']} admission | pre-authorised: {'yes' if claim['preauth'] else 'no'}",
           f"Admission {claim['admission_date']}, discharge {claim['discharge_date']} (stay {stay_days(claim)} days)",
           f"Diagnosis {dx['code']} {dx['name']} (type: {kind})",
           f"Documents received: {', '.join(claim['documents']) or 'none'}",
           "Bill lines:"]
    flagged = False
    for l in claim["lines"]:
        desc, bad = sanitize_doc_text(l["description"])
        flagged = flagged or bad
        if l["category"] == "room":
            out.append(f" {l['no']}. room | {desc} | {l['days']} days x {fmt_inr(l['rate'])} = {fmt_inr(l['amount'])}")
        elif l["category"] == "procedure":
            name = procs[l["code"]]["name"] if l["code"] in procs else "unknown code"
            out.append(f" {l['no']}. procedure {l['code']} ({name}) | {fmt_inr(l['amount'])}")
        else:
            out.append(f" {l['no']}. {l['category']} | {desc} | {fmt_inr(l['amount'])}")
    out.append(f"Total billed: {fmt_inr(billed_total(claim))}")
    out.append("Procedure lines: " + ", ".join(str(l["no"]) for l in claim["lines"] if l["category"] == "procedure"))
    return "\n".join(out), flagged


# ---------- tools -----------------------------------------------------------------------
def get_member_profile(args: MemberArgs, ctx):
    m = _member(ctx, args.member_id)
    return {"member_id": m["member_id"], "tier": m["tier"], "zone": m["zone"],
            "enrolment_date": m["enrolment_date"], "pre_existing_conditions": m["pre_existing"],
            "deductible_met_ytd": m["deductible_met_ytd"], "oop_used_ytd": m["oop_used_ytd"]}


def get_hospital(args: HospitalArgs, ctx):
    hid = args.hospital_id.strip().upper()
    if hid != ctx["claim"]["hospital_id"]:
        raise ToolError("only the hospital on this claim can be looked up")
    h = data.hospitals()[hid]
    return {k: h[k] for k in ("hospital_id", "name", "zone", "network", "tariff_band", "watchlist")}


def search_plan_docs(args: SearchArgs, ctx):
    m = _member(ctx)
    res = cc_tools.search_plan_docs(cc_tools.SearchArgs(query=args.query, tier=m["tier"], zone=m["zone"],
                                                        topic=args.topic), ctx)
    return res


def waiting_period_checker(args: WaitArgs, ctx):
    claim, m = ctx["claim"], _member(ctx)
    topic = {"pre_existing": "ped_wait", "ped": "ped_wait", "maternity": "maternity_wait"}.get(
        args.wait_type.strip().lower().replace("-", "_"))
    if topic is None:
        raise ToolError("wait_type must be 'pre_existing' or 'maternity'")
    dx = data.diagnoses()[claim["diagnosis_code"]]
    applies = dx["type"] == "maternity" if topic == "maternity_wait" else \
        dx["type"] == "ped" and dx["condition"] in m["pre_existing"]
    source = doc_id_for(topic, m["tier"], m["zone"], CURRENT_YEAR)
    if not applies:
        why = ("the diagnosis is not maternity" if topic == "maternity_wait" else
               "the diagnosis is not linked to a pre-existing condition the member declared")
        return {"wait_type": args.wait_type, "applies": False, "covered": True,
                "note": f"No waiting period applies: {why}.", "sources": [source]}
    months = fact_value(topic, m["tier"], m["zone"])
    out = calculator.waiting_period(date.fromisoformat(m["enrolment_date"]), months,
                                    date.fromisoformat(claim["admission_date"]))
    return {"wait_type": args.wait_type, "applies": True, "enrolment_date": m["enrolment_date"],
            "waiting_months": months, "admission_date": claim["admission_date"], **out, "sources": [source]}


def compute_payable(claim, covered_lines):
    """What Asteria pays on the covered lines of a claim. Shared by the tool and the decision rules."""
    m = data.members()[claim["member_id"]]
    h = data.hospitals()[claim["hospital_id"]]
    procs = data.procedures()
    covered = set(covered_lines)
    for no in covered:
        _line(claim, no)
    plan = {t: fact_value(t, m["tier"], m["zone"]) for t in ("deductible", "copay_pct", "oop_max", "room_rent_cap")}
    bill, room_days, room_rate, package_excess, disallowed = 0, 0, 0, 0, 0
    for l in claim["lines"]:
        if l["no"] not in covered:
            disallowed += l["amount"]
            continue
        amount = l["amount"]
        if l["category"] == "room":
            room_days, room_rate = l["days"], l["rate"]
        elif l["category"] == "procedure" and l.get("code") in procs:
            limit = package_limit(procs[l["code"]], h)
            if amount > limit:
                package_excess += amount - limit
                amount = limit
        bill += amount
    out = calculator.cost_share(bill, plan["deductible"], plan["copay_pct"], plan["oop_max"], plan["room_rent_cap"],
                                room_days, room_rate, m["deductible_met_ytd"], m["oop_used_ytd"])
    return {"covered_lines": sorted(covered), "billed_total": billed_total(claim), "disallowed_amount": disallowed,
            "package_excess": package_excess, "room_excess": out["room_excess"],
            "eligible_amount": out["eligible_amount"], "deductible_part": out["deductible_part"],
            "copay": out["copay"], "member_pays_on_covered": out["member_pays"], "asteria_pays": out["asteria_pays"],
            "plan_values": plan,
            "sources": [doc_id_for(t, m["tier"], m["zone"], CURRENT_YEAR) for t in plan]}


def cost_share_calculator(args: CostArgs, ctx):
    return compute_payable(ctx["claim"], args.covered_lines)


def lookup_procedure_code(args: CodeArgs, ctx):
    p = data.procedures().get(args.code.strip().upper())
    if p is None:
        raise ToolError(f"unknown procedure code '{args.code}'")
    return {k: p[k] for k in ("code", "name", "diagnoses", "typical_stay", "package_rate")}


def check_code_pair(args: PairArgs, ctx):
    p = data.procedures().get(args.procedure_code.strip().upper())
    if p is None:
        raise ToolError(f"unknown procedure code '{args.procedure_code}'")
    dx = args.diagnosis_code.strip().upper()
    if dx not in data.diagnoses():
        raise ToolError(f"unknown diagnosis code '{args.diagnosis_code}'")
    valid = dx in p["diagnoses"]
    long_stay = args.stay_days > 2 * p["typical_stay"]
    broken = ([] if valid else ["R-CODE-PAIR"]) + (["R-LOS"] if long_stay else [])
    return {"procedure_code": p["code"], "diagnosis_code": dx, "allowed_diagnoses": p["diagnoses"], "valid": valid,
            "stay_days": args.stay_days, "typical_stay": p["typical_stay"], "long_stay": long_stay,
            "rules_broken": broken}


def similar_claims(claim):
    """Past claims of the same member that look like this one. Shared by the tool and the gold labels."""
    adm = date.fromisoformat(claim["admission_date"])
    total = billed_total(claim)
    proc = next((l.get("code") for l in claim["lines"] if l["category"] == "procedure"), None)
    out = []
    for p in data.past_claims():
        if p["member_id"] != claim["member_id"] or p["hospital_id"] != claim["hospital_id"]:
            continue
        days = (adm - date.fromisoformat(p["admission_date"])).days
        if abs(days) > 7:
            continue
        ratio = round(total / p["billed_total"], 3)
        exact = days == 0 and abs(ratio - 1) <= 0.01
        out.append({"past_claim_id": p["claim_id"], "admission_date": p["admission_date"], "days_apart": days,
                    "procedure_code": p["procedure_code"], "same_procedure": p["procedure_code"] == proc,
                    "billed_total": p["billed_total"], "amount_ratio": ratio, "outcome": p["outcome"],
                    "exact": exact})
    return out


def find_similar_claims(args: NoArgs, ctx):
    found = similar_claims(ctx["claim"])
    return {"matches": found, "note": "exact true = same hospital, admission date and amount (a duplicate)"
            if found else "no past claim of this member is within 7 days at this hospital"}


def tariff_check(args: LineArgs, ctx):
    claim = ctx["claim"]
    l = _line(claim, args.line)
    if l["category"] != "procedure":
        raise ToolError(f"line {args.line} is not a procedure line")
    p = data.procedures().get(l["code"])
    if p is None:
        raise ToolError(f"unknown procedure code '{l['code']}'")
    h = data.hospitals()[claim["hospital_id"]]
    limit = package_limit(p, h)
    ratio = round(l["amount"] / limit, 2)
    return {"line": l["no"], "procedure_code": p["code"], "billed": l["amount"], "package_rate": p["package_rate"],
            "tariff_band": h["tariff_band"], "package_limit": limit, "ratio": ratio, "over_2x": ratio > 2}


# ---------- registry -----------------------------------------------------------------------
TOOLS = {
    "get_member_profile": (MemberArgs, get_member_profile),
    "get_hospital": (HospitalArgs, get_hospital),
    "search_plan_docs": (SearchArgs, search_plan_docs),
    "waiting_period_checker": (WaitArgs, waiting_period_checker),
    "cost_share_calculator": (CostArgs, cost_share_calculator),
    "lookup_procedure_code": (CodeArgs, lookup_procedure_code),
    "check_code_pair": (PairArgs, check_code_pair),
    "find_similar_claims": (NoArgs, find_similar_claims),
    "tariff_check": (LineArgs, tariff_check),
}

TOOL_HELP = {
    "get_member_profile": 'get_member_profile {"member_id": str}: tier, zone, enrolment date, declared pre-existing conditions, deductible and out-of-pocket used this year.',
    "get_hospital": 'get_hospital {"hospital_id": str}: zone, network (true/false), tariff band, watchlist (true/false).',
    "search_plan_docs": 'search_plan_docs {"query": str, "topic": optional topic key}: plan documents for the member\'s plan.',
    "waiting_period_checker": 'waiting_period_checker {"wait_type": "pre_existing" or "maternity"}: whether a waiting period applies to this claim and, if it does, whether it was served on the admission date (covered true/false).',
    "cost_share_calculator": 'cost_share_calculator {"covered_lines": [line numbers]}: what Asteria pays on those lines (room rent cap, package rates, deductible and co-pay applied). asteria_pays is the payable amount.',
    "lookup_procedure_code": 'lookup_procedure_code {"code": str}: procedure name, allowed diagnoses, typical stay, package rate.',
    "check_code_pair": 'check_code_pair {"procedure_code": str, "diagnosis_code": str, "stay_days": int}: valid (procedure allowed for the diagnosis) and long_stay.',
    "find_similar_claims": 'find_similar_claims {}: past claims of this member at this hospital within 7 days (exact true = duplicate).',
    "tariff_check": 'tariff_check {"line": int}: a procedure line\'s billed amount against its package limit (over_2x true/false).',
}


def run_tool(name, raw_args, ctx, allowed=None):
    if name not in TOOLS or (allowed is not None and name not in allowed):
        valid = allowed if allowed is not None else list(TOOLS)
        raise ToolError(f"unknown tool '{name}'. Your tools: {', '.join(valid)}")
    model, fn = TOOLS[name]
    try:
        args = model.model_validate(raw_args or {})
    except ValidationError as e:
        raise ToolError(f"bad arguments for {name}: {str(e)[:200]}")
    return fn(args, ctx)
