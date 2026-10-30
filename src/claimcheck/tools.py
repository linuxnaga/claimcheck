"""The five tools the agent can call. Plain Python: no model inside."""
import re
import sqlite3
from datetime import date
from typing import Optional

from pydantic import BaseModel, ValidationError

from . import calculator, config
from .corpus_gen import (CURRENT_YEAR, TIERS, TOPICS, YEARS, ZONES, doc_id_for, fact_value, load_claims,
                         load_members)
from .guard import sanitize_doc_text
from .tracing import mask_pii


class ToolError(Exception):
    """Raised when a tool call is wrong. The message is shown to the planner so it can fix the call."""


# ---------- argument models -----------------------------------------------------
class SearchArgs(BaseModel):
    query: str
    tier: Optional[str] = None
    zone: Optional[str] = None
    topic: Optional[str] = None
    year: Optional[str] = None


class ProfileArgs(BaseModel):
    member_id: str


class CostArgs(BaseModel):
    bill_amount: float
    room_days: float = 0
    room_rate_charged: float = 0
    member_id: Optional[str] = None      # fills tier, zone and year-to-date usage from the profile
    tier: Optional[str] = None
    zone: Optional[str] = None
    year: Optional[str] = None
    deductible: Optional[float] = None   # explicit plan values override the plan schedule
    copay_pct: Optional[float] = None
    oop_max: Optional[float] = None
    room_rent_cap: Optional[float] = None
    deductible_met: Optional[float] = None
    oop_used: Optional[float] = None


class WaitArgs(BaseModel):
    service_date: Optional[str] = None
    member_id: Optional[str] = None      # start date = enrolment date, months from the member's plan
    wait_type: Optional[str] = None      # maternity or pre_existing
    start_date: Optional[str] = None
    waiting_months: Optional[int] = None


class AppealArgs(BaseModel):
    member_id: str
    claim_id: str
    reason: str


# ---------- normalising what the model writes -------------------------------------
_TIER_ALIASES = {**{k: k for k in TIERS}, **{f"{k} plan": k for k in TIERS}}
_ZONE_ALIASES = {**{z.lower(): z for z in ZONES}, **{f"zone {z.lower()}": z for z in ZONES},
                 **{v: k for k, v in ZONES.items()}}
_TOPIC_ALIASES = {"copay": "copay_pct", "co_pay": "copay_pct", "co_payment": "copay_pct", "copayment": "copay_pct",
                  "room_rent": "room_rent_cap", "room": "room_rent_cap", "icu": "icu_cap", "opd": "opd_visits",
                  "outpatient": "opd_visits", "diagnostics": "diagnostics_limit", "maternity": "maternity_limit",
                  "ambulance": "ambulance_limit", "dental": "dental_limit", "mental_health": "mental_health_sessions",
                  "therapy": "mental_health_sessions", "pre_existing": "ped_wait", "out_of_pocket": "oop_max",
                  "out_of_pocket_maximum": "oop_max", "annual_deductible": "deductible"}
_WAIT_TOPIC = {"maternity": "maternity_wait", "pre_existing": "ped_wait", "ped": "ped_wait",
               "maternity_wait": "maternity_wait", "ped_wait": "ped_wait", "pre_existing_disease": "ped_wait"}


def _key(x):
    return re.sub(r"[\s\-]+", "_", str(x).strip().lower())


def norm_tier(x):
    if x is None:
        return None
    k = str(x).strip().lower()
    if k not in _TIER_ALIASES:
        raise ToolError(f"unknown tier '{x}'. Valid tiers: bronze, silver, gold, platinum")
    return _TIER_ALIASES[k]


def norm_zone(x):
    if x is None:
        return None
    k = str(x).strip().lower()
    if k not in _ZONE_ALIASES:
        raise ToolError(f"unknown zone '{x}'. Valid zones: A, B, C, D, E")
    return _ZONE_ALIASES[k]


def norm_topic(x):
    if x is None:
        return None
    k = _TOPIC_ALIASES.get(_key(x), _key(x))
    if k not in TOPICS:
        raise ToolError(f"unknown topic '{x}'. Valid topics: {', '.join(TOPICS)}")
    return k


def norm_year(x):
    if x is None:
        return None
    if str(x) not in YEARS:
        raise ToolError("year must be '2024' or '2025' (leave it empty for the current plan)")
    return str(x)


def _date(s, name):
    try:
        return date.fromisoformat(str(s))
    except Exception:
        raise ToolError(f"{name} must be a date like 2025-09-15")


def essential_text(text):
    """Keep the title and plan lines and sections 2-3 (the part that holds the facts)."""
    lines = text.splitlines()
    head = [l for l in lines[:5] if l.startswith(("Title:", "Plan tier:"))]
    m = re.search(r"2\. Entitlement.*?(?=\n4\. Exclusions)", text, re.S)
    body = m.group(0) if m else "\n".join(lines[5:14])
    return "\n".join(head) + "\n" + body.strip()


# ---------- tool 1: search_plan_docs -----------------------------------------------
_RETRIEVER = None


def set_retriever(r):
    global _RETRIEVER
    _RETRIEVER = r


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        from .retrieval import load_retriever
        _RETRIEVER = load_retriever()
    return _RETRIEVER


def search_plan_docs(args: SearchArgs, ctx):
    r = get_retriever()
    filters = {"tier": norm_tier(args.tier), "zone": norm_zone(args.zone), "topic": norm_topic(args.topic)}
    year = norm_year(args.year)
    if year:
        filters["year"] = year
    else:
        filters["status"] = "CURRENT"
    mode = "hybrid" if r.emb is not None else "bm25"
    ids = r.search_docs(args.query, mode=mode, k=3, filters=filters, rerank=True)
    results, flagged = [], False
    for did in ids:
        d = r.docs[did]
        text, bad = sanitize_doc_text(essential_text(d["text"]))
        flagged = flagged or bad
        results.append({"doc_id": did, "tier": d["tier"], "zone": d["zone"], "year": d["year"],
                        "status": d["status"], "text": text})
    if not results:
        raise ToolError("no documents matched these filters; try fewer filters")
    ctx.setdefault("filters", {}).update({k: v for k, v in filters.items() if v and k in ("tier", "zone", "topic")})
    if flagged:
        ctx["injection_flags"] = ctx.get("injection_flags", 0) + 1
    return {"documents": results}


# ---------- tool 2: get_member_profile ---------------------------------------------
def _own_member(member_id, ctx):
    user = ctx.get("user_id")
    if user is None:
        raise ToolError("nobody is signed in; ask the user for their plan tier and zone instead")
    if member_id != user:
        raise ToolError("access denied: you may only read the signed-in member's own data")
    for m in load_members():
        if m["member_id"] == user:
            return m
    raise ToolError("member not found")


def get_member_profile(args: ProfileArgs, ctx):
    m = _own_member(args.member_id, ctx)
    ctx.setdefault("filters", {}).update({"tier": m["tier"], "zone": m["zone"]})
    return {"member_id": m["member_id"], "tier": m["tier"], "zone": m["zone"],
            "enrolment_date": m["enrolment_date"], "pre_existing_conditions": m["pre_existing"],
            "deductible_met_ytd": m["deductible_met_ytd"], "oop_used_ytd": m["oop_used_ytd"],
            "opd_visits_used_ytd": m["opd_visits_used_ytd"],
            "claims": [{k: c[k] for k in ("claim_id", "date", "amount", "status", "reason")}
                       for c in load_claims() if c["member_id"] == m["member_id"]]}


# ---------- tool 3: cost_share_calculator ------------------------------------------
def cost_share_calculator(args: CostArgs, ctx):
    tier, zone = norm_tier(args.tier), norm_zone(args.zone)
    deductible_met, oop_used = args.deductible_met, args.oop_used
    if args.member_id:
        m = _own_member(args.member_id, ctx)
        tier, zone = m["tier"], m["zone"]
        deductible_met = m["deductible_met_ytd"] if deductible_met is None else deductible_met
        oop_used = m["oop_used_ytd"] if oop_used is None else oop_used
    year = norm_year(args.year) or CURRENT_YEAR
    plan, sources = {}, []
    for topic, given in (("deductible", args.deductible), ("copay_pct", args.copay_pct),
                         ("oop_max", args.oop_max), ("room_rent_cap", args.room_rent_cap)):
        if given is not None:
            plan[topic] = given
        elif tier and zone:
            plan[topic] = fact_value(topic, tier, zone, year)
            sources.append(doc_id_for(topic, tier, zone, year))
        else:
            raise ToolError(f"missing {topic}: give member_id, or tier and zone, or the value itself")
    if args.room_days and not args.room_rate_charged:
        raise ToolError("room_days was given without room_rate_charged")
    try:
        out = calculator.cost_share(args.bill_amount, plan["deductible"], plan["copay_pct"], plan["oop_max"],
                                    plan["room_rent_cap"], args.room_days, args.room_rate_charged,
                                    deductible_met or 0, oop_used or 0)
    except ValueError as e:
        raise ToolError(str(e))
    return {"bill_amount": args.bill_amount, "plan_values": plan, "deductible_met": deductible_met or 0,
            "oop_used": oop_used or 0, **out, "sources": sources}


# ---------- tool 4: waiting_period_checker -----------------------------------------
def waiting_period_checker(args: WaitArgs, ctx):
    service = _date(args.service_date or config.AS_OF_DATE, "service_date")
    start, months, sources = args.start_date, args.waiting_months, []
    if args.member_id:
        m = _own_member(args.member_id, ctx)
        start = start or m["enrolment_date"]
        if months is None:
            topic = _WAIT_TOPIC.get(_key(args.wait_type or ""))
            if topic is None:
                raise ToolError("wait_type must be 'maternity' or 'pre_existing'")
            months = fact_value(topic, m["tier"], m["zone"])
            sources.append(doc_id_for(topic, m["tier"], m["zone"], CURRENT_YEAR))
    if start is None or months is None:
        raise ToolError("give member_id and wait_type, or start_date and waiting_months")
    start_d = _date(start, "start_date")
    if start_d > service:
        raise ToolError("start_date is after service_date")
    out = calculator.waiting_period(start_d, int(months), service)
    return {"start_date": start_d.isoformat(), "waiting_months": int(months), "service_date": service.isoformat(),
            **out, "covered_value": 1 if out["covered"] else 0, "sources": sources}


# ---------- tool 5: create_appeal_draft --------------------------------------------
def _db():
    config.ensure_dirs()
    con = sqlite3.connect(config.DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS appeals (case_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "member_id TEXT, claim_id TEXT, reason TEXT, status TEXT, reviewer TEXT)")
    return con


def create_appeal_draft(args: AppealArgs, ctx):
    _own_member(args.member_id, ctx)
    claim = next((c for c in load_claims() if c["claim_id"] == args.claim_id.strip().upper()), None)
    if claim is None or claim["member_id"] != args.member_id:
        raise ToolError("claim not found for this member")
    if not args.reason.strip():
        raise ToolError("reason must not be empty")
    reason = mask_pii(args.reason)[:300]
    with _db() as con:
        cur = con.execute("INSERT INTO appeals (member_id, claim_id, reason, status) VALUES (?,?,?,?)",
                          (args.member_id, claim["claim_id"], reason, "DRAFT"))
        case_id = cur.lastrowid
    ctx["case_id"] = case_id
    return {"case_id": case_id, "claim_id": claim["claim_id"], "status": "DRAFT",
            "note": "Draft saved. A claims reviewer must confirm it before it is submitted."}


def list_draft_appeals():
    with _db() as con:
        rows = con.execute("SELECT case_id, member_id, claim_id, reason FROM appeals WHERE status='DRAFT' "
                           "ORDER BY case_id").fetchall()
    return [dict(zip(("case_id", "member_id", "claim_id", "reason"), r)) for r in rows]


def review_appeal(case_id: int, reviewer: str, decision: str):
    """Called ONLY by the app after a claims reviewer clicks Confirm or Reject. The agent cannot call this."""
    if decision not in ("SUBMITTED", "REJECTED"):
        raise ToolError("decision must be SUBMITTED or REJECTED")
    with _db() as con:
        row = con.execute("SELECT status FROM appeals WHERE case_id=?", (case_id,)).fetchone()
        if not row or row[0] != "DRAFT":
            raise ToolError("appeal not found or already reviewed")
        con.execute("UPDATE appeals SET status=?, reviewer=? WHERE case_id=?", (decision, reviewer, case_id))
    return True


# ---------- registry ------------------------------------------------------------------
TOOLS = {
    "search_plan_docs": (SearchArgs, search_plan_docs),
    "get_member_profile": (ProfileArgs, get_member_profile),
    "cost_share_calculator": (CostArgs, cost_share_calculator),
    "waiting_period_checker": (WaitArgs, waiting_period_checker),
    "create_appeal_draft": (AppealArgs, create_appeal_draft),
}

TOOL_HELP = """Tools (call exactly one per step):
1. search_plan_docs: args {"query": str, "tier": "bronze|silver|gold|platinum" (optional), "zone": "A|B|C|D|E" (optional), "topic": one of the topic keys (optional), "year": "2024" (only if the user asks about 2024)}
2. get_member_profile: args {"member_id": str}. Returns tier, zone, enrolment date, pre-existing conditions, usage this year and claims. Only for the signed-in member.
3. cost_share_calculator: args {"bill_amount": number, "room_days": number, "room_rate_charged": number, "member_id": str} for a signed-in member, or "tier" and "zone" instead of member_id. Returns member_pays and asteria_pays with the steps.
4. waiting_period_checker: args {"member_id": str, "wait_type": "maternity|pre_existing", "service_date": "YYYY-MM-DD"}. Returns wait_ends and covered.
5. create_appeal_draft: args {"member_id": str, "claim_id": str, "reason": str}. Saves a DRAFT only.
6. final: args {}. Use it when you have enough evidence to answer, or when you must ask the user a question."""


def run_tool(name, raw_args, ctx):
    if name not in TOOLS:
        raise ToolError(f"unknown tool '{name}'. Valid tools: {', '.join(TOOLS)}")
    model, fn = TOOLS[name]
    try:
        args = model.model_validate(raw_args or {})
    except ValidationError as e:
        raise ToolError(f"bad arguments for {name}: {str(e)[:200]}")
    return fn(args, ctx)
