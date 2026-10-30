"""Reference tables for ClaimDesk: rules, diagnoses, procedure codes, hospitals and members.

Everything is fictional and generated from a fixed seed. Codes are NOT real ICD or CPT codes.
"""
import json
import random
from datetime import date

from claimcheck import config
from claimcheck.corpus_gen import MEMBERS as CC_MEMBERS
from claimcheck.corpus_gen import TIERS, ZONES, fact_value

SEED = 42
REQUIRED_DOCS = ("itemised_bill", "discharge_summary")
HUMAN_REVIEW_ABOVE = 100000      # payouts above this need a human (rupees)
PED_CONDITIONS = ["diabetes", "hypertension", "asthma", "thyroid disorder", "arthritis"]

# Rule ids that agents and decisions cite.
RULES = {
    "R-DOCS": "An itemised bill and a discharge summary are needed before a claim is decided.",
    "R-EXCL-COSMETIC": "Cosmetic treatment is not covered (plan exclusions, section 4).",
    "R-NETWORK-PREAUTH": "Planned treatment at a non-network hospital is covered only with pre-authorisation.",
    "R-WAIT-PED": "A declared pre-existing condition is covered only after the pre-existing disease waiting period.",
    "R-WAIT-MAT": "Maternity treatment is covered only after the maternity waiting period.",
    "R-NONMED": "Non-medical items such as toiletries are not covered (plan exclusions, section 4).",
    "R-CODE-PAIR": "A procedure is paid only when it is allowed for the diagnosis.",
    "R-LOS": "A stay longer than twice the typical stay for the procedure is sent for human review.",
    "R-PACKAGE": "A procedure is paid up to its package rate times the hospital's tariff band.",
    "R-ROOM-CAP": "Room rent above the plan's daily cap is paid by the member.",
    "R-COST-SHARE": "The deductible and co-payment apply to the eligible amount.",
    "R-TARIFF": "A procedure billed above twice its package rate (times the tariff band) is flagged for review.",
    "R-DUP": "A bill that repeats an earlier claim (same member, hospital, admission date and amount) is not paid again.",
    "R-NEAR-DUP": "A claim close to an earlier one is held for review unless it is a follow-up for a different procedure.",
    "R-WATCH": "Claims from a watch-list hospital are sent for human review.",
    "R-HIGH-VALUE": "Payouts above ₹1,00,000 are confirmed by a human reviewer.",
}

# (name, type, linked pre-existing condition). type: general, ped, maternity, cosmetic
DIAGNOSES = [
    ("Acute appendicitis", "general", None), ("Gallstones", "general", None), ("Inguinal hernia", "general", None),
    ("Kidney stones", "general", None), ("Cataract", "general", None), ("Fractured forearm", "general", None),
    ("Fractured hip", "general", None), ("Dengue fever", "general", None), ("Typhoid fever", "general", None),
    ("Malaria", "general", None), ("Pneumonia", "general", None), ("Acute gastroenteritis", "general", None),
    ("Coronary artery disease", "general", None), ("Heart attack", "general", None), ("Stroke", "general", None),
    ("Tonsillitis", "general", None), ("Sinusitis", "general", None), ("Migraine", "general", None),
    ("Urinary tract infection", "general", None), ("Hepatitis A", "general", None),
    ("Diabetes complications", "ped", "diabetes"), ("Hypertensive crisis", "ped", "hypertension"),
    ("Severe asthma attack", "ped", "asthma"), ("Thyroid disorder treatment", "ped", "thyroid disorder"),
    ("Knee osteoarthritis", "ped", "arthritis"),
    ("Normal delivery", "maternity", None), ("Caesarean delivery", "maternity", None),
    ("Cosmetic nose reshaping", "cosmetic", None), ("Hair transplant", "cosmetic", None),
    ("Slipped disc", "general", None), ("Varicose veins", "general", None), ("Piles", "general", None),
    ("Chikungunya", "general", None), ("Burns", "general", None), ("Road accident injuries", "general", None),
    ("Dog bite", "general", None), ("Ear infection", "general", None), ("Anaemia", "general", None),
    ("Bronchitis", "general", None), ("Food poisoning", "general", None),
]
# (suffix, label, package rate range, typical stay range in days)
PROC_KINDS = [("S", "surgical treatment", (60000, 250000), (3, 6)),
              ("M", "medical management", (20000, 80000), (2, 4)),
              ("D", "day-care procedure", (10000, 40000), (1, 1))]

CITIES = {"A": ["Mumbai", "Delhi", "Bengaluru", "Chennai"], "B": ["Pune", "Hyderabad", "Ahmedabad", "Kolkata"],
          "C": ["Kochi", "Jaipur", "Lucknow", "Indore"], "D": ["Mysuru", "Nashik", "Udupi", "Guntur"],
          "E": ["Wayanad", "Bastar", "Ziro", "Kutch"]}
TARIFF_BAND = {"A": 1.3, "B": 1.2, "C": 1.1, "D": 1.0, "E": 1.0}

FIRST_F = ["Aditi", "Bhavna", "Chitra", "Deepa", "Esha", "Gauri", "Hema", "Isha", "Jaya", "Kavya", "Leela",
           "Maya", "Nisha", "Pooja", "Radha", "Sita", "Tara", "Uma", "Vani", "Zoya", "Neha", "Rekha"]
FIRST_M = ["Abhay", "Bala", "Chetan", "Dev", "Gopal", "Hari", "Jatin", "Kiran", "Manoj", "Naveen", "Om",
           "Pranav", "Rohit", "Sanjay", "Tarun", "Uday", "Varun", "Yash", "Ajay", "Mohan", "Nikhil", "Ramesh", "Sunil"]
LAST = ["Agarwal", "Banerjee", "Chopra", "Desai", "Fernandes", "Ghosh", "Hegde", "Joshi", "Kapoor", "Menon",
        "Naidu", "Patel", "Rao", "Saxena", "Thomas", "Varma", "Yadav", "Sethi", "Mehta", "Bhat"]
CC_SEX = {"M001": "F", "M002": "M", "M003": "M", "M004": "F", "M005": "M", "M006": "F", "M007": "M", "M008": "F",
          "M009": "M", "M010": "F", "M011": "M", "M012": "F", "M013": "F", "M014": "M", "M015": "F"}


def claims_dir():
    return config.DATA / "claims"


def _rand_date(r, lo, hi):
    a, b = date.fromisoformat(lo).toordinal(), date.fromisoformat(hi).toordinal()
    return date.fromordinal(r.randint(a, b)).isoformat()


def build_codes():
    r = random.Random(f"{SEED}|codes")
    diagnoses = [dict(code=f"DX{i:02d}", name=n, type=t, condition=c) for i, (n, t, c) in enumerate(DIAGNOSES, 1)]
    procedures = []
    for i, d in enumerate(diagnoses):
        for j, (kind, label, (lo, hi), (slo, shi)) in enumerate(PROC_KINDS):
            procedures.append(dict(code=f"PR{3 * i + j + 1:03d}", name=f"{d['name']}: {label}", kind=kind,
                                   diagnoses=[d["code"]], typical_stay=r.randint(slo, shi),
                                   package_rate=r.randrange(lo, hi + 1, 500)))
    return diagnoses, procedures


def build_hospitals():
    r = random.Random(f"{SEED}|hospitals")
    rows = []
    for zone, cities in CITIES.items():
        for city in cities:
            for suffix in ("General Hospital", "Care Hospital"):
                rows.append(dict(hospital_id=f"H{len(rows) + 1:02d}", name=f"{city} {suffix}", city=city, zone=zone,
                                 network=True, tariff_band=TARIFF_BAND[zone], watchlist=False))
    for h in r.sample(rows, 8):
        h["network"] = False
    for h in r.sample([h for h in rows if h["network"]], 4):
        h["watchlist"] = True
    return rows


def build_members():
    """The 15 ClaimCheck members plus 45 generated ones (M016-M060). No real people."""
    r = random.Random(f"{SEED}|members")
    keep = ("member_id", "name", "tier", "zone", "enrolment_date", "pre_existing", "deductible_met_ytd", "oop_used_ytd")
    rows = [dict({k: m[k] for k in keep}, sex=CC_SEX[m["member_id"]]) for m in CC_MEMBERS]
    for i in range(16, 61):
        sex = r.choice("FM")
        tier, zone = r.choice(list(TIERS)), r.choice(list(ZONES))
        enrolled = _rand_date(r, "2018-01-01", "2025-06-30")
        ped = [r.choice(PED_CONDITIONS)] if r.random() < 0.45 else []
        ded = fact_value("deductible", tier, zone)
        met = r.choice([0, 0, ded // 2, ded])
        rows.append(dict(member_id=f"M{i:03d}", name=f"{r.choice(FIRST_F if sex == 'F' else FIRST_M)} {r.choice(LAST)}",
                         tier=tier, zone=zone, enrolment_date=enrolled, pre_existing=ped, deductible_met_ytd=met,
                         oop_used_ytd=met + r.choice([0, 2000, 5000, 8000]), sex=sex))
    return rows


def write_tables():
    d = claims_dir()
    d.mkdir(parents=True, exist_ok=True)
    diagnoses, procedures = build_codes()
    (d / "codes.json").write_text(json.dumps({"diagnoses": diagnoses, "procedures": procedures}, indent=1))
    (d / "hospitals.json").write_text(json.dumps(build_hospitals(), indent=1))
    (d / "members.json").write_text(json.dumps(build_members(), indent=1, ensure_ascii=False))


# ---------- loaders (cached per data directory) ------------------------------------
_CACHE = {}
OVERRIDE = {}   # the generator puts its in-progress past claims here


def _load(name):
    path = claims_dir() / name
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _CACHE:
        p = path.read_text(encoding="utf-8")
        _CACHE[key] = [json.loads(l) for l in p.splitlines() if l.strip()] if name.endswith(".jsonl") else json.loads(p)
    return _CACHE[key]


def members():
    return {m["member_id"]: m for m in _load("members.json")}


def hospitals():
    return {h["hospital_id"]: h for h in _load("hospitals.json")}


def diagnoses():
    return {d["code"]: d for d in _load("codes.json")["diagnoses"]}


def procedures():
    return {p["code"]: p for p in _load("codes.json")["procedures"]}


def past_claims():
    return OVERRIDE["past_claims"] if "past_claims" in OVERRIDE else _load("past_claims.json")


def test_claims(split="test"):
    """split: 'test' (150 frozen claims, official runs only) or 'dev' (33 claims for tuning prompts)."""
    return _load(f"{split}_claims.jsonl")


def test_gold(split="test"):
    return {g["claim_id"]: g for g in _load(f"{split}_gold.jsonl")}
