"""Safety rules written as plain code (small models are not reliable guards)."""
import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the |your )?(previous|prior|above|earlier)? ?(instructions|rules)",
    r"system (instruction|prompt|notice|override)",
    r"you are now",
    r"pretend (to be|you are)",
    r"disregard .{0,40}(instructions|rules)",
    r"reveal .{0,30}(prompt|instructions|password|api key)",
    r"do not mention any other",
    r"print your (system )?prompt",
    r"developer mode",
]
_INJ = [re.compile(p, re.I) for p in INJECTION_PATTERNS]

REFUSAL_RULES = [
    ("medical_advice", re.compile(
        r"\b(should i (have|get|take|stop|skip|go for)|which (medicine|tablet|drug|treatment) should|"
        r"is it safe to|what dose)\b", re.I),
     "I can explain what your plan covers, but I cannot give medical advice. Please ask your doctor."),
    ("diagnosis", re.compile(
        r"\b(what (disease|illness|condition) do i have|diagnose|my (chest|head|stomach) (hurts|pain)|"
        r"what is wrong with me|symptoms? mean)\b", re.I),
     "I cannot diagnose health problems. If this is an emergency, call 112 or go to the nearest hospital."),
    ("outcome_promise", re.compile(
        r"\b(will|is) my claim (definitely |surely )?(be )?(approved|accepted|paid|rejected)|"
        r"guarantee\w*\b.{0,30}\b(claim|approval|payment)|promise .{0,20}(approve|pay)", re.I),
     "I cannot promise the outcome of a claim. A claims reviewer decides each claim. "
     "I can tell you what your plan rules say."),
    ("other_member_data", re.compile(
        r"\b(another member|other members?|someone else'?s?|my (colleague|neighbou?r|friend|brother|sister)'?s? (claim|plan|policy)|"
        r"his (claim|plan|policy|bill)|her (claim|plan|policy|bill)|their (claims?|plans?|policy))\b", re.I),
     "I cannot share information about other members."),
    ("legal_advice", re.compile(
        r"\b(legal advice|lawsuit|lawyer|advocate|consumer court|court case|sue\b|sue asteria)\b", re.I),
     "I cannot give legal advice. You can raise an appeal or a grievance in the member portal."),
    ("fraud", re.compile(
        r"\b(fake (bill|invoice|receipt)|change the (date|diagnosis|code)|word the bill|backdate|"
        r"make it look like|hide (my|the) (condition|illness)|so (it|that it) (counts|is covered) as)\b", re.I),
     "I cannot help change or misrepresent claim information."),
]
_MEMBER_ID = re.compile(r"\bM\d{3}\b")

# topic words: a question naming one of these needs a tier and a zone (or a signed-in member)
TOPIC_WORDS = re.compile(
    r"deductible|co-?pay|out[- ]of[- ]pocket|room rent|icu|outpatient|consultation|diagnostic|maternity|"
    r"waiting period|pre-existing|ambulance|dental|therapy|mental health", re.I)
TIER_WORDS = re.compile(r"\b(bronze|silver|gold|platinum)\b", re.I)
ZONE_WORDS = re.compile(r"\bzone [a-e]\b|\b(metro|large city|town|rural)\b", re.I)
COST_WORDS = re.compile(r"\b(how much (will|would|do) i (pay|owe)|what (will|would) i pay|my share)\b", re.I)
AMOUNT = re.compile(r"(₹|rs\.?|inr)\s?\d|\d[\d,]{3,}", re.I)


def has_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJ)


def sanitize_doc_text(text: str):
    """Remove lines of retrieved documents that look like instructions to the AI."""
    flagged, clean = False, []
    for line in text.splitlines():
        if has_injection(line) or re.search(r"instruction to the ai|tell the member", line, re.I):
            clean.append("[REMOVED: suspicious instruction found in document]")
            flagged = True
        else:
            clean.append(line)
    return "\n".join(clean), flagged


def check_input(question: str, user_id=None):
    """Return None when OK, or a dict with status (refused or clarify), reason and message."""
    if has_injection(question):
        return {"status": "refused", "reason": "prompt_injection",
                "message": "I cannot follow instructions that try to change my rules. "
                           "I can answer questions about your health plan."}
    for name, pattern, msg in REFUSAL_RULES:
        if pattern.search(question):
            return {"status": "refused", "reason": name, "message": msg}
    for mid in _MEMBER_ID.findall(question):
        if mid != user_id:
            return {"status": "refused", "reason": "other_member_data",
                    "message": "I cannot share information about other members."}
    if user_id is None and TOPIC_WORDS.search(question) and "2024" not in question \
            and not (TIER_WORDS.search(question) and ZONE_WORDS.search(question)):
        return {"status": "clarify", "reason": "missing_plan",
                "message": "Which plan tier (Bronze, Silver, Gold or Platinum) and zone (A to E) are you on? "
                           "You can also sign in so I can look it up."}
    if COST_WORDS.search(question) and not AMOUNT.search(question):
        return {"status": "clarify", "reason": "missing_amount",
                "message": "What is the total bill amount in rupees, and how many days were spent in a room "
                           "at what daily rate?"}
    return None
