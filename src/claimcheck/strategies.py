"""Week 1: three prompting strategies compared on the same questions, against a closed-book baseline."""
from collections import Counter

from .agent import Answer
from .llm import LLMError, Usage, llm

STRATEGIES = ["closed_book", "fewshot_json", "cot_sc", "decompose"]

BASE = ("You are a health-plan benefits assistant for Asteria Health in India. Reply with ONLY one JSON object "
        "with keys: status (\"answered\"), answer_text (string), value (number or null, no commas), "
        "unit (string or null), breakdown (object or null), citations (list of doc ids). "
        "Use ONLY the documents given. If the benefit is not covered, value is 0. Amounts are in Indian rupees.")

SHOTS = [
    ("BEN-SILVER-B-2025-deductible",
     "Title: Annual Deductible\nPlan tier: Silver | Zone: B (large city) | Plan year: 2025 | Status: CURRENT\n"
     "2. Entitlement\nThe member pays the first ₹17,500 of eligible hospital bills in each policy year before "
     "cost-sharing starts.\n\n3. Conditions\nDay-care procedures have a separate deductible of ₹3,000 per procedure.",
     "What is the annual deductible on the Silver plan in zone B?",
     '{"status": "answered", "answer_text": "On the Silver plan in zone B the annual deductible is ₹17,500.", '
     '"value": 17500, "unit": "INR_per_year", "breakdown": null, "citations": ["BEN-SILVER-B-2025-deductible"]}'),
    ("BEN-GOLD-D-2025-ambulance_limit",
     "Title: Road Ambulance\nPlan tier: Gold | Zone: D (town) | Plan year: 2025 | Status: CURRENT\n"
     "2. Entitlement\nRoad ambulance charges are covered up to ₹3,500 per trip.\n\n"
     "3. Conditions\nAt most 4 ambulance trips are covered per policy year.",
     "How much does the Gold plan in zone D pay per ambulance trip?",
     '{"status": "answered", "answer_text": "The Gold plan in zone D pays up to ₹3,500 per ambulance trip.", '
     '"value": 3500, "unit": "INR_per_trip", "breakdown": null, "citations": ["BEN-GOLD-D-2025-ambulance_limit"]}'),
]


class CoTAnswer(Answer):
    reasoning: str = ""


def doc_block(doc_id, text):
    return f"<doc id=\"{doc_id}\">\n{text}\n</doc>"


def answer_with(strategy, question, context, usage: Usage):
    """context: the documents (and member data) as text. Returns an Answer, or None when the model never
    produced valid JSON (retries=0, so JSON failures are counted as they happen)."""
    try:
        if strategy == "closed_book":
            msgs = [{"role": "system", "content": BASE}, {"role": "user", "content": question}]
            return llm.chat_json(msgs, Answer, usage, retries=0)
        if strategy == "fewshot_json":
            msgs = [{"role": "system", "content": BASE}]
            for doc_id, text, q, a in SHOTS:
                msgs += [{"role": "user", "content": f"{doc_block(doc_id, text)}\nQuestion: {q}"},
                         {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": f"{context}\nQuestion: {question}"})
            return llm.chat_json(msgs, Answer, usage, retries=0)
        if strategy == "cot_sc":   # chain of thought + self-consistency (3 samples, majority vote)
            sys = BASE + " First think step by step in the key 'reasoning', then fill the other keys."
            msgs = [{"role": "system", "content": sys}, {"role": "user", "content": f"{context}\nQuestion: {question}"}]
            answers = []
            for _ in range(3):
                try:
                    answers.append(llm.chat_json(msgs, CoTAnswer, usage, retries=0, temperature=0.7, max_tokens=700))
                except LLMError:
                    pass
            if not answers:
                return None
            top = Counter(a.value for a in answers).most_common(1)[0][0]
            return next(a for a in answers if a.value == top)
        if strategy == "decompose":   # step 1: list the numbers, step 2: answer from the list
            m1 = [{"role": "system", "content": "List every number in the documents and what it refers to, "
                                                "one short line each. No other text."},
                  {"role": "user", "content": context}]
            facts = llm.chat(m1, usage, max_tokens=400)
            msgs = [{"role": "system", "content": BASE},
                    {"role": "user", "content": f"{context}\nNumbers found:\n{facts}\n\nQuestion: {question}"}]
            return llm.chat_json(msgs, Answer, usage, retries=0)
    except LLMError:
        return None
    raise ValueError(strategy)
