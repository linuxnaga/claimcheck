"""Tracing. Uses Langfuse when LANGFUSE_* keys exist, always writes a local JSONL log too."""
import json
import re
import time

from . import config


def openai_client_class():
    """Return Langfuse's drop-in OpenAI client when tracing keys exist, else the normal one."""
    if config.get("LANGFUSE_PUBLIC_KEY") and config.get("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.openai import OpenAI  # traces every model call automatically
            return OpenAI
        except Exception:
            pass
    from openai import OpenAI
    return OpenAI


def observe(name=None):
    """Decorator: Langfuse span when available, otherwise does nothing."""
    def wrap(fn):
        if config.get("LANGFUSE_PUBLIC_KEY") and config.get("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import observe as lf_observe
                return lf_observe(name=name or fn.__name__)(fn)
            except Exception:
                return fn
        return fn
    return wrap


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s\-]{8,}\d")


def _member_names():
    from .corpus_gen import MEMBERS
    return [m["name"] for m in MEMBERS]


def mask_pii(text: str) -> str:
    """Mask emails, phone numbers and the names of the fake members."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    for name in _member_names():
        text = text.replace(name, "[NAME]")
    return text


def log_event(kind: str, payload: dict):
    """Append one line to data/results/trace.jsonl (PII masked)."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"t": time.time(), "kind": kind, "data": payload}, default=str)
    with open(config.RESULTS_DIR / "trace.jsonl", "a", encoding="utf-8") as f:
        f.write(mask_pii(line) + "\n")


def flush():
    """Send pending Langfuse traces before a short script exits."""
    if not (config.get("LANGFUSE_PUBLIC_KEY") and config.get("LANGFUSE_SECRET_KEY")):
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass
