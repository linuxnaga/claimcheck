"""The ONLY file that talks to the model endpoint.

Features: on-disk cache, token counting, token budget, JSON validation with retries.
"""
import hashlib
import json
import re
import time

import numpy as np
from pydantic import BaseModel, ValidationError

from . import config


class LLMError(Exception):
    pass


class EndpointError(LLMError):
    """The endpoint could not be reached or kept failing (not a JSON problem)."""


class BudgetExceeded(Exception):
    pass


class Usage:
    """Counts tokens and calls for ONE user query."""

    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.calls = 0
        self.json_retries = 0

    @property
    def total(self):
        return self.prompt + self.completion

    def add(self, p, c):
        self.prompt += p
        self.completion += c
        self.calls += 1


def extract_json(text: str) -> dict:
    """Pull the first complete JSON object out of a model reply (handles ```json fences and trailing text)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start = t.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(t[start:])
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return obj


def retry_wait(error, attempt):
    """Seconds to wait before retrying: the server's Retry-After if given, else 2, 4, 8, 16, 30."""
    try:
        after = float(error.response.headers.get("retry-after"))
        return min(max(after, 1.0), 60.0)
    except Exception:
        return min(2.0 * 2 ** attempt, 30.0)


class LLM:
    def __init__(self):
        self._client = None
        self._embed_client = None
        self._local_embedder = None
        self.use_response_format = True
        self.use_cache = config.USE_CACHE
        self.rate_limit_hits = 0     # HTTP 429 replies seen (ClaimDesk's pacer reads this)

    @property
    def client(self):
        if self._client is None:
            from .tracing import openai_client_class
            cls = openai_client_class()
            self._client = cls(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        return self._client

    @property
    def embed_client(self):
        if self._embed_client is None:
            if (config.EMBED_BASE_URL, config.EMBED_API_KEY) == (config.LLM_BASE_URL, config.LLM_API_KEY):
                self._embed_client = self.client
            else:
                from openai import OpenAI
                self._embed_client = OpenAI(api_key=config.EMBED_API_KEY, base_url=config.EMBED_BASE_URL)
        return self._embed_client

    # ---- raw call (tests replace this) -------------------------------------
    def chat_raw(self, messages, temperature=0.0, max_tokens=600, json_mode=False):
        kwargs = dict(model=config.LLM_MODEL, messages=messages,
                      temperature=temperature, max_tokens=max_tokens)
        if json_mode and self.use_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        last = None
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(**kwargs)
                text = r.choices[0].message.content or ""
                u = r.usage
                return text, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)
            except Exception as e:  # network, rate limit, unsupported option
                last = e
                status = getattr(e, "status_code", None)
                if status == 400 and "response_format" in kwargs:
                    # the endpoint rejected JSON mode: switch it off and retry at once
                    kwargs.pop("response_format")
                    self.use_response_format = False
                    continue
                if status == 429:
                    self.rate_limit_hits += 1
                if status is not None and status < 500 and status != 429:
                    break                                   # 401, 403, 404...: retrying will not help
                time.sleep(retry_wait(e, attempt))          # rate limit (429), timeout or server error
        raise EndpointError(f"endpoint call failed: {last}")

    # ---- cached call --------------------------------------------------------
    def cache_path(self, messages, temperature=0.0, max_tokens=600):
        key = hashlib.sha256(json.dumps(
            [config.LLM_MODEL, messages, temperature, max_tokens], sort_keys=True).encode()).hexdigest()
        return config.CACHE_DIR / f"{key}.json"

    def is_cached(self, messages, temperature=0.0, max_tokens=600):
        return self.use_cache and temperature == 0 and self.cache_path(messages, temperature, max_tokens).exists()

    def chat(self, messages, usage=None, temperature=0.0, max_tokens=600, json_mode=False):
        if usage is not None and usage.total > config.TOKEN_BUDGET:
            raise BudgetExceeded(f"token budget {config.TOKEN_BUDGET} exceeded")
        path = self.cache_path(messages, temperature, max_tokens)
        if self.use_cache and temperature == 0 and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return d["text"]
        text, p, c = self.chat_raw(messages, temperature, max_tokens, json_mode)
        if usage is not None:
            usage.add(p, c)
        if self.use_cache and temperature == 0:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text, "p": p, "c": c}), encoding="utf-8")
        return text

    def chat_json(self, messages, model_cls, usage=None, retries=2, temperature=0.0, max_tokens=600):
        """Ask for JSON, validate it with Pydantic, retry with the error message."""
        msgs = list(messages)
        last_err = None
        for attempt in range(retries + 1):
            text = self.chat(msgs, usage, temperature, max_tokens, json_mode=True)
            try:
                return model_cls.model_validate(extract_json(text))
            except (ValueError, ValidationError) as e:
                last_err = str(e)[:300]
                if usage is not None:
                    usage.json_retries += 1
                msgs = msgs + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"Your reply was invalid: {last_err}. "
                                                 "Reply again with ONLY one valid JSON object."}]
        raise LLMError(f"invalid JSON after {retries + 1} tries: {last_err}")

    # ---- embeddings ---------------------------------------------------------
    @property
    def local_embedder(self):
        """Fallback: run the embedding model on this machine (EMBED_PROVIDER=local). Uses no token quota."""
        if self._local_embedder is None:
            from sentence_transformers import SentenceTransformer
            self._local_embedder = SentenceTransformer(config.EMBED_MODEL)
        return self._local_embedder

    def embed(self, texts, batch=64):
        if config.EMBED_PROVIDER == "local":
            out = self.local_embedder.encode(list(texts), batch_size=batch, show_progress_bar=len(texts) > 500)
        else:
            out = []
            for i in range(0, len(texts), batch):
                part = texts[i:i + batch]
                r = self.embed_client.embeddings.create(model=config.EMBED_MODEL, input=part)
                out.extend([d.embedding for d in sorted(r.data, key=lambda d: d.index)])
        arr = np.array(out, dtype="float32")
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        return arr


llm = LLM()
