from claimcheck.ratelimit import RateLimiter
from claimcheck.llm import extract_json
from claimcheck.tracing import mask_pii


def test_rate_limiter():
    r = RateLimiter(2)
    assert r.allow("a", now=0) and r.allow("a", now=1) and not r.allow("a", now=2)
    assert r.allow("a", now=62) and r.allow("b", now=2)


def test_extract_json_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}


def test_mask_pii():
    s = mask_pii("mail me at a.b@x.com or +91 98765 43210")
    assert "x.com" not in s and "98765" not in s


def test_local_embeddings_switch(monkeypatch):
    """EMBED_PROVIDER=local uses the on-machine model and never calls the embeddings API."""
    import numpy as np
    from claimcheck import config
    from claimcheck.llm import LLM

    class FakeModel:
        def encode(self, texts, batch_size=64, show_progress_bar=False):
            return [[3.0, 4.0]] * len(texts)

    m = LLM()
    m._local_embedder = FakeModel()
    monkeypatch.setattr(config, "EMBED_PROVIDER", "local")
    out = m.embed(["a", "b"])
    assert out.shape == (2, 2) and np.allclose(out[0], [0.6, 0.8])   # normalised to length 1


def test_extract_json_ignores_trailing_text_and_second_object():
    assert extract_json('{"a": 1} {"b": 2}') == {"a": 1}
    assert extract_json('{"a": {"x": 1}} Note: done.') == {"a": {"x": 1}}


class _Err(Exception):
    def __init__(self, status):
        self.status_code = status


class _FakeClient:
    """Fails with the given HTTP statuses, then answers."""
    def __init__(self, statuses):
        self.statuses, self.calls = list(statuses), []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.statuses:
            raise _Err(self.statuses.pop(0))
        class R:
            class U:
                prompt_tokens, completion_tokens = 10, 5
            usage = U()
            choices = [type("C", (), {"message": type("M", (), {"content": '{"ok": true}'})()})()]
        return R()


def test_rate_limit_retries_and_keeps_json_mode(monkeypatch):
    import claimcheck.llm as L
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    m = L.LLM()
    m._client = _FakeClient([429, 429])
    text, p, c = m.chat_raw([{"role": "user", "content": "x"}], json_mode=True)
    assert text == '{"ok": true}' and m.use_response_format is True
    assert all("response_format" in kw for kw in m._client.calls)


def test_bad_request_turns_json_mode_off_and_auth_error_stops(monkeypatch):
    import pytest
    import claimcheck.llm as L
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    m = L.LLM()
    m._client = _FakeClient([400])
    m.chat_raw([{"role": "user", "content": "x"}], json_mode=True)
    assert m.use_response_format is False
    m._client = _FakeClient([401, 401, 401])
    with pytest.raises(L.EndpointError):
        m.chat_raw([{"role": "user", "content": "x"}])
    assert len(m._client.calls) == 1
