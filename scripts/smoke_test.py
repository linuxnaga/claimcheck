"""Day-1 check: can we reach the chat endpoint and the embeddings endpoint?"""
import _boot  # noqa
from claimcheck import config
from claimcheck.llm import llm

missing = [n for n in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "EMBED_MODEL") if not getattr(config, n)]
if missing:
    raise SystemExit(f"Missing settings: {missing}. Add them in Colab Secrets / .env and retry.")
llm.use_cache = False
text = llm.chat([{"role": "user", "content": 'Reply with ONLY this JSON: {"msg": "hello"}'}], json_mode=True)
print("chat reply:", text)
print("JSON mode supported:", llm.use_response_format)
print("embeddings provider:", config.EMBED_PROVIDER)
vec = llm.embed(["hello world"])
print("embedding size:", vec.shape[1])
print("ALL GOOD. Write 'JSON mode supported' and the embedding size in your notes.")
