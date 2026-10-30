"""Central settings. Reads Colab Secrets first, then environment variables, then the .env file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path=ROOT / ".env"):
    """Copy KEY=value lines from .env into the environment, without overriding values already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


_load_dotenv()


def get(name, default=None):
    try:
        from google.colab import userdata  # exists only on Colab
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)


DATA = Path(get("CC_DATA_DIR", str(ROOT / "data")))
POLICY_DIR = DATA / "policies"
INDEX_DIR = DATA / "index"
CACHE_DIR = DATA / "cache"
RESULTS_DIR = DATA / "results"
EVAL_PATH = DATA / "eval" / "eval_set.jsonl"
DB_PATH = DATA / "claimcheck.db"

LLM_BASE_URL = get("LLM_BASE_URL")
LLM_API_KEY = get("LLM_API_KEY")
LLM_MODEL = get("LLM_MODEL")
EMBED_MODEL = get("EMBED_MODEL")
# embeddings may live on a different server; empty means "same as the chat endpoint"
EMBED_BASE_URL = get("EMBED_BASE_URL") or LLM_BASE_URL
EMBED_API_KEY = get("EMBED_API_KEY") or LLM_API_KEY
# "endpoint" (default) calls the embeddings API; "local" runs EMBED_MODEL on this machine (needs mentor approval)
EMBED_PROVIDER = (get("EMBED_PROVIDER") or "endpoint").strip().lower()
RERANK_MODEL = get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

MAX_STEPS = int(get("MAX_STEPS", 10))            # template cap: 10 steps per query
TOKEN_BUDGET = int(get("TOKEN_BUDGET", 12000))   # hard cap of tokens per query
AS_OF_DATE = get("AS_OF_DATE", "2025-09-15")     # fixed "today" so results repeat
PRICE_PER_1M_TOKENS = float(get("PRICE_PER_1M_TOKENS", 0.20))  # assumption, edit it
USE_LLM_CRITIC = get("USE_LLM_CRITIC", "1") == "1"
CHUNKING = get("CHUNKING", "sections")
RATE_LIMIT_PER_MIN = int(get("RATE_LIMIT_PER_MIN", 10))
USE_CACHE = get("USE_CACHE", "1") == "1"   # set USE_CACHE=0 for the official 3 evaluation runs


def ensure_dirs():
    for d in (POLICY_DIR, INDEX_DIR, CACHE_DIR, RESULTS_DIR, EVAL_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
