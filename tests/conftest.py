import os, sys, tempfile
from pathlib import Path

tmp = tempfile.mkdtemp(prefix="cc_test_")
os.environ["CC_DATA_DIR"] = tmp
os.environ["USE_LLM_CRITIC"] = "0"
os.environ["CD_PACING"] = "0"     # fake models need no pacing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from claimcheck import corpus_gen, evalset
from claimdesk.gen import build_claims


@pytest.fixture(scope="session", autouse=True)
def built():
    corpus_gen.build_all()
    corpus_gen.write_members()
    evalset.build_eval()
    build_claims()
