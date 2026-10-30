"""Build the ClaimDesk data: members, hospitals, codes, past claims and the 150 test claims with gold labels.
Run once after build_data.py:  python scripts/build_claims.py"""
import _boot  # noqa: F401

from claimdesk.gen import build_claims

if __name__ == "__main__":
    counts = build_claims()
    print(f"{sum(counts.values())} test claims:", dict(counts))
