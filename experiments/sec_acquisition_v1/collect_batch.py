"""Validate or collect a sequential batch scoped to a reviewed issuer ledger."""
import argparse
import csv
import os
from pathlib import Path
from urllib.error import HTTPError

from experiments.sec_acquisition_v1.submissions import identifying_agent, read_candidates


def reviewed_ciks(ledger):
    with ledger.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(not row["identity_basis"].startswith("https://www.sec.gov/Archives/")
           for row in rows):
        raise ValueError("Every issuer needs dated SEC filing evidence")
    return list(dict.fromkeys(row["cik"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("reviewed_issuers", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    try:
        agent = identifying_agent(os.getenv("SEC_USER_AGENT", "")) if args.fetch else None
        for cik in reviewed_ciks(args.reviewed_issuers):
            candidates = list(read_candidates(args.cache, cik, agent))
            print(f"{cik}: {len(candidates)} validated periodic candidates", flush=True)
    except HTTPError as error:
        raise SystemExit(f"Stopped batch on SEC HTTP {error.code}. Existing cache retained.") from None


if __name__ == "__main__":
    main()
