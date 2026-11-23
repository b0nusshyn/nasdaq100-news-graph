"""Cache targeted SEC submissions and emit unreviewed accession candidates."""
import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

BASE = "https://data.sec.gov/submissions/"
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"}
PATTERN = re.compile(r"CIK\d{10}(?:-submissions-\d{3,})?\.json")


def identifying_agent(value):
    name, _, address = value.strip().rpartition(" ")
    if (not name or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}", address)
            or re.search(r"placeholder|your.?email|example\.|@localhost|sample company|<|>|[\r\n]",
                         value, re.I) or address.lower().endswith((".invalid", ".test", ".example"))):
        raise ValueError("Set SEC_USER_AGENT to the real project name and contact email")
    return value.strip()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def load(cache, name, agent=None):
    if not PATTERN.fullmatch(name):
        raise ValueError("Unexpected SEC submissions filename")
    target = cache / name
    if target.exists():
        return json.loads(target.read_bytes())
    if agent is None:
        raise FileNotFoundError(target)
    agent = identifying_agent(agent)
    if shutil.disk_usage(cache.parent).free < 2 * 1024**3 + 16 * 1024**2:
        raise RuntimeError("Insufficient disk above 2 GiB reserve")
    opener = build_opener(NoRedirect)
    for delay in (1, 3, 9):
        time.sleep(delay)
        try:
            with opener.open(Request(BASE + name, headers={"User-Agent": agent}), timeout=30) as response:
                payload = response.read(8 * 1024**2 + 1)
            if len(payload) > 8 * 1024**2:
                raise ValueError("Submissions response exceeded 8 MiB")
            result = json.loads(payload)
            if not isinstance(result, dict):
                raise ValueError("Unexpected submissions schema")
            cache.mkdir(exist_ok=True)
            with target.open("xb") as output:
                output.write(payload)
            return result
        except HTTPError as exc:
            if exc.code not in (500, 502, 503, 504) or exc.headers.get("Retry-After") or delay == 9:
                raise
    raise AssertionError("Unreachable")


def read_candidates(cache, cik, agent=None, probe=False):
    if not re.fullmatch(r"\d{10}", cik):
        raise ValueError("CIK must be a ten-digit reviewed issuer identifier")
    root = f"CIK{cik}.json"
    submission = load(cache, root, agent)
    if str(submission["cik"]).zfill(10) != cik:
        raise ValueError("Issuer mismatch")
    if probe:
        return
    filing = submission["filings"]
    sources = [(root, filing["recent"])]
    for page in filing["files"]:
        if page["filingFrom"] <= "2023-12-31" and page["filingTo"] >= "2019-01-01":
            if not page["name"].startswith(f"CIK{cik}-submissions-"):
                raise ValueError("History page issuer mismatch")
            sources.append((page["name"], load(cache, page["name"], agent)))
    seen = set()
    for name, data in sources:
        names = ("accessionNumber", "form", "filingDate", "reportDate", "acceptanceDateTime")
        columns = [data[key] for key in names]
        digest = hashlib.sha256((cache / name).read_bytes()).hexdigest()
        for accession, form, filed, period, accepted in zip(*columns, strict=True):
            if form not in FORMS or not "2019-01-01" <= filed <= "2023-12-31":
                continue
            if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
                raise ValueError("Invalid accession")
            if accession in seen:
                continue
            seen.add(accession)
            yield {"cik": cik, "accession": accession, "form": form, "filed": filed,
                   "period": period, "accepted_raw": accepted, "source": BASE + name,
                   "sha256": digest, "status": "requires_original_accession_review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="Local directory with an existing parent")
    parser.add_argument("cik")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Load and validate only root document")
    args = parser.parse_args()
    agent = identifying_agent(os.getenv("SEC_USER_AGENT", "")) if args.fetch else None
    for row in read_candidates(args.cache, args.cik, agent, args.probe):
        print(json.dumps(row))
    if args.probe:
        print("Root submissions record validated.")


if __name__ == "__main__":
    main()
