"""Index audited CIKs in a local SEC quarterly ZIP without extracting it."""
import argparse
import csv
import io
import re
from pathlib import Path
from zipfile import ZipFile

from experiments.sec_lagged_proxy.build import HERE, _sha256

FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "8-K", "8-K/A", "20-F", "20-F/A",
         "6-K", "6-K/A", "40-F", "40-F/A"}


def index_submissions(archive, ciks):
    with ZipFile(archive) as source, source.open("sub.txt") as member:
        for row in csv.DictReader(io.TextIOWrapper(member, encoding="utf-8-sig"), delimiter="\t"):
            cik = row["cik"]
            if not cik.isdecimal() or cik.zfill(10) not in ciks or row["form"] not in FORMS:
                continue
            accession = row["adsh"]
            if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
                raise ValueError(f"Invalid accession in {archive.name}")
            yield {"cik": cik.zfill(10), "accession": accession, "form": row["form"],
                   "filed_raw": row["filed"], "accepted_raw": row["accepted"],
                   "period_raw": row["period"], "registrant_name": row["name"],
                   "archive": archive.name, "review_state": "verify_original_accession",
                   "index_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                                f"{accession.replace('-', '')}/{accession}-index.htm"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with (HERE / "identities.csv").open(newline="") as stream:
        ciks = {row["cik"] for row in csv.DictReader(stream)}
    print(f"Archive SHA-256 {_sha256(args.archive)}")
    for submission in index_submissions(args.archive, ciks):
        print(submission)


if __name__ == "__main__":
    main()
