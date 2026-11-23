"""Offline: filter one SEC Financial Statement Data Sets quarterly ZIP to the rows this study uses.

FSDS (https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets) holds
the XBRL financial-statement values of every submission *as filed* under its own accession,
with the EDGAR filing date and acceptance timestamp. A later amendment or restatement is a
separate accession; values are never revised in place. That is what makes it usable for a
point-in-time study, provided availability is taken from `filed`/`accepted`, never `period`.

Kept from each ZIP:
- sub: every periodic/annual original or amended report (FORMS), all columns needed for
  identity (instance prefix, name, former names) and timing (filed, accepted, period).
- num: consolidated (no coreg, no segments) values of TAGS for those submissions, standard
  taxonomy only (version starts with us-gaap/ or ifrs), quarter (qtrs 1) or annual (qtrs 4).

The output Parquet pair is written next to a JSON record of the source ZIP's SHA-256, so the
filtered rows can be re-derived from the official file and checked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A",
         "10-KT", "10-QT"}
TAGS = {
    # revenue, in precedence order (features.REVENUE)
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
    "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense", "Revenue",
    # operating income
    "OperatingIncomeLoss", "ProfitLossFromOperatingActivities",
    # net income (parent, then incl. NCI)
    "NetIncomeLoss", "ProfitLossAttributableToOwnersOfParent", "ProfitLoss",
    # diluted EPS
    "EarningsPerShareDiluted", "DilutedEarningsLossPerShare",
    "EarningsPerShareBasicAndDiluted",
}
SUB_COLS = ["adsh", "cik", "name", "sic", "countryinc", "former", "changed", "fye", "form",
            "period", "fy", "fp", "filed", "accepted", "prevrpt", "instance", "nciks", "aciks"]
NUM_COLS = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "value"]
ADSH = re.compile(r"\d{10}-\d{2}-\d{6}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(archive: ZipFile, member: str):
    with archive.open(member) as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="strict"),
                                  delimiter="\t", quoting=csv.QUOTE_NONE)


def filter_zip(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        if not {"sub.txt", "num.txt"} <= names:
            raise ValueError(f"{path.name}: missing sub.txt/num.txt")
        subs = []
        for row in _rows(archive, "sub.txt"):
            if row["form"] in FORMS:
                if not ADSH.fullmatch(row["adsh"]):
                    raise ValueError(f"{path.name}: bad accession {row['adsh']!r}")
                subs.append({c: row.get(c, "") for c in SUB_COLS})
        keep = {s["adsh"] for s in subs}
        nums = []
        for row in _rows(archive, "num.txt"):
            if (row["adsh"] in keep and row["tag"] in TAGS and row["qtrs"] in ("1", "4")
                    and not row.get("coreg") and not row.get("segments")
                    and row["version"].startswith(("us-gaap/", "ifrs")) and row["value"] != ""):
                nums.append({c: row[c] for c in NUM_COLS})
    sub = pd.DataFrame(subs, columns=SUB_COLS)
    sub["cik"] = sub.cik.str.zfill(10)
    num = pd.DataFrame(nums, columns=NUM_COLS)
    num["qtrs"] = num.qtrs.astype(int)
    num["value"] = num.value.astype(float)
    return sub, num


def extract(path: Path, out_dir: Path) -> dict:
    """Write <quarter>.sub.parquet / .num.parquet / .json under out_dir; return the record."""
    quarter = path.stem  # e.g. 2022q3
    if not re.fullmatch(r"\d{4}q[1-4]", quarter):
        raise ValueError(f"Unexpected FSDS archive name {path.name}")
    sub, num = filter_zip(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out_dir / f"{quarter}.sub.parquet", index=False)
    num.to_parquet(out_dir / f"{quarter}.num.parquet", index=False)
    record = {"quarter": quarter, "zip_sha256": sha256(path), "zip_bytes": path.stat().st_size,
              "url": f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{path.name}",
              "sub_rows": len(sub), "num_rows": len(num),
              "sub_sha256": sha256(out_dir / f"{quarter}.sub.parquet"),
              "num_sha256": sha256(out_dir / f"{quarter}.num.parquet")}
    (out_dir / f"{quarter}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zips", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "extract")
    args = p.parse_args()
    for z in args.zips:
        print(json.dumps(extract(z, args.out)))


if __name__ == "__main__":
    main()
