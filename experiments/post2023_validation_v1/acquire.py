"""Fetch just the frozen-cohort daily files from a pinned public archive."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from src.core.io import read_table

ROOT = Path(__file__).parent
REV = "2f8763bb9ed0c09fca6e871c8d660bdd8e4ce0ea"
BASE = f"https://huggingface.co/datasets/AmirTrader/YahooFinance/resolve/{REV}/"
MAX_BYTES = 2_000_000


def cohort():
    m = read_table("membership")
    d = pd.Timestamp("2023-12-29")
    return sorted(m.loc[m.start.le(d) & (m.end.isna() | m.end.ge(d)), "ticker"].unique())


def fetch(item):
    url = BASE + item
    dest = ROOT / "raw" / Path(item).name
    try:
        if dest.exists():
            data = dest.read_bytes()
        else:
            req = Request(url, headers={"User-Agent": "NASDAQ-100 research audit"})
            with urlopen(req, timeout=30) as f:
                data = f.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError("File exceeds per-file cap")
            if dest.suffix == ".parquet" and not (data[:4] == b"PAR1" and data[-4:] == b"PAR1"):
                raise ValueError("Response is not Parquet")
            with dest.open("xb") as out:
                out.write(data)
        return {"path": item, "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "retrieval_record_utc": datetime.now(timezone.utc).isoformat()}
    except (ValueError, OSError) as exc:
        return {"path": item, "url": url, "error": str(exc)}


def main():
    root = ROOT / "raw"
    release = json.loads((root / "hf_release.json").read_text())
    if release["sha"] != REV:
        raise ValueError("Unexpected archive revision")
    listing = {r["rfilename"] for r in release["siblings"]}
    fixed = cohort()
    requested = {t: f"data/daily/{t}.parquet" for t in fixed}
    files = ["README.md", *[p for p in requested.values() if p in listing]]
    assert MAX_BYTES * len(files) < 500_000_000
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, files))
    result = {"revision": REV, "cohort_asof": "2023-12-29", "cohort": fixed,
              "archive_missing_symbols": [t for t, p in requested.items() if p not in listing],
              "files": records, "total_bytes": sum(r.get("bytes", 0) for r in records)}
    with (ROOT / "acquisition.json").open("x") as out:
        json.dump(result, out, indent=2)
    print(len(fixed), result["archive_missing_symbols"], result["total_bytes"],
          sum("error" in r for r in records), flush=True)


if __name__ == "__main__":
    main()
