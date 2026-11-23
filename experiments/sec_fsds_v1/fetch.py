"""Network (run from a terminal that exports SEC_USER_AGENT): fetch FSDS quarterly ZIPs.

One sequential request per quarter (22 in total for 2018q3-2023q4), at least 2 s apart,
resumable: a quarter whose filtered extract already exists is skipped. Each ZIP is streamed to
`zips/<q>.zip.part`, renamed on completion, filtered by `extract.py`, and (unless --keep-zips)
deleted afterwards; the SHA-256 and byte size of the original ZIP stay in `extract/<q>.json`
and `manifest.jsonl` so anyone can re-download and verify it.

Stops at once on HTTP 403/429 (or any non-retryable error); retries 5xx with backoff unless
the server sends Retry-After. Refuses to run if it would leave < 2 GiB free.

    .venv/bin/python -B -m experiments.sec_fsds_v1.fetch --fetch
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener

from experiments.sec_acquisition_v1.submissions import NoRedirect, identifying_agent
from experiments.sec_fsds_v1.extract import extract

HERE = Path(__file__).resolve().parent
URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"
RESERVE = 2 * 1024**3
MAX_ZIP = 400 * 1024**2


def quarters(first: str = "2018q3", last: str = "2023q4") -> list[str]:
    y, q = int(first[:4]), int(first[-1])
    out = []
    while f"{y}q{q}" <= last:
        out.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def _free(path: Path) -> int:
    return shutil.disk_usage(path).free


def download(q: str, zips: Path, agent: str) -> Path:
    target = zips / f"{q}.zip"
    if target.exists():
        return target
    part = zips / f"{q}.zip.part"
    opener = build_opener(NoRedirect)
    for delay in (2, 6, 18):
        time.sleep(delay)
        try:
            with opener.open(Request(URL.format(q=q), headers={"User-Agent": agent}), timeout=120) as r:
                size = int(r.headers.get("Content-Length") or 0)
                if size > MAX_ZIP or _free(zips) - size < RESERVE:
                    raise RuntimeError(f"{q}: {size} bytes would breach the 2 GiB reserve or size cap")
                with part.open("wb") as out:
                    copied = 0
                    while block := r.read(1 << 20):
                        copied += len(block)
                        if copied > MAX_ZIP:
                            raise RuntimeError(f"{q}: response exceeded {MAX_ZIP} bytes")
                        out.write(block)
            if size and copied != size:
                raise RuntimeError(f"{q}: truncated ({copied} of {size} bytes)")
            part.rename(target)
            return target
        except HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code not in (500, 502, 503, 504) or exc.headers.get("Retry-After") or delay == 18:
                raise
    raise AssertionError("unreachable")


def run(out: Path, qs: list[str], agent: str | None, keep_zips: bool) -> list[dict]:
    extract_dir, zips = out / "extract", out / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    done = []
    for q in qs:
        if (extract_dir / f"{q}.json").exists():
            done.append(json.loads((extract_dir / f"{q}.json").read_text()))
            continue
        if agent is None and not (zips / f"{q}.zip").exists():
            print(f"{q}: not cached; rerun with --fetch from a terminal exporting SEC_USER_AGENT")
            continue
        if _free(out) < RESERVE + 64 * 1024**2:
            raise RuntimeError("Stopping: free disk below 2 GiB reserve")
        path = download(q, zips, agent) if agent else zips / f"{q}.zip"
        record = extract(path, extract_dir)
        record["fetched_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with (out / "manifest.jsonl").open("a") as m:
            m.write(json.dumps(record) + "\n")
        if not keep_zips:
            path.unlink()
        print(f"{q}: {record['sub_rows']} submissions, {record['num_rows']} values, "
              f"zip sha256 {record['zip_sha256'][:16]}")
        done.append(record)
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE)
    p.add_argument("--first", default="2018q3")
    p.add_argument("--last", default="2023q4")
    p.add_argument("--fetch", action="store_true", help="allow network (needs SEC_USER_AGENT)")
    p.add_argument("--keep-zips", action="store_true")
    args = p.parse_args()
    agent = identifying_agent(os.getenv("SEC_USER_AGENT", "")) if args.fetch else None
    done = run(args.out, quarters(args.first, args.last), agent, args.keep_zips)
    print(f"{len(done)} of {len(quarters(args.first, args.last))} quarters extracted")


if __name__ == "__main__":
    main()
