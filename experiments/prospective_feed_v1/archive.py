"""Convert hash-verified archive files into a panel and audit availability without issuing scores."""
import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.prospective_cohort_v1.live import calendar, digest, encode, load_freeze, timestamp, write_once

ROOT = Path(__file__).parent
PRIOR = ROOT.parent / "post2023_validation_v1"
FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]


def convert(data, ticker, expected_hash):
    if digest(data) != expected_hash:
        raise ValueError(f"archive hash mismatch for {ticker}")
    f = pd.read_parquet(io.BytesIO(data), columns=["date", *FIELDS])
    f.date = pd.to_datetime(f.date, format="%Y-%m-%d")
    if f.date.isna().any() or f.date.duplicated().any():
        raise ValueError(f"invalid or duplicate dates for {ticker}")
    return f.assign(ticker=ticker)


def summarize(frame, dates):
    f = frame.set_index("date").reindex(dates)
    finite = np.isfinite(f[FIELDS].to_numpy(dtype=float)).all(axis=1)
    valid = (finite & f[FIELDS].gt(0).all(axis=1) & f.high.ge(f[["open", "close"]].max(axis=1))
             & f.low.le(f[["open", "close"]].min(axis=1)))
    return {"latest": frame.date.max().date().isoformat(), "valid": int(valid.sum()),
            "missing": [d.date().isoformat() for d in dates.difference(frame.date)],
            "present_invalid": [d.date().isoformat() for d in dates[~valid & dates.isin(frame.date)]]}


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--day", required=True)
    ns = args.parse_args()
    if shutil.disk_usage(ROOT).free < 2.1 * 1024**3:
        raise OSError("disk safety margin")
    cfg, freeze_hash = load_freeze()
    meta_raw = (ROOT / "archive_metadata.json").read_bytes()
    head = json.loads(meta_raw)
    m = json.loads((PRIOR / "acquisition.json").read_bytes())
    if head["sha"] != m["revision"]:
        raise ValueError("remote revision differs from cache; obtain the matching immutable export first")
    entries = {Path(x["path"]).stem: x for x in m["files"] if x["path"].endswith(".parquet")}
    sched, idx = calendar(ns.day)
    dates = sched.index[idx+1-cfg["warmup_sessions"]:idx+1]
    release = timestamp(head["lastModified"])
    records, panels = {}, []
    for ticker in cfg["cohort"]:
        e = entries[ticker]
        t = convert((PRIOR / "raw" / f"{ticker}.parquet").read_bytes(), ticker, e["sha256"])
        records[ticker] = summarize(t, dates) | {"raw_sha256": e["sha256"]}
        panels.append(t[t.date.isin(dates)])
    close, cutoff = sched.market_close.iloc[idx], sched.market_open.iloc[idx+1]
    report = dict(signal_date=ns.day, expected=len(dates)*len(records), release=release.isoformat(),
                  signal_close=close.isoformat(), cutoff=cutoff.isoformat(),
                  release_in_window=close <= release < cutoff,
                  quote_rows_present=sum(len(p) for p in panels), valid_rows=sum(r["valid"] for r in records.values()),
                  readiness="BLOCKED", approved_source=False, freeze_sha256=freeze_hash, revision=head["sha"],
                  metadata_sha256=digest(meta_raw), symbols=records)
    print(write_once(ROOT / f"audit_{ns.day}.json", encode(report)))


if __name__ == "__main__":
    main()
