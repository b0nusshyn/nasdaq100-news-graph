"""Fetch FINRA Reg SHO daily short-sale volume (consolidated NMS file), keep our tickers only.

Source: https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt (free, public).
Columns: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market. FINRA posts each day's
file after that day's close, so a value for day t may be used only from day t+1 on (the
feature builder lags it by one session).

Sequential, >= 0.5 s between requests, resumable (one small Parquet per day under days/),
stops on 403/429 or any non-retryable error, SHA-256 of every original file in manifest.jsonl.

    .venv/bin/python -B -m experiments.finra_short_v1.fetch
"""
import hashlib
import io
import json
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"


def symbols() -> set[str]:
    from src.core.tickers import RENAMES
    t = set(pd.read_parquet(REPO / "data/prices.parquet", columns=["ticker"]).ticker.astype(str))
    return t | {old for old, new in RENAMES.items() if new in t}


def fetch_day(day: pd.Timestamp, keep: set[str]) -> pd.DataFrame | None:
    for delay in (0.5, 3, 9):
        time.sleep(delay)
        try:
            with urlopen(Request(URL.format(d=day.strftime("%Y%m%d")),
                                 headers={"User-Agent": "nasdaq100-news-graph academic research"}),
                         timeout=60) as r:
                raw = r.read(64 * 1024**2)
            break
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in (500, 502, 503, 504) or delay == 9:
                raise
    df = pd.read_csv(io.BytesIO(raw), sep="|", dtype={"Date": str, "Symbol": str})
    df = df.loc[df.Symbol.isin(keep) & df.Date.str.fullmatch(r"\d{8}")]
    df = df.assign(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return df


def main():
    import pandas_market_calendars as mcal
    days = mcal.get_calendar("NASDAQ").schedule("2018-10-01", "2023-12-29").index
    out = HERE / "days"
    out.mkdir(parents=True, exist_ok=True)
    keep = symbols()
    done = 0
    with (HERE / "manifest.jsonl").open("a") as manifest:
        for day in days:
            target = out / f"{day:%Y%m%d}.parquet"
            if target.exists():
                done += 1
                continue
            if shutil.disk_usage(HERE).free < 2.1 * 1024**3:
                raise OSError("Disk below 2 GiB margin")
            df = fetch_day(day, keep)
            if df is None:
                manifest.write(json.dumps({"date": f"{day:%Y-%m-%d}", "status": 404}) + "\n")
                continue
            df.to_parquet(target, index=False)
            manifest.write(json.dumps({"date": f"{day:%Y-%m-%d}", "rows": len(df),
                                       "sha256": df.sha256.iloc[0] if len(df) else None}) + "\n")
            manifest.flush()
            done += 1
            if done % 100 == 0:
                print(f"{day:%Y-%m-%d}: {done} days", flush=True)
    print(f"{done} of {len(days)} days cached")


if __name__ == "__main__":
    main()
