"""Download FNSPID and keep NASDAQ-100 ever-members, 2019–2023. Owner: NLP. Issue #10.

Output: `<data>/raw/fnspid_filtered.csv` (raw, not a SCHEMA.md artifact), plus
`<data>/raw/fnspid_filter_summary.json` with the source revision, bounds and row counts.

Source: `Stock_news/nasdaq_exteral_data.csv` (23 GB, sic) from the FNSPID HuggingFace dataset,
pinned to an immutable revision. Columns: a row index, `Date` (UTC, e.g.
`2023-12-16 23:00:00 UTC`), `Article_title`, `Stock_symbol`, `Url`, `Publisher`, `Author`,
`Article` (full text) and four summaries. The smaller `All_external.csv` (5.7 GB) is not used:
its news stops in 2020 and it has no rows for MSFT, COST, PYPL and other members.

Filtering:
- **Tickers**: `Stock_symbol` goes through `canonical_ticker`, then must belong to a ticker
  with a membership spell overlapping the window (survivorship: never today's list). This is a
  universe filter only; `in_index` per day is applied downstream.
- **Dates**: kept by US/Eastern calendar date in `[start - PAD_DAYS, end]`. The pad keeps
  after-close and weekend articles just before `start` so the 16:00 ET cutoff (task 2.4) can
  roll them onto the first trading days of the window; 2.4 drops anything landing before it.

Timestamps stay in UTC here; conversion to ET and trading-day assignment is task 2.4.

    python -m src.nlp.filter_fnspid
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.core.io import read_table
from src.core.paths import raw_dir
from src.core.tickers import canonical_ticker

HF_DATASET = "Zihan1004/FNSPID"
# Same immutable revision Data used for the archived prices (src/data/fetch_archive.py).
REVISION = "bf9189c41527198897d1af3e17b1a0095279fc45"
SOURCE_FILE = "Stock_news/nasdaq_exteral_data.csv"
PAD_DAYS = 7
CHUNK_ROWS = 500_000
ET = "America/New_York"

# Columns kept in the output. Summaries and Author are dropped; `ticker` is added.
KEEP = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Article"]
OUT_COLUMNS = ["Date", "ticker", "Stock_symbol", "Article_title", "Url", "Publisher", "Article"]


def source_url(revision: str = REVISION) -> str:
    return f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{revision}/{SOURCE_FILE}"


def universe(membership: pd.DataFrame, start: str, end: str) -> set[str]:
    """Tickers with at least one membership spell overlapping [start, end]."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    still_member = membership["end"].isna()
    overlap = (membership["start"] <= e) & (still_member | (membership["end"] >= s))
    return set(membership.loc[overlap, "ticker"].astype(str))


def filter_chunk(df: pd.DataFrame, tickers: set[str], start: str, end: str) -> pd.DataFrame:
    """Keep rows whose canonical ticker is in `tickers` and whose ET date is in the padded window."""
    df = df.dropna(subset=["Date", "Stock_symbol"])
    sym = df["Stock_symbol"].astype(str)
    canon = sym.map(_canonical_cached)
    df = df.loc[canon.isin(tickers)].assign(ticker=canon[canon.isin(tickers)])
    if df.empty:
        return df.reindex(columns=OUT_COLUMNS)
    ts = pd.to_datetime(df["Date"].str.replace(" UTC", "", regex=False), utc=True, errors="coerce")
    et_day = ts.dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    lo = pd.Timestamp(start) - pd.Timedelta(days=PAD_DAYS)
    keep = ts.notna() & (et_day >= lo) & (et_day <= pd.Timestamp(end))
    return df.loc[keep, OUT_COLUMNS]


_CANON: dict[str, str] = {}


def _canonical_cached(symbol: str) -> str:
    if symbol not in _CANON:
        _CANON[symbol] = canonical_ticker(symbol)
    return _CANON[symbol]


def filter_csv(chunks: Iterable[pd.DataFrame], tickers: set[str], start: str, end: str,
               output: Path) -> dict:
    """Stream chunks through `filter_chunk` into `output`; return row counts."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    n_in = n_out = 0
    per_ticker: dict[str, int] = {}
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        header = True
        for chunk in chunks:
            n_in += len(chunk)
            kept = filter_chunk(chunk, tickers, start, end)
            n_out += len(kept)
            for t, n in kept["ticker"].value_counts().items():
                per_ticker[t] = per_ticker.get(t, 0) + int(n)
            kept.to_csv(f, index=False, header=header)
            header = False
    tmp.replace(output)
    return {"rows_in": n_in, "rows_out": n_out,
            "tickers_with_news": len(per_ticker),
            "tickers_without_news": sorted(tickers - set(per_ticker)),
            "rows_per_ticker": dict(sorted(per_ticker.items()))}


def download(dest: Path, revision: str = REVISION, chunk_bytes: int = 8 << 20) -> Path:
    """Download the source CSV to `dest`, resuming a partial `.part` file if present."""
    import requests  # light dependency, but keep --help import-free

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    url = source_url(revision)
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # already complete
            part.replace(dest)
            return dest
        r.raise_for_status()
        if done and r.status_code != 206:
            done = 0  # server ignored Range; start over
        total = done + int(r.headers.get("Content-Length", 0))
        with open(part, "ab" if done else "wb") as f:
            for i, block in enumerate(r.iter_content(chunk_bytes)):
                f.write(block)
                done += len(block)
                if i % 64 == 0:
                    print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
    part.replace(dest)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=None, help="default: <data>/raw/fnspid_filtered.csv")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2023-12-31")
    p.add_argument("--source", type=Path, default=None,
                   help="local copy of the source CSV; default: <data>/raw/fnspid/nasdaq_exteral_data.csv, "
                        "downloaded if missing")
    p.add_argument("--revision", default=REVISION, help="HuggingFace dataset revision (40-char sha)")
    args = p.parse_args()

    output = args.output or raw_dir() / "fnspid_filtered.csv"
    source = args.source or raw_dir() / "fnspid" / Path(SOURCE_FILE).name
    if not source.exists():
        print(f"downloading {source_url(args.revision)} -> {source}")
        download(source, args.revision)

    tickers = universe(read_table("membership"), args.start, args.end)
    print(f"{len(tickers)} ever-member tickers in {args.start}..{args.end}")
    chunks = pd.read_csv(source, usecols=KEEP, dtype=str, chunksize=CHUNK_ROWS)
    stats = filter_csv(chunks, tickers, args.start, args.end, output)

    summary = {"source": source_url(args.revision), "revision": args.revision,
               "start": args.start, "end": args.end, "pad_days": PAD_DAYS,
               "date_rule": "US/Eastern calendar date of the UTC timestamp", **stats}
    (output.parent / "fnspid_filter_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"kept {stats['rows_out']:,} of {stats['rows_in']:,} rows -> {output}")
    print(f"{stats['tickers_with_news']} tickers with news; none for: {stats['tickers_without_news']}")


if __name__ == "__main__":
    main()
