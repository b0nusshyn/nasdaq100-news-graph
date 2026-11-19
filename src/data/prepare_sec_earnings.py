"""Prepare an earnings diagnostic input from the Quant500 SEC 8-K CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.core.io import read_table

# Both share classes share one issuer release, keyed by the source's stable SEC CIK.
SHARE_CLASSES = {"1652044": ("GOOGL", "GOOG"), "1754301": ("FOXA", "FOX")}


def prepare_events(source: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Keep regular Item 2.02 filings with a reported date inside a membership spell."""
    rows = source.loc[
        source.form.eq("8-K") & source.rule.eq("2.02")
        & source.date_source.eq("reportDate") & source.date_uncertain.eq("no"),
        ["cik", "ticker", "announcement_date", "sec_url"],
    ].rename(columns={"announcement_date": "date", "sec_url": "source_url"})
    rows["date"] = pd.to_datetime(rows.date, format="%Y-%m-%d")
    rows = rows.loc[rows.date.between("2019-01-01", "2023-12-31")]
    rows["ticker"] = [
        SHARE_CLASSES.get(str(cik).lstrip("0"), (ticker,))
        for cik, ticker in zip(rows.cik, rows.ticker, strict=True)
    ]
    rows = rows.explode("ticker").merge(membership, on="ticker")
    eligible = rows.date.ge(rows.start) & (rows.end.isna() | rows.date.le(rows.end))
    return (
        rows.loc[eligible, ["date", "ticker", "source_url"]]
        .sort_values(["date", "ticker", "source_url"])
        .drop_duplicates(["date", "ticker"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="downloaded Quant500 CSV")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = pd.read_csv(args.source, comment="#", dtype=str)
    events = prepare_events(source, read_table("membership"))
    if events.empty:
        raise ValueError("No SEC earnings events matched historical membership")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output, index=False)
    print(f"{len(events)} ticker-date events across {events.ticker.nunique()} tickers")


if __name__ == "__main__":
    main()
