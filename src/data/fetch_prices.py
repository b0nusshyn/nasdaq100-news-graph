"""Download daily OHLCV for every ticker that was ever a member, and flag index membership.

Output: `prices.parquet` (SCHEMA.md). Owner: Data. Issues #7, #14.

    python -m src.data.fetch_prices --start 2018-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.core.todo import todo


def download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Adjusted OHLCV from yfinance in long format (date, ticker, open, high, low, close, volume)."""
    todo(7, "download prices")


def add_in_index(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Set `in_index` from the membership intervals (survivorship-bias guard)."""
    todo(14, "add in_index flag")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2023-12-31")
    p.parse_args()
    todo(7, "fetch_prices")


if __name__ == "__main__":
    main()
