"""Reconstruct historical NASDAQ-100 membership from Wikipedia's changes table.

Output: `membership.parquet` (SCHEMA.md). Owner: Data. Issue #6.

    python -m src.data.build_index_history
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.core.todo import todo

WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def scrape_changes() -> pd.DataFrame:
    """Return the changes table: one row per (date, added ticker, removed ticker)."""
    todo(6, "scrape Wikipedia NASDAQ-100 changes")


def scrape_current() -> pd.DataFrame:
    """Return today's constituents with GICS sector/sub-industry (also used for sector edges, #8)."""
    todo(6, "scrape current constituents")


def membership_intervals(changes: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Walk the changes backwards from today's list into (ticker, start, end) intervals."""
    todo(6, "build membership intervals")


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    todo(6, "build_index_history")


if __name__ == "__main__":
    main()
