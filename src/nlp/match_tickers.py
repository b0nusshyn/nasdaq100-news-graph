"""Map each article to the tickers it is about. Owner: NLP. Issues #16, #17.

Output: `article_tickers.parquet` (SCHEMA.md), already aligned to trading days.
Modelling builds co-mention edges from it (#21).
"""

from __future__ import annotations

import pandas as pd

from src.core.todo import todo


def build_aliases(membership: pd.DataFrame) -> dict[str, list[str]]:
    """Ticker -> name variants to search for (symbol, company name, short name, old names)."""
    todo(16, "ticker aliases")


def match_articles(articles: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Return (article_id, published_at, ticker), one row per article-ticker match."""
    todo(16, "match articles to tickers")


def assign_trading_day(published_at: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Convert to America/New_York; after 16:00 ET, or on a non-trading day, roll to the next trading day."""
    todo(17, "timestamp alignment")
