"""Coverage diagnostics for the NLP hand-off (Data, preparatory work for issue #27).

Consumes matched article_tickers and an optional sourced earnings CSV. No network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.core.io import read_table
from src.core.paths import raw_dir
from src.data.build_index_history import trading_days
from src.data.validation import canonical_table, checked_prices


def news_coverage(prices: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker/year; denominators include only observed in-index days.

    This measures matched-news coverage, not raw FNSPID completeness. Missing
    price rows must be resolved using the quality report before interpreting it.
    """
    prices = checked_prices(prices)
    articles = canonical_table(articles, "article_tickers")
    counts = articles.groupby(["date", "ticker"]).article_id.nunique().rename("n_articles")
    rows = prices.loc[prices.in_index, ["date", "ticker"]].join(counts, on=["date", "ticker"])
    rows["has_news"] = rows.n_articles.fillna(0).gt(0)
    rows["year"] = rows.date.dt.year
    report = rows.groupby(["ticker", "year"]).agg(
        in_index_days=("date", "size"), covered_days=("has_news", "sum"),
        matched_articles=("n_articles", "sum"),
    ).reset_index()
    report["uncovered_days"] = report.in_index_days - report.covered_days
    report["coverage"] = report.covered_days / report.in_index_days
    return report


def earnings_coverage(
    earnings: pd.DataFrame, articles: pd.DataFrame, prices: pd.DataFrame,
    *, end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Review window: earnings calendar day through the following NASDAQ session.

    Required reviewed CSV columns: date (ISO local calendar day), ticker,
    source_url (Investing.com or SEC filing provenance). Announcement times are unknown, so
    this is a coverage diagnostic, never an available-at-time-t model feature.
    """
    from urllib.parse import urlparse

    earnings = earnings.copy()
    if not {"date", "ticker", "source_url"} <= set(earnings):
        raise ValueError("Earnings CSV needs date, ticker, source_url")
    if earnings.empty:
        raise ValueError("Earnings CSV is empty")
    earnings["date"] = pd.to_datetime(earnings.date).astype("datetime64[ns]")
    # Reuse a frozen key/date validator without creating an earnings Parquet contract.
    normalized = canonical_table(
        earnings[["date", "ticker"]].assign(article_id=[str(i) for i in range(len(earnings))]),
        "article_tickers",
    )
    earnings["ticker"] = normalized.ticker.to_numpy()
    if earnings.date.isna().any() or earnings.duplicated(["date", "ticker"]).any():
        raise ValueError("Null or duplicate earnings dates/tickers")
    for url in earnings.source_url:
        source = urlparse(str(url))
        host = source.hostname or ""
        investing = host == "investing.com" or host.endswith(".investing.com")
        sec_filing = host in {"www.sec.gov", "sec.gov"} and source.path.startswith("/Archives/edgar/data/")
        if source.scheme != "https" or not (investing or sec_filing):
            raise ValueError("Earnings source_url must identify an Investing.com page or SEC filing")
    articles = canonical_table(articles, "article_tickers")
    prices = checked_prices(prices)
    sessions = trading_days(earnings.date.min(), earnings.date.max() + pd.Timedelta(days=30))
    in_index_prices = prices.loc[prices.in_index].groupby("ticker")
    matched_articles = articles.groupby("ticker")
    rows = []
    for event in earnings.itertuples(index=False):
        next_session = sessions[sessions > event.date][0]
        window_within_range = end is None or next_session <= end
        ticker_prices = (in_index_prices.get_group(event.ticker)
                         if event.ticker in in_index_prices.groups else prices.head(0))
        ticker_articles = (matched_articles.get_group(event.ticker)
                           if event.ticker in matched_articles.groups else articles.head(0))
        observed = ticker_prices.loc[ticker_prices.date.between(event.date, next_session)]
        matched = ticker_articles.loc[ticker_articles.date.isin(observed.date)]
        rows.append({"date": event.date, "ticker": event.ticker, "source_url": event.source_url,
                     "window_end": next_session, "observed_in_index": not observed.empty,
                     "window_within_range": window_within_range,
                     "n_articles": matched.article_id.nunique(),
                     "coverage_gap": bool(window_within_range and not observed.empty and matched.empty)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2023-12-31", help="inclusive final date")
    p.add_argument("--articles", type=Path, help="matched article_tickers Parquet from NLP")
    p.add_argument("--earnings", type=Path, help="sourced earnings CSV: date,ticker,source_url")
    args = p.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start > end:
        raise ValueError("start must not exceed end")
    prices = read_table("prices")
    prices = prices.loc[prices.date.between(start, end)]
    articles = read_table("article_tickers", args.articles)
    folder = raw_dir()
    folder.mkdir(parents=True, exist_ok=True)
    report = news_coverage(prices, articles)
    report.to_csv(folder / "news_coverage.csv", index=False)
    print(f"{folder / 'news_coverage.csv'}: {len(report)} ticker-years (matched-news coverage)")
    if args.earnings:
        earnings = pd.read_csv(args.earnings)
        earnings["date"] = pd.to_datetime(earnings.date)
        earnings = earnings.loc[earnings.date.between(start, end)]
        events = earnings_coverage(earnings, articles, prices, end=end)
        events.to_csv(folder / "earnings_coverage.csv", index=False)
        print(f"{folder / 'earnings_coverage.csv'}: {int(events.coverage_gap.sum())} candidate gaps")


if __name__ == "__main__":
    main()
