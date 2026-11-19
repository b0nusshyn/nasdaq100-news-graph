import pandas as pd
import pytest

from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.data.coverage import earnings_coverage, news_coverage
from src.data.prepare_sec_earnings import prepare_events


def inputs():
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    articles = read_table("article_tickers", FIXTURES_DIR / "article_tickers.parquet")
    return prices, articles


def test_news_denominator_uses_in_index_observed_days():
    prices, articles = inputs()
    report = news_coverage(prices, articles)
    assert report.in_index_days.sum() == prices.in_index.sum()
    assert report.coverage.between(0, 1).all()
    fff = report.loc[report.ticker.eq("FFF")].iloc[0]
    assert fff.in_index_days == 30
    empty = news_coverage(prices, articles.head(0))
    assert empty.coverage.eq(0).all()


def test_earnings_repeat_company_and_no_news_gap():
    prices, articles = inputs()
    events = pd.DataFrame({
        "date": ["2021-01-04", "2021-01-05", "2021-01-04"],
        "ticker": ["AAA", "AAA", "FFF"],
        "source_url": ["https://www.investing.com/equities/example-earnings"] * 3,
    })
    report = earnings_coverage(events, articles.head(0), prices)
    assert report.coverage_gap.tolist() == [True, True, False]
    assert report.observed_in_index.tolist() == [True, True, False]
    assert report.window_end.iloc[0] == pd.Timestamp("2021-01-05")
    bounded = earnings_coverage(events, articles.head(0), prices, end=pd.Timestamp("2021-01-05"))
    assert bounded.window_within_range.tolist() == [True, False, True]
    assert bounded.coverage_gap.tolist() == [True, False, False]
    duplicate = pd.concat([events, events.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        earnings_coverage(duplicate, articles, prices)
    with pytest.raises(ValueError, match="Investing.com"):
        earnings_coverage(events.assign(source_url="https://example.com"), articles, prices)


def test_sec_earnings_respects_historical_share_class_membership():
    url = "https://www.sec.gov/Archives/edgar/data/1652044/example.htm"
    source = pd.DataFrame({
        "cik": ["0001652044"] * 4,
        "ticker": ["GOOGL"] * 4,
        "announcement_date": ["2021-01-04", "2021-01-04", "2021-01-05", "2021-01-06"],
        "sec_url": [url] * 4,
        "date_source": ["reportDate"] * 3 + ["edgarEntry"],
        "date_uncertain": ["no"] * 4,
        "form": ["8-K", "8-K", "8-K/A", "8-K"],
        "rule": ["2.02"] * 4,
    })
    membership = pd.DataFrame({
        "ticker": ["GOOG", "GOOGL"],
        "start": pd.to_datetime(["2021-01-05", "2021-01-04"]),
        "end": pd.to_datetime([None, "2021-01-04"]),
    })
    events = prepare_events(source, membership)
    assert events.ticker.tolist() == ["GOOGL"]
    assert events.date.tolist() == [pd.Timestamp("2021-01-04")]
    prices, articles = inputs()
    assert len(earnings_coverage(events, articles, prices)) == 1
