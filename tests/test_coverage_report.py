import pandas as pd
import pytest

from src.nlp.coverage_report import UNKNOWN_SECTOR, coverage_tables


def news():
    d = pd.to_datetime(["2019-01-02", "2019-01-02", "2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"])
    return pd.DataFrame({
        "date": d,
        "ticker": ["AAA", "BBB", "AAA", "BBB", "AAA", "CCC"],
        "has_news": [True, False, True, False, False, True],
        "n_articles": [2, 0, 1, 0, 0, 4],
    })


SECTORS = pd.DataFrame({"ticker": ["AAA", "BBB"], "gics_sector": ["Tech", "Health"]})


def test_by_year_shares_and_counts():
    y = coverage_tables(news(), SECTORS)["by_year"].set_index("year")
    assert y.loc[2019, "ticker_days"] == 2 and y.loc[2019, "share_with_news"] == pytest.approx(0.5)
    assert y.loc[2020, "share_with_news"] == pytest.approx(0.5)
    # Mean articles counts only covered days: 2020 -> (1 + 4) / 2.
    assert y.loc[2020, "articles_per_covered_day"] == pytest.approx(2.5)
    assert y.loc[2020, "tickers_without_news"] == 1  # BBB


def test_by_sector_and_unknown_sector():
    s = coverage_tables(news(), SECTORS)["by_sector"].set_index("sector")
    assert s.loc["Tech", "share_with_news"] == pytest.approx(2 / 3)
    assert s.loc["Health", "with_news"] == 0
    assert s.loc[UNKNOWN_SECTOR, "tickers"] == 1  # CCC has no GICS row


def test_grid_has_all_row_and_column():
    g = coverage_tables(news(), SECTORS)["year_sector"].set_index("sector")
    assert list(g.columns) == ["2019", "2020", "All"]
    assert g.loc["All", "All"] == pytest.approx(news()["has_news"].mean())
    assert g.loc["All", "2019"] == pytest.approx(0.5)
    assert g.index[-1] == "All"
