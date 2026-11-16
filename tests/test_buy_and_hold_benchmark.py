import numpy as np
import pandas as pd
import pytest

from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.eval.buy_and_hold import buy_and_hold_benchmark


@pytest.fixture
def prices():
    dates = pd.bdate_range("2021-01-04", periods=3).as_unit("ns")
    frame = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"]).to_frame(index=False)
    frame["close"] = [100.0, 100.0, 200.0, 100.0, 100.0, 100.0]
    frame["in_index"] = True
    return frame, dates


def test_weights_drift_instead_of_daily_rebalancing(prices):
    frame, dates = prices
    result = buy_and_hold_benchmark(frame, dates, cost_bps_per_side=0)
    np.testing.assert_allclose(result.equity, [0, 0.5, 0])
    assert result.cumulative_return == 0
    assert result.max_drawdown == pytest.approx(1 / 3)
    assert result.turnover == 2
    assert np.isfinite(result.sharpe)


def test_entry_exit_fees_and_initial_drawdown(prices):
    frame, dates = prices
    frame["close"] = 100.0
    result = buy_and_hold_benchmark(frame, dates, cost_bps_per_side=5)
    invested = 1 / 1.0005
    expected_wealth = invested * 0.9995
    np.testing.assert_allclose(result.equity, [invested - 1, invested - 1, expected_wealth - 1])
    assert result.max_drawdown == pytest.approx(1 - expected_wealth)
    assert result.turnover == pytest.approx(2 * invested)


def test_membership_exit_keeps_cash_and_does_not_buy_new_entrants(prices):
    frame, dates = prices
    frame.loc[(frame["ticker"] == "AAA") & (frame["date"] >= dates[1]), "in_index"] = False
    entrant = frame.loc[frame["ticker"] == "BBB"].copy()
    entrant["ticker"] = "NEW"
    entrant["in_index"] = [False, True, True]
    entrant["close"] = [1.0, 100.0, 1000.0]
    result = buy_and_hold_benchmark(pd.concat([frame, entrant]), dates, cost_bps_per_side=0)
    np.testing.assert_allclose(result.equity, [0, 0.5, 0.5])
    # No price is needed after a holding has been sold.
    frame = frame.loc[~((frame["ticker"] == "AAA") & (frame["date"] == dates[2]))]
    same = buy_and_hold_benchmark(frame, dates, cost_bps_per_side=0)
    pd.testing.assert_series_equal(result.equity, same.equity)


@pytest.mark.parametrize("problem", ["missing_row", "missing_date", "nan", "zero", "membership"])
def test_missing_holding_data_raises(prices, problem):
    frame, dates = prices
    if problem == "missing_row":
        frame = frame.drop(index=2)
    elif problem == "missing_date":
        frame = frame.loc[frame["date"] != dates[1]]
    elif problem == "membership":
        frame["in_index"] = frame["in_index"].astype("boolean")
        frame.loc[2, "in_index"] = None
    else:
        frame.loc[2, "close"] = np.nan if problem == "nan" else 0.0
    with pytest.raises(ValueError, match="require prices and membership"):
        buy_and_hold_benchmark(frame, dates)


def test_each_fold_starts_fresh_and_future_rows_do_not_affect_results(prices):
    frame, dates = prices
    original = frame.copy(deep=True)
    expected = buy_and_hold_benchmark(frame, dates[:2], cost_bps_per_side=0)
    frame.loc[frame["date"] == dates[2], "close"] = np.nan
    actual = buy_and_hold_benchmark(frame, dates[:2], cost_bps_per_side=0)
    pd.testing.assert_series_equal(actual.equity, expected.equity)
    fresh = buy_and_hold_benchmark(original, dates[1:], cost_bps_per_side=0)
    assert fresh.cumulative_return == pytest.approx(-0.25)


def test_committed_synthetic_prices():
    frame = read_table("prices", FIXTURES_DIR / "prices.parquet")
    original = frame.copy(deep=True)
    dates = pd.DatetimeIndex(frame["date"].unique()).sort_values()
    result = buy_and_hold_benchmark(frame, dates)
    assert np.isfinite(result.cumulative_return)
    assert result.equity.index.equals(dates.rename("date"))
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("cost", [-1, np.nan, np.inf, 10_000])
def test_invalid_costs(prices, cost):
    frame, dates = prices
    with pytest.raises(ValueError, match="cost_bps_per_side"):
        buy_and_hold_benchmark(frame, dates, cost_bps_per_side=cost)


def test_invalid_calendar_and_no_initial_members(prices):
    frame, dates = prices
    with pytest.raises(ValueError, match="sorted"):
        buy_and_hold_benchmark(frame, dates[::-1])
    with pytest.raises(ValueError, match="two trading"):
        buy_and_hold_benchmark(frame, dates[:1])
    with pytest.raises(ValueError, match="no eligible"):
        buy_and_hold_benchmark(frame.assign(in_index=False), dates)


@pytest.mark.parametrize("ticker,still_member", [("ALXN", False), ("MXIM", False), ("XLNX", True)])
def test_acquisition_or_halt_without_exit_quote_requires_policy(prices, ticker, still_member):
    frame, dates = prices
    frame["ticker"] = frame["ticker"].replace({"AAA": ticker})
    affected_rows = frame["ticker"].eq(ticker) & frame["date"].ge(dates[1])
    frame.loc[affected_rows, "in_index"] = still_member
    missing_quote = frame["ticker"].eq(ticker) & frame["date"].eq(dates[1])
    frame = frame.loc[~missing_quote].copy()
    original = frame.copy(deep=True)

    with pytest.raises(ValueError, match=f"{ticker}.*no zero-return filling or assumed liquidation proceeds"):
        buy_and_hold_benchmark(frame, dates)

    pd.testing.assert_frame_equal(frame, original)
