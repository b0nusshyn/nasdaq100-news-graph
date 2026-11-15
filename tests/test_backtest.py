import numpy as np
import pandas as pd
import pytest

from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.eval.backtest import BacktestResult, backtest


@pytest.fixture
def panel():
    dates = pd.bdate_range("2021-01-04", periods=16).as_unit("ns")
    labels = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"]).to_frame(index=False)
    labels["fwd_ret_1d"] = 0.0
    labels["fwd_ret_5d"] = 0.0
    predictions = labels[["date", "ticker"]].copy()
    predictions["model"] = "synthetic"
    predictions["horizon"] = 1
    predictions["fold"] = 0
    predictions["p_up"] = np.tile([0.9, 0.1], len(dates))
    predictions["y_true"] = 0
    return predictions, labels, dates


def test_known_metrics_and_per_side_costs(panel):
    predictions, labels, dates = panel
    predictions = predictions[predictions["date"].isin(dates[:3])]
    for date, value in zip(dates[:3], [0.02, -0.04, 0.06], strict=True):
        labels.loc[(labels["date"] == date) & (labels["ticker"] == "AAA"), "fwd_ret_1d"] = value
    result = backtest(predictions, labels, 1)
    net = np.array([0.009, -0.021, 0.029])
    wealth = np.r_[1, np.cumprod(1 + net)]
    assert isinstance(result, BacktestResult)
    np.testing.assert_allclose(result.equity, wealth - 1)
    assert result.equity.index.equals(dates[:4].rename("date"))
    assert result.cumulative_return == pytest.approx(wealth[-1] - 1)
    assert result.sharpe == pytest.approx(net.mean() / net.std(ddof=1) * np.sqrt(252))
    assert result.max_drawdown == pytest.approx(0.021)
    assert result.turnover == 6
    free = backtest(predictions, labels, 1, cost_bps_per_side=0)
    assert free.cumulative_return > result.cumulative_return
    pd.testing.assert_series_equal(backtest(predictions, labels, 1, 5).equity, result.equity)


def test_five_day_periods_do_not_overlap_even_across_folds(panel):
    predictions, labels, dates = panel
    predictions["horizon"] = 5
    predictions.loc[predictions["date"] >= dates[3], "fold"] = 1
    # Poison intervening returns: counting daily 5-day returns must fail this test.
    labels["fwd_ret_5d"] = 100.0
    for date, value in zip(dates[[0, 5, 10]], [0.02, -0.04, 0.06], strict=True):
        labels.loc[labels["date"] == date, "fwd_ret_5d"] = 0.0
        labels.loc[(labels["date"] == date) & (labels["ticker"] == "AAA"), "fwd_ret_5d"] = value
    result = backtest(predictions, labels, 5)
    net = np.array([0.009, -0.021, 0.029])
    np.testing.assert_allclose(result.equity, np.r_[0, np.cumprod(1 + net) - 1])
    assert result.equity.index.equals(dates[[0, 5, 10, 15]].rename("date"))
    assert result.sharpe == pytest.approx(net.mean() / net.std(ddof=1) * np.sqrt(252 / 5))
    assert result.turnover == 6


def test_ranking_weights_and_ties_are_independent_of_row_order(panel):
    _, _, dates = panel
    labels = pd.DataFrame({
        "date": [dates[0]] * 5 + [dates[1]],
        "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "AAA"],
        "fwd_ret_1d": [0.08, -0.02, 10, 10, 10, np.nan],
    })
    labels["date"] = labels["date"].astype("datetime64[ns]")
    predictions = labels.iloc[:5][["date", "ticker"]].assign(
        model="synthetic", horizon=1, p_up=[0.9, 0.1, 0.5, 0.5, 0.5],
    )
    result = backtest(predictions, labels, 1)
    assert result.cumulative_return == pytest.approx(0.049)
    pd.testing.assert_series_equal(
        backtest(predictions.sample(frac=1, random_state=2), labels, 1).equity, result.equity,
    )
    predictions["p_up"] = [0.9, 0.1, 0.9, 0.5, 0.5]
    # The top boundary tie picks CCC, consistently by ticker order.
    assert backtest(predictions, labels, 1).cumulative_return == pytest.approx(5.009)


def test_cash_and_initial_drawdown(panel):
    predictions, labels, dates = panel
    predictions = predictions[predictions["date"].isin(dates[:2])].copy()
    result = backtest(predictions, labels, 1)
    assert result.max_drawdown == pytest.approx(1 - 0.999**2)
    assert result.turnover == 4
    predictions["p_up"] = 0.5
    cash = backtest(predictions, labels, 1)
    assert cash.cumulative_return == cash.max_drawdown == cash.turnover == 0
    assert np.isnan(cash.sharpe)
    assert backtest(predictions[predictions["ticker"] == "AAA"], labels, 1).turnover == 0


def test_missing_signals_leave_cash_and_calendar_controls_holding_period(panel):
    predictions, labels, dates = panel
    predictions["horizon"] = 5
    predictions = predictions[predictions["date"].isin(dates[[0, 6, 10]])]
    result = backtest(predictions, labels, 5)
    np.testing.assert_allclose(result.equity, [0, -0.001, -0.001, 0.999**2 - 1])
    assert result.turnover == 4


def test_missing_selected_return_does_not_change_selection(panel):
    predictions, labels, dates = panel
    predictions = predictions[predictions["date"] == dates[0]]
    labels.loc[0, "fwd_ret_1d"] = np.nan
    with pytest.raises(ValueError, match="selected trades"):
        backtest(predictions, labels, 1)
    with pytest.raises(ValueError, match="selected trades"):
        backtest(predictions, labels.iloc[1:], 1)


def test_empty_and_incomplete_periods(panel):
    predictions, labels, dates = panel
    empty = backtest(predictions.iloc[:0], labels, 1)
    assert empty.equity.empty
    assert empty.cumulative_return == empty.turnover == empty.max_drawdown == 0
    assert np.isnan(empty.sharpe)
    incomplete = backtest(predictions[predictions["date"] == dates[-1]], labels, 1)
    assert incomplete.equity.tolist() == [0]
    assert incomplete.turnover == 0


@pytest.mark.parametrize("horizon", [1, 5])
def test_committed_synthetic_labels(horizon):
    labels = read_table("labels", FIXTURES_DIR / "labels.parquet")
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    members = prices.loc[prices["in_index"], ["date", "ticker"]]
    predictions = members.loc[members["date"] <= labels["date"].max() - pd.Timedelta(days=10)].copy()
    predictions["model"] = "synthetic"
    predictions["horizon"] = horizon
    predictions["p_up"] = predictions["ticker"].map({
        "AAA": 0.9, "BBB": 0.8, "CCC": 0.6, "DDD": 0.4, "EEE": 0.2, "FFF": 0.1,
    })

    # Verify the fixture returns against future prices for each horizon.
    sorted_prices = prices.sort_values(["ticker", "date"])
    future_close = sorted_prices.groupby("ticker")["close"].shift(-horizon)
    expected_returns = sorted_prices[["date", "ticker"]].copy()
    expected_returns["forward_return"] = future_close / sorted_prices["close"] - 1
    checked_labels = labels.merge(expected_returns, on=["date", "ticker"], validate="one_to_one")
    np.testing.assert_allclose(
        checked_labels[f"fwd_ret_{horizon}d"], checked_labels["forward_return"], equal_nan=True,
    )

    result = backtest(predictions, labels, horizon)

    assert result.turnover > 0
    assert np.isfinite(result.cumulative_return)
    assert result.equity.index.is_monotonic_increasing


@pytest.mark.parametrize(
    "kwargs", [dict(horizon=2), dict(horizon=True), dict(horizon=1, quantile=0.6),
               dict(horizon=1, quantile=0), dict(horizon=1, cost_bps_per_side=-1),
               dict(horizon=1, cost_bps_per_side=np.inf), dict(horizon=1, cost_bps_per_side=np.nan),
               dict(horizon=1, cost_bps=10, cost_bps_per_side=3)],
)
def test_invalid_parameters(panel, kwargs):
    predictions, labels, _ = panel
    with pytest.raises(ValueError):
        backtest(predictions, labels, **kwargs)


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_probabilities(panel, probability):
    predictions, labels, _ = panel
    predictions.loc[0, "p_up"] = probability
    with pytest.raises(ValueError, match="p_up"):
        backtest(predictions, labels, 1)


def test_duplicate_keys_and_mixed_models(panel):
    predictions, labels, _ = panel
    with pytest.raises(ValueError, match="duplicate"):
        backtest(pd.concat([predictions, predictions]), labels, 1)
    with pytest.raises(ValueError, match="duplicate"):
        backtest(predictions, pd.concat([labels, labels]), 1)
    predictions.loc[0, "model"] = "another"
    with pytest.raises(ValueError, match="one model"):
        backtest(predictions, labels, 1)


def test_horizon_filter_and_truth_are_not_trading_signals(panel):
    predictions, labels, dates = panel
    predictions = predictions[predictions["date"] == dates[0]].copy()
    expected = backtest(predictions, labels, 1)
    other = predictions.assign(horizon=5, p_up=1 - predictions["p_up"])
    predictions["y_true"] = 1
    actual = backtest(pd.concat([predictions, other]), labels, 1)
    pd.testing.assert_series_equal(actual.equity, expected.equity)


def test_equal_weights_with_multiple_stocks_per_side(panel):
    _, _, dates = panel
    labels = pd.DataFrame({
        "date": pd.Series([dates[0]] * 4 + [dates[1]], dtype="datetime64[ns]"),
        "ticker": ["AAA", "BBB", "CCC", "DDD", "AAA"],
        "fwd_ret_1d": [0.08, 0.04, -0.02, -0.06, np.nan],
    })
    predictions = labels.iloc[:4][["date", "ticker"]].assign(
        model="synthetic", horizon=1, p_up=[0.9, 0.8, 0.2, 0.1],
    )
    result = backtest(predictions, labels, 1, quantile=0.5)
    assert result.cumulative_return == pytest.approx(0.049)
    assert result.turnover == 2


def test_missing_unselected_returns_do_not_affect_portfolio(panel):
    predictions, labels, dates = panel
    predictions = predictions[predictions["date"] == dates[0]]
    extra = predictions.iloc[:1].assign(ticker="CCC", p_up=0.5)
    extra_label = labels.iloc[:1].assign(ticker="CCC", fwd_ret_1d=np.nan)
    expected = backtest(predictions, labels, 1)
    actual = backtest(pd.concat([predictions, extra]), pd.concat([labels, extra_label]), 1)
    pd.testing.assert_series_equal(actual.equity, expected.equity)


def test_invalid_dates_and_missing_calendar_date(panel):
    predictions, labels, dates = panel
    with pytest.raises(ValueError, match="trading calendar"):
        backtest(predictions, labels[labels["date"] != dates[0]], 1)
    predictions.loc[0, "date"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="midnight"):
        backtest(predictions, labels, 1)


def test_bankruptcy_is_reported(panel):
    predictions, labels, _ = panel
    labels.loc[labels["ticker"] == "BBB", "fwd_ret_1d"] = 3.0
    with pytest.raises(ValueError, match="lost all capital"):
        backtest(predictions, labels, 1)


@pytest.mark.parametrize("horizon", [1, 5])
def test_forward_return_columns_take_priority(panel, horizon):
    predictions, labels, dates = panel
    predictions = predictions[predictions["date"] == dates[0]].assign(horizon=horizon)
    # Conflicting legacy values must not override the requested forward returns.
    labels[f"ret_{horizon}d"] = 100.0
    result = backtest(predictions, labels, horizon)
    assert result.cumulative_return == pytest.approx(-0.001)

    labels.loc[0, f"fwd_ret_{horizon}d"] = np.nan
    with pytest.raises(ValueError, match="selected trades"):
        backtest(predictions, labels, horizon)


@pytest.mark.parametrize("horizon", [1, 5])
def test_missing_forward_return_column_is_rejected(panel, horizon):
    predictions, labels, _ = panel
    predictions["horizon"] = horizon
    labels = labels.rename(columns={f"fwd_ret_{horizon}d": f"ret_{horizon}d"})
    with pytest.raises(ValueError, match=f"fwd_ret_{horizon}d"):
        backtest(predictions, labels, horizon)
