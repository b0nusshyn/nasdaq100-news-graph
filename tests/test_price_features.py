import numpy as np
import pandas as pd

from src.models.price_features import compute_extended_price


def sample_prices():
    days = pd.bdate_range("2021-01-04", periods=25)
    prices = pd.DataFrame(
        [(date, ticker, 100 * np.exp(rate * step), ticker != "C")
         for step, date in enumerate(days) for ticker, rate in (("A", 0.01), ("B", 0.03), ("C", 0.10))],
        columns=["date", "ticker", "close", "in_index"],
    )
    return days, prices


def test_extended_returns_and_member_market_benchmark():
    days, prices = sample_prices()
    rows = prices.date.eq(days[-1]) & prices.ticker.eq("A")
    features = compute_extended_price(prices)
    np.testing.assert_allclose(features.loc[rows].iloc[0], [0.05, 0.20, 0, -0.05, -0.20], atol=1e-12)


def test_future_prices_and_membership_do_not_change_past_features():
    days, prices = sample_prices()
    before = prices.date <= days[20]
    changed = prices.copy()
    changed.loc[~before, "close"] *= 100
    changed.loc[~before, "in_index"] = False
    pd.testing.assert_frame_equal(compute_extended_price(prices).loc[before],
                                  compute_extended_price(changed).loc[before])


def test_missing_quote_is_not_bridged_as_a_one_session_return():
    days, prices = sample_prices()
    prices = prices.loc[~(prices.date.eq(days[10]) & prices.ticker.eq("A"))]
    features = compute_extended_price(prices)
    key = prices.date.eq(days[11]) & prices.ticker.eq("A")
    assert features.loc[key].isna().all(axis=None)
