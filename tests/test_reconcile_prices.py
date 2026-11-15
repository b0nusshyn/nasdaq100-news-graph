import numpy as np
import pandas as pd
import pytest

from src.data.reconcile_prices import reconcile_quote_history


def daily_and_quotes():
    dates = pd.date_range("2021-01-01", periods=25)
    close = np.arange(100., 125.)
    daily = pd.DataFrame({"date": dates, "ticker": "AAA", "open": close - 1, "high": close + 2,
                          "low": close - 2, "close": close, "volume": 200., "source": "alpaca",
                          "adj_factor": 0.5})
    for field in ["open", "high", "low", "close"]:
        daily[f'a{field}'] = daily[field] * daily.adj_factor
    quote = daily[["date", "open", "high", "low", "close"]].copy()
    quote["volume"] = 100.
    # A stale adjustment vintage in the old archive must never be reused.
    quote["adj close"] = quote.close * 0.9
    return daily, quote


def test_reconciles_raw_quotes_with_one_adjustment_and_volume_basis():
    daily, quote = daily_and_quotes()
    before = quote.copy()
    daily.loc[5, "close"] += 0.05  # provider-specific quote discrepancy, not adjustment
    daily.loc[5, "aclose"] = daily.loc[5, "close"] * daily.loc[5, "adj_factor"]
    result, report = reconcile_quote_history(daily, quote)
    np.testing.assert_allclose(result.close, quote.close * 0.5)
    assert result.volume.eq(200).all()  # source volume basis is consistent throughout
    assert report["archived_quote_rows_used"] == 25
    pd.testing.assert_frame_equal(quote, before)


def test_adjustment_math_and_raw_unit_mismatch_fail():
    daily, quote = daily_and_quotes()
    daily.loc[0, "aclose"] = 999
    with pytest.raises(ValueError, match="adjustment factor"):
        reconcile_quote_history(daily, quote)
    daily, quote = daily_and_quotes()
    quote["close"] *= 2
    with pytest.raises(ValueError, match="units or issuer"):
        reconcile_quote_history(daily, quote)


def test_no_reference_uses_published_adjusted_prices_without_inventing_values():
    daily, _ = daily_and_quotes()
    result, report = reconcile_quote_history(daily, None)
    np.testing.assert_allclose(result.close, daily.aclose)
    assert report["archived_quote_rows_used"] == 0
