import numpy as np
import pandas as pd
import pytest

from experiments.prospective_feed_v1.overlap import compare


def test_comparison_detects_adjustments_and_volume_and_rejects_incomplete_panels():
    dates = pd.bdate_range("2025-01-01", periods=61)
    tickers = ["A", "B"]
    f = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"]).to_frame(index=False)
    i = np.arange(len(f))
    f["close"] = 100 + i/10 + np.sin(i/3)
    f = f.assign(open=f.close, high=f.close+1, low=f.close-1, adj_close=f.close, volume=100+i)
    out = compare(f, f, dates, tickers)
    assert out["volume_exact_rows"] == 122
    assert out["adjusted_daily_close_return"]["max_abs"] == 0
    assert all(k["max_abs"] == 0 for k in out["frozen_features"].values())
    changed = f.copy()
    changed.loc[90, "adj_close"] *= 1.02
    changed.loc[91, "volume"] *= 2
    out = compare(f, changed, dates, tickers)
    assert out["volume_exact_rows"] == 121
    assert out["adjusted_daily_close_return"]["max_abs"] > .01
    assert out["frozen_features"]["volume_relative"]["max_abs"] > .6
    with pytest.raises(ValueError, match="exactly one row"):
        compare(f, f.iloc[:-1], dates, tickers)
