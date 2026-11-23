import numpy as np
import pandas as pd
import pytest

from experiments.offline_rank_v1.study import qids, residual_features


def test_qids_preserve_variable_date_groups():
    x = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-05"])})
    assert qids(x).tolist() == [0, 0, 1, 2]
    with pytest.raises(ValueError, match="sorted"):
        qids(x.iloc[::-1])


def test_future_prices_do_not_change_residual_features():
    dates = pd.bdate_range("2024-01-01", periods=85)
    p = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "ticker"]).to_frame(index=False)
    rng = np.random.default_rng(2)
    p["close"] = np.exp(rng.normal(0, .01, len(p))).reshape(85, 3).cumprod(axis=0).ravel()
    altered = p.copy()
    altered.loc[(p.date > dates[65]) & p.ticker.eq("B"), "close"] *= 1.3
    a = residual_features(p).query("date <= @dates[65]")
    b = residual_features(altered).query("date <= @dates[65]")
    assert a.res_20.notna().sum() > 0
    pd.testing.assert_frame_equal(a, b)
