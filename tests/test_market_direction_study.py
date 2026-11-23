import numpy as np
import pandas as pd
import pytest

# The study module imports torch and scikit-learn at top level; CI's light set has neither.
pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("threadpoolctl")

from experiments.market_direction_v1.study import features, selective  # noqa: E402


def test_market_features_are_prefix_invariant_and_exclude_nonmembers():
    dates = pd.bdate_range("2020-01-02", periods=60)
    rows = []
    for i, d in enumerate(dates):
        for j, ticker in enumerate(("A", "B", "C")):
            x = 100 * np.exp(.01 * (j + 1) * i)
            rows.append((d, ticker, x, 1.01*x, .99*x, 1000 + i, ticker != "C"))
    f = pd.DataFrame(rows, columns=["date", "ticker", "close", "high", "low", "volume", "in_index"])
    full, macro = features(f)
    left, _ = features(f[f.date <= dates[49]])
    pd.testing.assert_frame_equal(left, full[full.date <= dates[49]])
    assert set(left.ticker) == {"A", "B"}
    assert left.loc[left.date.eq(dates[49]), "market_1"].tolist() == [left.market_1.iloc[-1]] * 2
    np.testing.assert_allclose(left.market_1.dropna(), .015)
    assert "breadth" in macro


def test_abstention_uses_fixed_threshold_and_counts_actual_coverage_and_directions():
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-03"]),
                          "ticker": ["A", "A", "B"], "confidence": [.01, .1, .2],
                          "pred_up": [True, True, False], "y_true": [0, 1, 1],
                          "correct": [False, True, False]})
    result = selective(frame, .1)
    assert result["coverage"] == 2/3
    assert result["dates"] == 1 and result["rows"] == 2
    assert result["accuracy"] == .5 and result["majority"] == 1
    assert result["up_accuracy"] == 1 and result["down_accuracy"] == 0
