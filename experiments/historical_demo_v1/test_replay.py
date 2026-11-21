import pandas as pd
import pytest

from experiments.historical_demo_v1.replay import compare


def test_probability_comparison_is_keyed_and_rejects_label_drift():
    fresh = pd.DataFrame({"date": pd.to_datetime(["2022-07-18"] * 2), "ticker": ["A", "B"],
                          "y_true": [0, 1], "p_up": [.4, .7]})
    assert compare(fresh, fresh.iloc[::-1]) == 0
    with pytest.raises(ValueError, match="keys or labels differ"):
        compare(fresh, fresh.assign(y_true=[1, 1]))
    with pytest.raises(ValueError, match="keys or labels differ"):
        compare(fresh, fresh.iloc[:1])
