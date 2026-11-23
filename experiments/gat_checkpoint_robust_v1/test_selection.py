import pandas as pd
import pytest

from experiments.gat_checkpoint_robust_v1.study import worst_half


def test_worst_half_is_chronological_and_penalizes_inconsistent_ranking():
    dates = pd.bdate_range("2023-01-02", periods=4)
    frame = pd.DataFrame({"date": dates.repeat(2), "y_true": [1, 0] * 4,
                          "p_up": [.9, .1, .9, .1, .1, .9, .5, .5]})
    assert worst_half(frame.sample(frac=1, random_state=3)) == .25
    with pytest.raises(ValueError):
        worst_half(frame[frame.date.eq(dates[0])])
