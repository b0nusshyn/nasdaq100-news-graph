import pandas as pd
import pytest

from experiments.gat_data_scaling_v1.study import nested


def test_nested_sizes_share_last_date_and_do_not_reorder():
    d = pd.bdate_range("2022-01-03", periods=253)
    assert [len(nested(d, f)) for f in (.25, .5, .75, 1)] == [64, 127, 190, 253]
    assert nested(d, .25).equals(nested(d, .50)[-64:])
    assert all(nested(d, f)[-1] == d[-1] for f in (.25, .50, .75, 1.))
    with pytest.raises(ValueError):
        nested(d[::-1], .5)
    with pytest.raises(ValueError):
        nested(d, 0)


def test_age_weights_are_fit_only_and_halve_over_252_sessions():
    from types import SimpleNamespace

    import numpy as np

    from experiments.gat_data_scaling_v1.recency import age_weights

    days = pd.bdate_range("2020-01-01", periods=260)
    p = SimpleNamespace(days=days, day_index=days.get_indexer)
    w = age_weights(p, days[:253])
    assert w[0] == .5 and w[252] == 1
    assert np.isnan(w[253:]).all()
