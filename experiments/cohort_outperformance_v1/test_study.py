import numpy as np
import pandas as pd

from experiments.cohort_outperformance_v1.study import partition, targets
from src.data.build_index_history import trading_days
from src.models.train import training_partitions


def test_complete_benchmark_and_exact_endpoint():
    dates = pd.bdate_range("2025-01-02", periods=8)
    closes = pd.DataFrame({"A": [100, 110, 110, 110, 110, 120, 120, 120],
                           "B": [100, 104, 104, 104, 104, 106, 106, 106]}, index=dates)
    row = targets(closes, dates, 5).query("date == @dates[0]")
    np.testing.assert_allclose(row.excess, [.07, -.07])
    assert row.y.tolist() == [1, 0]
    missing = targets(closes.drop(dates[5]), dates, 5)
    assert dates[0] not in set(missing.date)
    assert dates[1] in set(missing.date)


def test_zero_excess_is_not_outperformance_and_invalid_endpoints_drop_every_stock():
    dates = pd.bdate_range("2025-01-02", periods=4)
    closes = pd.DataFrame({"A": [100, 105, 105, 105], "B": [200, 210, np.inf, 210]}, index=dates)
    valid = targets(closes, dates, 1)
    assert valid.date.tolist() == [dates[0], dates[0]]
    assert valid.y.tolist() == [0, 0]


def test_purge_covers_exact_horizon_without_using_2026():
    days = trading_days("2023-01-01", "2025-12-31")
    for year in (2024, 2025):
        fold, cal = partition(days, year)
        fit, val = training_partitions(fold, {"embargo_days": 5, "train": {"validation_fraction": 25 / 282}})
        assert tuple(map(len, (fit, val, cal))) == (252, 25, 50)
        for before, after in ((fit, val), (val, cal), (cal, fold.test)):
            assert days.get_loc(before[-1]) + 5 < days.get_loc(after[0])
        assert days[days.get_loc(fold.test[-1]) + 5].year == year
