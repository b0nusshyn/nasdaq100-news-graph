import pandas as pd
import pytest

from src.core.config import walk_forward_config
from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.eval.splitter import walk_forward


@pytest.fixture
def trading_days():
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    return pd.DatetimeIndex(prices["date"].unique()).sort_values()


@pytest.mark.parametrize("horizon", [1, 5])
@pytest.mark.parametrize("expanding", [False, True])
def test_folds_and_label_leakage(trading_days, horizon, expanding):
    folds = walk_forward(trading_days, train_days=20, test_days=10, horizon=horizon, expanding=expanding)

    assert len(folds) == 3
    for index, fold in enumerate(folds):
        test_start = 25 + index * 10
        train_start = 0 if expanding else index * 10
        assert fold.index == index
        pd.testing.assert_index_equal(fold.train, trading_days[train_start:test_start - 5])
        pd.testing.assert_index_equal(fold.test, trading_days[test_start:test_start + 10])
        assert fold.train.intersection(fold.test).empty
        # Every training label ends before the first test day.
        label_end_positions = trading_days.get_indexer(fold.train) + horizon
        assert (label_end_positions < test_start).all()


@pytest.mark.parametrize("use_default_embargo", [False, True])
def test_shared_config_gives_identical_folds(use_default_embargo):
    config = walk_forward_config()
    days = pd.bdate_range("2019-01-01", periods=900)
    settings = {key: value for key, value in config.items() if key != "period"}
    if use_default_embargo:
        settings.pop("embargo_days")
    one_day = walk_forward(days, horizon=1, **settings)
    five_day = walk_forward(days, horizon=5, **settings)

    assert len(one_day) == len(five_day) == 3
    for index, (first, second) in enumerate(zip(one_day, five_day, strict=True)):
        pd.testing.assert_index_equal(first.train, second.train)
        pd.testing.assert_index_equal(first.test, second.test)
        train_start = index * 126
        pd.testing.assert_index_equal(first.train, days[train_start:train_start + 504])
        pd.testing.assert_index_equal(first.test, days[train_start + 509:train_start + 635])
        embargo = days[(days > first.train[-1]) & (days < first.test[0])]
        pd.testing.assert_index_equal(embargo, days[train_start + 504:train_start + 509])
        assert len(first.train) == 504
        assert len(embargo) == 5
        assert len(first.test) == 126


@pytest.mark.parametrize("horizon", [1, 5])
@pytest.mark.parametrize("n_days, expected", [(634, 0), (635, 1), (760, 1), (761, 2)])
def test_config_requires_complete_635_day_fold(horizon, n_days, expected):
    config = walk_forward_config()
    settings = {key: value for key, value in config.items() if key != "period"}
    days = pd.bdate_range("2019-01-01", periods=n_days)
    assert len(walk_forward(days, horizon=horizon, **settings)) == expected


def test_embargo_counts_trading_days(trading_days):
    days = trading_days.delete(27)  # Simulate a market holiday in the embargo.
    fold = walk_forward(days, train_days=25, test_days=10, horizon=5)[0]
    pd.testing.assert_index_equal(fold.train, days[:25])
    pd.testing.assert_index_equal(fold.test, days[30:40])
    assert (fold.test[0] - fold.train[-1]).days > 5


def test_custom_step_and_partial_final_window(trading_days):
    folds = walk_forward(trading_days[:59], train_days=20, test_days=10, step_days=15, horizon=1)
    assert len(folds) == 2
    pd.testing.assert_index_equal(folds[1].test, trading_days[40:50])


@pytest.mark.parametrize("n_days, expected", [(0, 0), (34, 0), (35, 1)])
def test_minimum_data(trading_days, n_days, expected):
    assert len(walk_forward(trading_days[:n_days], train_days=20, test_days=10, horizon=5)) == expected


def test_training_window_can_be_shorter_than_embargo(trading_days):
    fold = walk_forward(trading_days, train_days=3, test_days=10, horizon=5)[0]
    pd.testing.assert_index_equal(fold.train, trading_days[:3])
    pd.testing.assert_index_equal(fold.test, trading_days[8:18])


@pytest.mark.parametrize(
    "settings",
    [
        {"train_days": 0}, {"train_days": -1}, {"test_days": -1},
        {"test_days": 2.5}, {"train_days": True}, {"horizon": 2},
        {"horizon": True}, {"step_days": 0}, {"step_days": 5},
        {"embargo_days": 1}, {"embargo_days": 0}, {"expanding": "false"},
    ],
)
def test_invalid_settings(trading_days, settings):
    arguments = {"train_days": 20, "test_days": 10, "horizon": 5}
    arguments.update(settings)
    with pytest.raises(ValueError):
        walk_forward(trading_days, **arguments)


@pytest.mark.parametrize(
    "days",
    [
        pd.DatetimeIndex(["2021-01-05", "2021-01-04"]),
        pd.DatetimeIndex(["2021-01-04", "2021-01-04"]),
        pd.DatetimeIndex(["2021-01-04", pd.NaT]),
        pd.date_range("2021-01-04", periods=2, tz="America/New_York"),
        pd.date_range("2021-01-04 16:00", periods=2),
    ],
)
def test_invalid_dates(days):
    with pytest.raises(ValueError):
        walk_forward(days, train_days=20, test_days=10, horizon=1)


def test_requires_datetime_index():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        walk_forward(["2021-01-04"], train_days=20, test_days=10, horizon=1)
