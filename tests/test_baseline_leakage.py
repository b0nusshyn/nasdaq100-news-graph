"""Leak audit of baselines 1-4 (#35), and the fold adapter that gives them the shared procedure.

The two property tests from 3.6, applied to each baseline's own preprocessing:

1. Nothing fitted moves when data after the training window changes: fit on a panel whose
   post-window rows are scrambled, then predict from the clean panel; predictions must match a
   model fitted on the clean panel.
2. A prediction for day d does not change when every row after d is removed.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")
pytest.importorskip("xgboost")

from src.core.io import read_table  # noqa: E402
from src.core.paths import FIXTURES_DIR  # noqa: E402
from src.data.indicators import compute_indicators  # noqa: E402
from src.models.baselines import Arima, BuyAndHold, Logistic, XGBoost  # noqa: E402

FRAME_BASELINES = [BuyAndHold, Arima, Logistic, XGBoost]


@pytest.fixture(scope="module")
def frame():
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    labels = read_table("labels", FIXTURES_DIR / "labels.parquet")
    return (
        prices.merge(labels, on=["date", "ticker"])
        .merge(compute_indicators(prices), on=["date", "ticker"])
        .set_index(["date", "ticker"])
        .sort_index()
    )


@pytest.fixture(scope="module")
def dates(frame):
    return pd.DatetimeIndex(frame.index.get_level_values("date").unique()).sort_values()


def scrambled_after(frame: pd.DataFrame, day) -> pd.DataFrame:
    """Every numeric value after `day` replaced by noise; membership kept."""
    out = frame.copy()
    later = out.index.get_level_values("date") > day
    rng = np.random.default_rng(1)
    for col in out.columns:
        if col == "in_index":
            continue
        if col.startswith("y_"):
            out.loc[later, col] = rng.integers(0, 2, later.sum())
        else:
            out.loc[later, col] = out.loc[later, col].abs() * rng.uniform(0.2, 5.0, later.sum()) + 1.0
    return out


def quiet(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args)


@pytest.mark.parametrize("cls", FRAME_BASELINES, ids=lambda c: c.name)
@pytest.mark.parametrize("horizon", [1, 5])
def test_nothing_fitted_moves_when_post_window_data_changes(frame, dates, cls, horizon):
    train, test = dates[:35], dates[41:46]
    clean, dirty = cls(), cls()
    quiet(clean.fit, frame, train, horizon)
    quiet(dirty.fit, scrambled_after(frame, train[-1]), train, horizon)
    pd.testing.assert_series_equal(quiet(clean.predict_proba, frame, test), quiet(dirty.predict_proba, frame, test))


@pytest.mark.parametrize("cls", FRAME_BASELINES, ids=lambda c: c.name)
def test_a_prediction_does_not_depend_on_rows_after_its_day(frame, dates, cls):
    train, day = dates[:35], dates[43]
    model = cls()
    quiet(model.fit, frame, train, 1)
    cut = frame[frame.index.get_level_values("date") <= day]
    full = quiet(model.predict_proba, frame, pd.DatetimeIndex([day]))
    pd.testing.assert_series_equal(full, quiet(model.predict_proba, cut, pd.DatetimeIndex([day])))


def test_the_audit_catches_a_baseline_that_scales_over_the_whole_panel(frame, dates):
    # Negative control: a z-score fitted on every row, the leak this audit exists to find.
    class Leaky(Logistic):
        name = "leaky"

        def fit(self, panel, train_dates, horizon):
            whole = panel[list(self.features)]
            super().fit((panel.assign(**((whole - whole.mean()) / whole.std()))), train_dates, horizon)

    train, test = dates[:35], dates[41:46]
    clean, dirty = Leaky(), Leaky()
    quiet(clean.fit, frame, train, 1)
    quiet(dirty.fit, scrambled_after(frame, train[-1]), train, 1)
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(quiet(clean.predict_proba, frame, test),
                                       quiet(dirty.predict_proba, frame, test))


# --------------------------------------------------------------------------- fold adapter


@pytest.fixture(scope="module")
def panel_and_fold():
    from src.models.panel import build_panel
    from src.models.train import folds_for

    t = {n: read_table(n, FIXTURES_DIR / f"{n}.parquet") for n in ("prices", "labels", "news_features")}
    panel = build_panel(**t)
    cfg = {"period": {"start": "2021-01-01", "end": "2021-12-31"}, "train_days": 30, "test_days": 10,
           "step_days": 10, "expanding": False, "embargo_days": 5}
    return panel, folds_for(panel, cfg)[0]


@pytest.mark.parametrize("cls", FRAME_BASELINES, ids=lambda c: c.name)
def test_adapter_returns_exactly_the_shared_row_set(frame, panel_and_fold, cls):
    from src.core.schema import conform, validate
    from src.models.baselines.base import run_frame_baseline

    panel, fold = panel_and_fold
    preds, info = quiet(run_frame_baseline, cls(), frame, panel, fold, 5)
    expected = panel.rows(fold.test, 5)
    assert preds[["date", "ticker"]].reset_index(drop=True).equals(expected[["date", "ticker"]])
    assert (preds["y_true"].to_numpy() == expected["y"].to_numpy()).all()
    assert 0 <= info["threshold"] <= 1
    validate(conform(preds, "predictions"), "predictions")


def test_adapter_rejects_a_baseline_that_skips_rows(frame, panel_and_fold):
    from src.models.baselines.base import run_frame_baseline

    class SkipsOne(BuyAndHold):
        def predict_proba(self, panel, test_dates):
            return super().predict_proba(panel, test_dates).iloc[1:]

    panel, fold = panel_and_fold
    with pytest.raises(ValueError, match="no prediction for 1 rows"):
        run_frame_baseline(SkipsOne(), frame, panel, fold, 1)
