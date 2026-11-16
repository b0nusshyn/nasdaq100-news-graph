import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.core.io import read_table  # noqa: E402
from src.core.paths import FIXTURES_DIR  # noqa: E402
from src.data.indicators import FEATURES, compute_indicators  # noqa: E402
from src.models.baselines.logistic import Logistic  # noqa: E402


@pytest.fixture
def panel():
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    labels = read_table("labels", FIXTURES_DIR / "labels.parquet")
    features = compute_indicators(prices)
    return (
        prices.merge(labels, on=["date", "ticker"]).merge(features, on=["date", "ticker"]).set_index(["date", "ticker"])
    )


def test_scaler_and_imputer_use_only_eligible_labelled_training_rows(panel):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    training_dates = dates[:40]
    model = Logistic()
    model.fit(panel, training_dates, 1)
    training = panel.loc[panel.index.get_level_values("date").isin(training_dates) & panel["in_index"]]
    training = training.dropna(subset=["y_1d"])
    imputer, scaler, classifier = model.pipeline.steps
    np.testing.assert_allclose(imputer[1].statistics_, training[list(FEATURES)].median())
    np.testing.assert_allclose(scaler[1].mean_, imputer[1].transform(training[list(FEATURES)]).mean(axis=0))
    assert classifier[1].C == 1.0

    changed = panel.copy()
    test_rows = changed.index.get_level_values("date") >= dates[40]
    changed.loc[test_rows, list(FEATURES)] = 1e8
    changed.loc[test_rows, "y_1d"] = 1
    second = Logistic()
    second.fit(changed, training_dates, 1)
    np.testing.assert_allclose(second.pipeline[-1].coef_, model.pipeline[-1].coef_)
    before = model.pipeline[1].mean_.copy()
    model.predict_proba(changed, dates[45:50])
    np.testing.assert_array_equal(model.pipeline[1].mean_, before)


@pytest.mark.parametrize("horizon", [1, 5])
def test_missing_test_features_and_labels_keep_rows(panel, horizon):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    model = Logistic()
    model.fit(panel, dates[:40], horizon)
    original = panel.copy(deep=True)
    test = panel.copy()
    test.loc[:, list(FEATURES)] = np.nan
    test = test.drop(columns=["y_1d", "y_5d", "fwd_ret_1d", "fwd_ret_5d"])
    output = model.predict_proba(test, dates[-5:])
    assert len(output) == 30
    assert output.between(0, 1).all()
    assert str(output.dtype) == "float32"
    pd.testing.assert_frame_equal(panel, original)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_fallback(panel, label):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    panel["y_1d"] = label
    model = Logistic()
    with pytest.warns(UserWarning, match="class frequency"):
        model.fit(panel, dates[:40], 1)
    assert model.predict_proba(panel, dates[45:]).eq(label).all()


def test_all_missing_training_features_and_empty_selection(panel):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    panel.loc[:, list(FEATURES)] = np.nan
    model = Logistic()
    with pytest.warns(UserWarning, match="class frequency"):
        model.fit(panel, dates[:40], 5)
    assert model.predict_proba(panel, dates[45:]).eq(np.float32(model.fallback_probability)).all()
    assert model.predict_proba(panel, pd.DatetimeIndex([])).empty


def test_empty_feature_column_is_retained(panel):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    panel["macd_signal"] = np.nan
    model = Logistic()
    model.fit(panel, dates[:40], 1)
    assert model.predict_proba(panel, dates[45:]).between(0, 1).all()


@pytest.mark.parametrize("horizon", [True, 0, 2, 1.0])
def test_invalid_horizon(panel, horizon):
    with pytest.raises(ValueError, match="horizon"):
        Logistic().fit(panel, pd.DatetimeIndex([]), horizon)


def test_missing_labels_and_prediction_before_fit(panel):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    with pytest.raises(ValueError, match="call fit"):
        Logistic().predict_proba(panel, dates[-5:])
    with pytest.raises(ValueError, match="nonmissing binary labels"):
        Logistic().fit(panel.assign(y_1d=pd.NA), dates[:40], 1)
    model = Logistic()
    model.fit(panel, dates[:40], 1)
    with pytest.raises(ValueError, match="follow"):
        model.predict_proba(panel, dates[:5])
