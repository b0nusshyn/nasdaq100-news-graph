import numpy as np
import pandas as pd
import pytest

from src.core.config import walk_forward_config
from src.core.fixtures import make_fixtures
from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.core.schema import conform, validate
from src.data.indicators import FEATURES, compute_indicators
from src.eval.backtest import backtest
from src.eval.metrics import classification_metrics
from src.eval.splitter import walk_forward
from src.models.baselines.xgb import XGBoost


@pytest.fixture
def xgboost_library():
    return pytest.importorskip("xgboost")


@pytest.fixture
def panel():
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    labels = read_table("labels", FIXTURES_DIR / "labels.parquet")
    indicators = compute_indicators(prices)
    return (
        prices.merge(labels, on=["date", "ticker"], validate="one_to_one")
        .merge(indicators, on=["date", "ticker"], validate="one_to_one")
        .set_index(["date", "ticker"])
    )


@pytest.fixture
def dates(panel):
    return pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()


@pytest.mark.parametrize("horizon", [1, 5])
def test_walk_forward_schema_metrics_and_backtest(panel, dates, xgboost_library, horizon):
    labels = read_table("labels", FIXTURES_DIR / "labels.parquet")
    original = panel.copy(deep=True)
    folds = walk_forward(dates[:-5], train_days=35, test_days=5, horizon=horizon)

    for fold in folds:
        model = XGBoost()
        assert model.fit(panel, fold.train, horizon) is None
        assert model.model is not None

        probabilities = model.predict_proba(panel, fold.test)
        date_index = panel.index.get_level_values("date")
        eligible_test = panel.loc[date_index.isin(fold.test) & panel["in_index"]]
        pd.testing.assert_index_equal(probabilities.index, eligible_test.index)
        assert not probabilities.empty
        assert probabilities.name == "p_up"
        assert str(probabilities.dtype) == "float32"
        assert probabilities.between(0, 1).all()
        assert len(dates[(dates > fold.train[-1]) & (dates < fold.test[0])]) == 5

        output = probabilities.to_frame().join(panel[f"y_{horizon}d"].rename("y_true"))
        output = output.dropna(subset=["y_true"]).reset_index()
        output = conform(output.assign(model=model.name, horizon=horizon, fold=fold.index), "predictions")
        validate(output, "predictions")

        metrics = classification_metrics(output["y_true"].to_numpy(), output["p_up"].to_numpy())
        assert 0 <= metrics["accuracy"] <= 1
        assert np.isfinite(metrics["auc"])

        result = backtest(output, labels, horizon)
        assert result.equity.notna().all()
        assert np.isfinite(result.cumulative_return)

    pd.testing.assert_frame_equal(panel, original)


@pytest.mark.parametrize("horizon", [1, 5])
def test_production_fold_sizes_without_changing_configuration(xgboost_library, horizon):
    tables = make_fixtures(n_days=640, seed=32)
    prices = conform(tables["prices"], "prices")
    labels = conform(tables["labels"], "labels")
    indicators = compute_indicators(prices)
    panel = prices.merge(labels, on=["date", "ticker"]).merge(indicators, on=["date", "ticker"])
    panel = panel.set_index(["date", "ticker"])
    days = pd.DatetimeIndex(prices["date"].unique()).sort_values()
    config = walk_forward_config()
    settings = {key: value for key, value in config.items() if key != "period"}
    fold = walk_forward(days, horizon=horizon, **settings)[0]

    assert len(fold.train) == 504
    assert len(fold.test) == 126
    assert len(days[(days > fold.train[-1]) & (days < fold.test[0])]) == 5

    model = XGBoost()
    model.fit(panel, fold.train, horizon)
    probabilities = model.predict_proba(panel, fold.test)
    assert model.model is not None
    assert len(probabilities) == 126 * 6
    assert probabilities.between(0, 1).all()


@pytest.mark.parametrize("horizon", [1, 5])
def test_fit_receives_only_eligible_labelled_training_data(panel, dates, xgboost_library, monkeypatch, horizon):
    original_fit = xgboost_library.XGBClassifier.fit
    observed = {}

    def record_fit(classifier, features, labels, **kwargs):
        observed["features"] = features.copy()
        observed["labels"] = labels.copy()
        observed["kwargs"] = kwargs
        return original_fit(classifier, features, labels, **kwargs)

    monkeypatch.setattr(xgboost_library.XGBClassifier, "fit", record_fit)
    panel.loc[(dates[25], "AAA"), f"y_{horizon}d"] = pd.NA
    panel.loc[(dates[26], "AAA"), "ret_1d"] = np.inf
    model = XGBoost()
    model.fit(panel, dates[:35], horizon)

    selected = panel.index.get_level_values("date").isin(dates[:35]) & panel["in_index"]
    training = panel.loc[selected].dropna(subset=[f"y_{horizon}d"])
    expected = training[list(FEATURES)].replace([np.inf, -np.inf], np.nan)
    pd.testing.assert_frame_equal(observed["features"], expected)
    pd.testing.assert_series_equal(observed["labels"], training[f"y_{horizon}d"].astype(int))
    assert observed["kwargs"] == {}  # No test eval_set or early stopping.
    assert model.model.get_booster().feature_names == list(FEATURES)


@pytest.mark.parametrize("horizon", [1, 5])
def test_held_out_changes_do_not_affect_fitting_or_earlier_predictions(panel, dates, xgboost_library, horizon):
    original = panel.copy(deep=True)
    first = XGBoost()
    first.fit(panel, dates[:35], horizon)
    expected = first.predict_proba(panel, dates[40:45])
    booster_before = first.model.get_booster().save_raw()

    changed = panel.copy()
    held_out = changed.index.get_level_values("date") >= dates[35]
    changed.loc[held_out, list(FEATURES)] = 1e6
    changed.loc[held_out, f"y_{horizon}d"] = 1
    second = XGBoost()
    second.fit(changed, dates[:35], horizon)
    assert first.model.get_booster().get_dump() == second.model.get_booster().get_dump()

    future = panel.index.get_level_values("date") > dates[42]
    changed = panel.copy()
    changed.loc[future, list(FEATURES)] = 1e6
    changed.loc[future, "close"] *= 10
    changed = changed.drop(columns=["y_1d", "y_5d", "fwd_ret_1d", "fwd_ret_5d"])
    actual = first.predict_proba(changed, dates[40:45])
    earlier = expected.index.get_level_values("date") <= dates[42]
    pd.testing.assert_series_equal(actual.loc[earlier], expected.loc[earlier])
    assert first.model.get_booster().save_raw() == booster_before
    pd.testing.assert_frame_equal(panel, original)


def test_deterministic_predictions_and_parameters(panel, dates, xgboost_library):
    first = XGBoost()
    second = XGBoost()
    first.fit(panel, dates[:35], 1)
    second.fit(panel, dates[:35], 1)
    pd.testing.assert_series_equal(first.predict_proba(panel, dates[40:]), second.predict_proba(panel, dates[40:]))
    parameters = first.model.get_params()
    assert parameters["random_state"] == 42
    assert parameters["n_jobs"] == 1
    assert parameters["max_depth"] == 3
    assert parameters["n_estimators"] == 100


def test_single_missing_feature_column_is_supported(panel, dates, xgboost_library):
    panel["macd_signal"] = np.nan
    model = XGBoost()
    model.fit(panel, dates[:35], 1)
    assert model.model is not None
    assert model.predict_proba(panel, dates[40:45]).between(0, 1).all()


@pytest.mark.parametrize("horizon", [1, 5])
def test_missing_test_features_labels_and_empty_selection(panel, dates, xgboost_library, horizon):
    model = XGBoost()
    model.fit(panel, dates[:35], horizon)
    test = panel.copy()
    test.loc[:, list(FEATURES)] = np.nan
    test.loc[(dates[-1], "AAA"), "ret_1d"] = -np.inf
    test = test.drop(columns=["y_1d", "y_5d", "fwd_ret_1d", "fwd_ret_5d"])
    probabilities = model.predict_proba(test, dates[-5:])
    assert len(probabilities) == 30
    assert probabilities.between(0, 1).all()
    assert model.predict_proba(test, pd.DatetimeIndex([])).empty
    assert model.predict_proba(test.iloc[:0], dates[-5:]).empty


def test_membership_and_shuffled_input_order(panel, dates, xgboost_library):
    model = XGBoost()
    model.fit(panel, dates[:20], 1)
    test_dates = dates[[29, 30]]
    probabilities = model.predict_proba(panel, test_dates)
    assert (dates[29], "FFF") not in probabilities.index
    assert (dates[30], "FFF") in probabilities.index

    shuffled = panel.sample(frac=1, random_state=32)
    actual = model.predict_proba(shuffled, test_dates)
    date_index = shuffled.index.get_level_values("date")
    pd.testing.assert_index_equal(actual.index, shuffled.loc[date_index.isin(test_dates) & shuffled["in_index"]].index)
    pd.testing.assert_series_equal(actual.sort_index(), probabilities.sort_index())


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_training_uses_constant_probability(panel, dates, label):
    panel["y_1d"] = label
    model = XGBoost()
    with pytest.warns(UserWarning, match="training class frequency"):
        model.fit(panel, dates[:35], 1)
    assert model.model is None
    assert model.predict_proba(panel, dates[40:45]).eq(label).all()


def test_too_few_training_rows_use_training_frequency(panel, dates):
    panel = panel.loc[panel.index.get_level_values("ticker") == "AAA"].copy()
    panel.loc[panel.index.get_level_values("date").isin(dates[:4]), "y_1d"] = [0, 1, 1, 1]
    model = XGBoost()
    with pytest.warns(UserWarning, match="insufficient rows"):
        model.fit(panel, dates[:4], 1)
    assert model.model is None
    assert model.predict_proba(panel, dates[40:45]).eq(0.75).all()


@pytest.mark.parametrize("reason", ["missing_labels", "no_members", "no_training_rows"])
def test_no_usable_labels_use_neutral_probability(panel, dates, reason):
    if reason == "missing_labels":
        panel["y_5d"] = pd.NA
    elif reason == "no_members":
        panel["in_index"] = False
        panel.loc[panel.index.get_level_values("date").isin(dates[40:45]), "in_index"] = True
    else:
        panel = panel.loc[panel.index.get_level_values("date") >= dates[40]]

    model = XGBoost()
    with pytest.warns(UserWarning, match="0.5 when no labels"):
        model.fit(panel, dates[:35], 5)
    probabilities = model.predict_proba(panel, dates[40:45])
    assert not probabilities.empty
    assert probabilities.eq(0.5).all()


def test_all_training_features_missing_use_training_frequency(panel, dates):
    panel.loc[:, list(FEATURES)] = np.nan
    model = XGBoost()
    with pytest.warns(UserWarning, match="features"):
        model.fit(panel, dates[:35], 1)
    assert model.model is None
    assert model.predict_proba(panel, dates[40:45]).eq(np.float32(model.fallback_probability)).all()


@pytest.mark.parametrize("horizon", [0, 2, True, 1.0, "1", None])
def test_invalid_horizon(panel, dates, horizon):
    with pytest.raises(ValueError, match="horizon"):
        XGBoost().fit(panel, dates[:35], horizon)


def test_invalid_training_dates_and_labels(panel, dates):
    for invalid_dates in [pd.DatetimeIndex([]), dates[:35][::-1], dates[[0, 0]], pd.DatetimeIndex([pd.NaT])]:
        with pytest.raises(ValueError, match="train_dates"):
            XGBoost().fit(panel, invalid_dates, 1)

    panel.loc[(dates[10], "AAA"), "y_1d"] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        XGBoost().fit(panel, dates[:35], 1)


def test_missing_feature_columns_and_invalid_index(panel, dates):
    with pytest.raises(ValueError, match="missing indicator columns"):
        XGBoost().fit(panel.drop(columns="ret_1d"), dates[:35], 1)
    with pytest.raises(ValueError, match="unique.*index"):
        XGBoost().fit(pd.concat([panel, panel.iloc[:1]]), dates[:35], 1)
    with pytest.raises(ValueError, match="unique.*index"):
        XGBoost().fit(panel.reset_index(), dates[:35], 1)


def test_prediction_before_fit_and_overlapping_dates(panel, dates):
    with pytest.raises(ValueError, match="call fit"):
        XGBoost().predict_proba(panel, dates[40:45])
    model = XGBoost()
    with pytest.warns(UserWarning):
        model.fit(panel.assign(y_1d=1), dates[:35], 1)
    with pytest.raises(ValueError, match="follow"):
        model.predict_proba(panel, dates[:5])
    with pytest.raises(ValueError, match="unique.*index"):
        model.predict_proba(pd.concat([panel, panel.iloc[-1:]]), dates[40:45])


def test_failed_fit_does_not_silently_enable_fallback_predictions(panel, dates, xgboost_library, monkeypatch):
    def fail_fit(*args, **kwargs):
        raise ValueError("synthetic fitting failure")

    monkeypatch.setattr(xgboost_library.XGBClassifier, "fit", fail_fit)
    model = XGBoost()
    with pytest.raises(ValueError, match="synthetic fitting failure"):
        model.fit(panel, dates[:35], 1)
    with pytest.raises(ValueError, match="call fit"):
        model.predict_proba(panel, dates[40:45])
