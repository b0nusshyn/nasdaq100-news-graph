import warnings

import numpy as np
import pandas as pd
import pytest

from src.core.config import walk_forward_config
from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.core.schema import conform, validate
from src.data.indicators import FEATURES, compute_indicators
from src.eval.backtest import backtest
from src.eval.metrics import classification_metrics
from src.eval.splitter import walk_forward
from src.models.baselines.lstm_price import LSTMPrice


@pytest.fixture
def torch_library():
    return pytest.importorskip("torch")


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


@pytest.fixture
def sequence_panel():
    days = pd.bdate_range("2021-01-04", periods=8).as_unit("ns")
    index = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "ticker"])
    frame = pd.DataFrame(index=index)
    for offset, feature in enumerate(FEATURES):
        frame[feature] = np.repeat(np.arange(8), 2) + np.tile([0, 100], 8) + offset
    frame["in_index"] = True
    return frame.astype({feature: "float64" for feature in FEATURES})


@pytest.fixture
def sequence_model():
    model = LSTMPrice(window_days=3)
    model.feature_medians = pd.Series(0.0, index=FEATURES)
    model.feature_means = pd.Series(0.0, index=FEATURES)
    model.feature_scales = pd.Series(1.0, index=FEATURES)
    return model


def test_sequences_are_chronological_and_keep_endpoint_positions(sequence_panel, sequence_model):
    days = sequence_panel.index.get_level_values("date").unique()
    endpoints = pd.MultiIndex.from_tuples(
        [(days[5], "BBB"), (days[2], "AAA"), (days[0], "AAA")], names=["date", "ticker"]
    )
    shuffled = sequence_panel.sample(frac=1, random_state=33)
    sequences, positions = sequence_model._prepare_sequences(shuffled, endpoints)
    assert sequences.shape == (2, 3, 6)
    assert str(sequences.dtype) == "float32"
    for sequence, position in zip(sequences, positions, strict=True):
        date, ticker = endpoints[position]
        stop = days.get_loc(date) + 1
        expected = sequence_panel.xs(ticker, level="ticker").loc[days[stop - 3 : stop], list(FEATURES)]
        np.testing.assert_array_equal(sequence, expected.to_numpy("float32"))
    assert set(positions) == {0, 1}


def test_missing_session_breaks_window_instead_of_compressing_history(sequence_panel, sequence_model):
    days = sequence_panel.index.get_level_values("date").unique()
    panel = sequence_panel.drop(index=(days[3], "AAA"))
    endpoints = pd.MultiIndex.from_tuples(
        [(days[4], "AAA"), (days[5], "AAA"), (days[6], "AAA")], names=["date", "ticker"]
    )
    sequences, positions = sequence_model._prepare_sequences(panel, endpoints)
    assert positions.tolist() == [2]
    np.testing.assert_array_equal(sequences[0, :, 0], [4, 5, 6])


def test_sequence_imputation_never_backfills_from_future(sequence_panel, sequence_model):
    days = sequence_panel.index.get_level_values("date").unique()
    sequence_panel.loc[(days[2], "AAA"), "ret_1d"] = np.nan
    sequence_panel.loc[(days[3], "AAA"), "ret_1d"] = 9999
    endpoints = pd.MultiIndex.from_tuples([(days[2], "AAA")], names=["date", "ticker"])
    sequences, positions = sequence_model._prepare_sequences(sequence_panel, endpoints)
    np.testing.assert_array_equal(sequences[0, :, 0], [0, 1, 0])
    assert positions.tolist() == [0]


def test_default_twenty_day_windows_end_at_the_prediction_date(panel, dates):
    model = LSTMPrice()
    assert model.window_days == 20
    model.feature_medians = pd.Series(0.0, index=FEATURES)
    model.feature_means = pd.Series(0.0, index=FEATURES)
    model.feature_scales = pd.Series(1.0, index=FEATURES)
    endpoints = pd.MultiIndex.from_tuples([(dates[19], "AAA"), (dates[40], "BBB")], names=["date", "ticker"])
    sequences, positions = model._prepare_sequences(panel, endpoints)
    assert sequences.shape == (2, 20, 6)
    for sequence, position in zip(sequences, positions, strict=True):
        date, ticker = endpoints[position]
        stop = dates.get_loc(date) + 1
        history = panel.xs(ticker, level="ticker").loc[dates[stop - 20 : stop], list(FEATURES)]
        expected = history.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy("float32")
        np.testing.assert_array_equal(sequence, expected)


@pytest.mark.parametrize("horizon", [1, 5])
@pytest.mark.parametrize("window_days", [3, 20])
def test_actual_fitting_schema_metrics_and_backtest(panel, dates, torch_library, horizon, window_days):
    from src.models.encoder import CompanyEncoder

    original = panel.copy(deep=True)
    labels = read_table("labels", FIXTURES_DIR / "labels.parquet")
    fold = walk_forward(dates, train_days=35, test_days=5, horizon=horizon)[0]
    model = LSTMPrice(window_days=window_days, hidden=4, epochs=1, batch_size=256)
    assert model.fit(panel, fold.train, horizon) is None
    assert model.model is not None
    assert isinstance(model.model[0], CompanyEncoder)
    assert model.model[0].lstm.input_size == 6
    assert model.model[0].lstm.num_layers == 1
    assert not model.model.training

    probabilities = model.predict_proba(panel, fold.test)
    date_index = panel.index.get_level_values("date")
    expected = panel.loc[date_index.isin(fold.test) & panel["in_index"]]
    pd.testing.assert_index_equal(probabilities.index, expected.index)
    assert probabilities.name == "p_up"
    assert str(probabilities.dtype) == "float32"
    assert probabilities.between(0, 1).all()
    assert len(dates[(dates > fold.train[-1]) & (dates < fold.test[0])]) == 5

    output = probabilities.to_frame().join(panel[f"y_{horizon}d"].rename("y_true"))
    output = output.dropna(subset=["y_true"]).reset_index()
    output = conform(output.assign(model=model.name, horizon=horizon, fold=fold.index), "predictions")
    validate(output, "predictions")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metrics = classification_metrics(output["y_true"].to_numpy(), output["p_up"].to_numpy())
    assert 0 <= metrics["accuracy"] <= 1
    assert np.isfinite(metrics["auc"])
    result = backtest(output, labels, horizon)
    assert np.isfinite(result.cumulative_return)
    assert result.equity.notna().all()
    pd.testing.assert_frame_equal(panel, original)


def test_training_updates_shared_encoder_weights(panel, dates, torch_library, monkeypatch):
    from src.models.encoder import CompanyEncoder

    original_init = CompanyEncoder.__init__
    initial_weights = []

    def record_initial_weights(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        initial_weights.append(self.lstm.weight_ih_l0.detach().clone())

    monkeypatch.setattr(CompanyEncoder, "__init__", record_initial_weights)
    model = LSTMPrice(window_days=3, hidden=4, epochs=1, batch_size=256)
    model.fit(panel, dates[:35], 1)
    trained = model.model[0].lstm.weight_ih_l0.detach()
    assert not torch_library.equal(initial_weights[0], trained)


@pytest.mark.parametrize("horizon", [1, 5])
def test_preprocessing_uses_only_eligible_labelled_training_rows(panel, dates, horizon):
    panel = panel.copy()
    panel[f"y_{horizon}d"] = 1
    panel.loc[(dates[10], "AAA"), f"y_{horizon}d"] = pd.NA
    panel.loc[(dates[11], "AAA"), "ret_1d"] = np.inf
    selected = panel.index.get_level_values("date").isin(dates[:35]) & panel["in_index"]
    training = panel.loc[selected].dropna(subset=[f"y_{horizon}d"])
    features = training[list(FEATURES)].replace([np.inf, -np.inf], np.nan)
    median = features.median().fillna(0)
    imputed = features.fillna(median)
    model = LSTMPrice(window_days=3)
    with pytest.warns(UserWarning, match="training class frequency"):
        model.fit(panel, dates[:35], horizon)
    pd.testing.assert_series_equal(model.feature_medians, median)
    pd.testing.assert_series_equal(model.feature_means, imputed.mean())
    pd.testing.assert_series_equal(model.feature_scales, imputed.std(ddof=0).replace(0, 1))


@pytest.mark.parametrize("horizon", [1, 5])
def test_held_out_data_never_changes_fitting_or_earlier_predictions(panel, dates, torch_library, horizon):
    settings = dict(window_days=3, hidden=4, epochs=1, batch_size=256)
    first = LSTMPrice(**settings)
    random_state = torch_library.random.get_rng_state().clone()
    first.fit(panel, dates[:35], horizon)
    assert torch_library.equal(random_state, torch_library.random.get_rng_state())
    expected = first.predict_proba(panel, dates[40:45])
    weights = {key: value.clone() for key, value in first.model.state_dict().items()}

    changed = panel.copy()
    future = changed.index.get_level_values("date") >= dates[35]
    changed.loc[future, list(FEATURES)] = 1e6
    changed.loc[future, f"y_{horizon}d"] = 1
    second = LSTMPrice(**settings)
    second.fit(changed, dates[:35], horizon)
    pd.testing.assert_series_equal(first.feature_medians, second.feature_medians)
    pd.testing.assert_series_equal(first.feature_means, second.feature_means)
    pd.testing.assert_series_equal(first.feature_scales, second.feature_scales)
    for key, value in weights.items():
        torch_library.testing.assert_close(value, second.model.state_dict()[key], rtol=0, atol=0)

    changed = panel.copy()
    later = changed.index.get_level_values("date") > dates[42]
    changed.loc[later, list(FEATURES)] = 1e6
    changed["sentiment"] = 1e6
    changed["emb_0"] = 1e6
    changed["fwd_ret_1d"] = 9999
    changed = changed.drop(columns=["y_1d", "y_5d"])
    actual = first.predict_proba(changed, dates[40:43])
    pd.testing.assert_series_equal(actual, expected.loc[expected.index.get_level_values("date") <= dates[42]])
    for key, value in weights.items():
        torch_library.testing.assert_close(value, first.model.state_dict()[key], rtol=0, atol=0)


def test_deterministic_predictions_and_shuffled_panel(panel, dates, torch_library):
    settings = dict(window_days=3, hidden=4, epochs=1, batch_size=256)
    first = LSTMPrice(**settings)
    second = LSTMPrice(**settings)
    first.fit(panel, dates[:35], 1)
    second.fit(panel.sample(frac=1, random_state=33), dates[:35], 1)
    expected = first.predict_proba(panel, dates[40:45])
    actual = second.predict_proba(panel.sample(frac=1, random_state=34), dates[40:45])
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index())


def test_membership_and_prejoin_history_as_context(panel, dates, torch_library):
    model = LSTMPrice(window_days=3, hidden=4, epochs=1, batch_size=256)
    model.fit(panel, dates[:20], 1)
    probabilities = model.predict_proba(panel, dates[[29, 30]])
    assert (dates[29], "FFF") not in probabilities.index
    assert (dates[30], "FFF") in probabilities.index
    endpoints = pd.MultiIndex.from_tuples([(dates[30], "FFF")], names=["date", "ticker"])
    sequences, positions = model._prepare_sequences(panel, endpoints)
    assert sequences.shape == (1, 3, 6)
    assert positions.tolist() == [0]


def test_missing_features_and_incomplete_windows_preserve_rows(panel, dates, torch_library):
    model = LSTMPrice(window_days=3, hidden=4, epochs=1, batch_size=256)
    model.fit(panel, dates[:35], 1)
    changed = panel.drop(index=(dates[41], "AAA")).copy()
    changed.loc[(dates[40], "BBB"), list(FEATURES)] = np.nan
    changed.loc[(dates[40], "CCC"), "ret_1d"] = np.inf
    with pytest.warns(UserWarning, match="2 incomplete windows"):
        probabilities = model.predict_proba(changed, dates[40:45])
    assert len(probabilities) == 29
    assert probabilities.between(0, 1).all()
    assert probabilities.loc[(dates[42], "AAA")] == np.float32(model.fallback_probability)
    assert probabilities.loc[(dates[43], "AAA")] == np.float32(model.fallback_probability)
    assert model.predict_proba(panel, pd.DatetimeIndex([])).empty
    assert model.predict_proba(panel.iloc[:0], dates[40:45]).empty


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_uses_prior_without_neural_training(panel, dates, label):
    model = LSTMPrice(window_days=3)
    with pytest.warns(UserWarning, match="training class frequency"):
        model.fit(panel.assign(y_1d=label), dates[:35], 1)
    assert model.model is None
    assert model.predict_proba(panel, dates[40:45]).eq(label).all()


@pytest.mark.parametrize("reason", ["short_history", "missing_labels", "missing_features"])
def test_unusable_training_data_has_explicit_safe_fallback(panel, dates, reason):
    window_days = 100 if reason == "short_history" else 3
    if reason == "missing_labels":
        panel["y_1d"] = pd.NA
    elif reason == "missing_features":
        panel.loc[:, list(FEATURES)] = np.nan
    model = LSTMPrice(window_days=window_days)
    with pytest.warns(UserWarning, match="insufficient history, classes or features"):
        model.fit(panel, dates[:35], 1)
    assert model.model is None
    probabilities = model.predict_proba(panel, dates[40:45])
    assert probabilities.between(0, 1).all()
    assert probabilities.eq(np.float32(model.fallback_probability)).all()
    if reason == "missing_labels":
        assert probabilities.eq(0.5).all()


def test_production_calendar_has_same_embargo_for_both_horizons():
    config = walk_forward_config()
    settings = {key: value for key, value in config.items() if key != "period"}
    days = pd.bdate_range("2019-01-02", periods=635)
    first = walk_forward(days, horizon=1, **settings)[0]
    fifth = walk_forward(days, horizon=5, **settings)[0]
    assert len(first.train) == 504
    assert len(first.test) == 126
    assert len(days[(days > first.train[-1]) & (days < first.test[0])]) == 5
    pd.testing.assert_index_equal(first.train, fifth.train)
    pd.testing.assert_index_equal(first.test, fifth.test)


@pytest.mark.parametrize("horizon", [0, 2, True, 1.0, "1", None])
def test_invalid_horizon(panel, dates, horizon):
    with pytest.raises(ValueError, match="horizon"):
        LSTMPrice().fit(panel, dates[:35], horizon)


@pytest.mark.parametrize(
    "settings",
    [
        {"window_days": 0},
        {"hidden": -1},
        {"epochs": True},
        {"batch_size": 1.5},
        {"learning_rate": 0},
        {"learning_rate": np.nan},
        {"seed": -1},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        LSTMPrice(**settings)


def test_invalid_dates_index_features_and_labels(panel, dates):
    for invalid in [pd.DatetimeIndex([]), dates[:5][::-1], dates[[0, 0]], pd.DatetimeIndex([pd.NaT])]:
        with pytest.raises(ValueError, match="train_dates"):
            LSTMPrice().fit(panel, invalid, 1)
    with pytest.raises(ValueError, match="missing columns"):
        LSTMPrice().fit(panel.drop(columns="ret_1d"), dates[:35], 1)
    with pytest.raises(ValueError, match="unique.*index"):
        LSTMPrice().fit(pd.concat([panel, panel.iloc[:1]]), dates[:35], 1)
    panel.loc[(dates[10], "AAA"), "y_1d"] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        LSTMPrice().fit(panel, dates[:35], 1)


def test_prediction_before_fit_and_overlapping_dates(panel, dates):
    model = LSTMPrice()
    with pytest.raises(ValueError, match="call fit"):
        model.predict_proba(panel, dates[40:45])
    with pytest.warns(UserWarning):
        model.fit(panel.assign(y_1d=1), dates[:35], 1)
    with pytest.raises(ValueError, match="follow"):
        model.predict_proba(panel, dates[:5])
