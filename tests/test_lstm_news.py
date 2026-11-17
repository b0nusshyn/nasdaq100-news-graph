"""Task 4.7: verify the shared no-graph path without replacing upstream code."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.core.config import load_config
from src.core.io import read_table
from src.core.paths import CONFIGS_DIR, FIXTURES_DIR
from src.core.schema import conform, validate
from src.eval.backtest import backtest
from src.eval.metrics import classification_metrics
from src.eval.splitter import Fold, walk_forward
from src.models.baselines.lstm_news import LSTMNews
from src.models.panel import build_panel, fit_features


@pytest.fixture(scope="module")
def tables():
    return {name: read_table(name, FIXTURES_DIR / f"{name}.parquet") for name in ("prices", "labels", "news_features")}


@pytest.fixture(scope="module")
def panel(tables):
    return build_panel(**tables)


@pytest.fixture
def fold(panel):
    return walk_forward(panel.days, train_days=35, test_days=10, horizon=1)[0]


@pytest.fixture
def model(tmp_path):
    config = deepcopy(load_config(CONFIGS_DIR / "lstm_news.yaml"))
    # The fixture has too few covered training rows for the production 64 PCs.
    config["features"]["embedding_dim"] = 4
    config["window_days"] = 3
    config["encoder"].update(hidden=4, dropout=0.0)
    config["train"].update(epochs=1, batch_days=16, early_stopping_patience=1)
    path = tmp_path / "lstm_news.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return LSTMNews(path)


def test_production_inputs_and_training_match_the_proposed_model():
    baseline = LSTMNews().cfg
    proposed = load_config(CONFIGS_DIR / "gat_lstm.yaml")
    for key in ("seed", "horizons", "window_days", "features", "encoder", "train"):
        assert baseline[key] == proposed[key]
    assert baseline["gat"]["edge_types"] == []
    assert baseline["features"]["embedding_dim"] == 64


@pytest.mark.parametrize("horizon", [1, 5])
def test_wrapper_delegates_without_edges(model, panel, fold, monkeypatch, horizon):
    import src.models.train

    sentinel = (pd.DataFrame(), {"delegated": True})

    def record(panel_arg, edges, fold_arg, horizon_arg, cfg, **kwargs):
        assert panel_arg is panel and fold_arg is fold
        assert edges is None and cfg is model.cfg
        assert horizon_arg == horizon
        return sentinel

    monkeypatch.setattr(src.models.train, "train_fold", record)
    assert model.run_fold(panel, fold, horizon) is sentinel


@pytest.mark.parametrize("horizon", [0, 2, True, 1.0, "1", None])
def test_invalid_horizon(model, panel, fold, horizon):
    with pytest.raises(ValueError, match="horizon"):
        model.run_fold(panel, fold, horizon)


@pytest.mark.parametrize("problem", ["overlap", "short_gap", "reversed", "duplicate", "gap", "empty"])
def test_invalid_fold_is_rejected_before_training(model, panel, fold, problem):
    train, test = fold.train, fold.test
    if problem == "overlap":
        test = panel.days[30:40]
    elif problem == "short_gap":
        test = panel.days[39:49]
    elif problem == "reversed":
        train = train[::-1]
    elif problem == "duplicate":
        train = train.insert(0, train[0])
    elif problem == "gap":
        train = train.delete(10)
    else:
        test = pd.DatetimeIndex([])
    with pytest.raises(ValueError):
        model.run_fold(panel, Fold(0, train, test), 1)


def test_mutated_graph_config_is_rejected(model, panel, fold):
    model.cfg["gat"]["edge_types"] = ["sector"]
    with pytest.raises(ValueError, match="not a lstm_news config"):
        model.run_fold(panel, fold, 1)


def test_feature_order_shapes_missing_values_and_chronology(panel, fold):
    features = fit_features(panel, fold.train, window=3, embedding_dim=4)
    expected = [
        "ret_1d",
        "volatility_20d",
        "rsi_14",
        "macd",
        "macd_signal",
        "volume_z_20d",
        "has_full_window",
        "has_news",
        "sentiment",
        "confidence",
        "emb_pca_0",
        "emb_pca_1",
        "emb_pca_2",
        "emb_pca_3",
    ]
    assert features.names == expected
    assert np.isfinite(features.x).all()
    window = features.windows(42, np.array([0, 2]))
    assert window.shape == (2, 3, 14)
    np.testing.assert_array_equal(window[1], features.x[40:43, 2])
    early = features.windows(0, np.array([0]))
    assert (early[:, :2] == 0).all()
    assert features.x[0, 0, expected.index("has_full_window")] == 0


def test_absent_news_and_neutral_news_remain_distinct(panel, fold):
    news = panel.news.copy()
    news[41, 0] = [0, 0, 0]
    news[41, 1] = [1, 0, 0.95]
    features = fit_features(replace(panel, news=news), fold.train, window=3, embedding_dim=4)
    flag = features.names.index("has_news")
    assert features.x[41, 0, flag] == 0
    assert features.x[41, 1, flag] == 1
    no_news = panel.news[:, :, 0] == 0
    original = fit_features(panel, fold.train, window=3, embedding_dim=4)
    columns = [
        i for i, name in enumerate(original.names) if name in ("sentiment", "confidence") or name.startswith("emb_pca_")
    ]
    assert (original.x[no_news][:, columns] == 0).all()


def test_future_data_and_labels_do_not_change_training_features(panel, fold):
    original = fit_features(panel, fold.train, window=3, embedding_dim=4)
    price, news, embeddings = panel.price.copy(), panel.news.copy(), panel.emb.copy()
    price[35:] += 1000
    news[35:, :, 1:] += 1000
    embeddings[panel.emb_at[:, 0] >= 35] += 1000
    labels = {h: np.ones_like(values) for h, values in panel.labels.items()}
    changed = replace(panel, price=price, news=news, emb=embeddings, labels=labels)
    actual = fit_features(changed, fold.train, window=3, embedding_dim=4)
    np.testing.assert_array_equal(actual.x[:35], original.x[:35])
    np.testing.assert_array_equal(actual.stats["pca"]["components"], original.stats["pca"]["components"])
    expected_mean = np.nanmean(panel.price[:35][panel.in_index[:35]], axis=0)
    np.testing.assert_allclose(actual.stats["price"]["mean"], expected_mean)


def test_news_never_moves_backwards_when_future_rows_are_removed(tables, panel, fold):
    truncated = {name: table[table["date"] <= panel.days[42]] for name, table in tables.items()}
    short_panel = build_panel(**truncated)
    full = fit_features(panel, fold.train, window=3, embedding_dim=4)
    short = fit_features(short_panel, fold.train, window=3, embedding_dim=4)
    np.testing.assert_allclose(full.windows(42, np.arange(6)), short.windows(42, np.arange(6)))


def test_upstream_news_cutoff_is_causal():
    from src.nlp.match_tickers import assign_trading_day

    sessions = pd.to_datetime(["2021-01-08", "2021-01-11", "2021-01-12"])
    published = pd.Series(
        pd.to_datetime(
            [
                "2021-01-08 20:59:00Z",
                "2021-01-08 21:00:00Z",
                "2021-01-08 21:01:00Z",
                "2021-01-09 18:00:00Z",
            ],
            utc=True,
        )
    )
    assigned = assign_trading_day(published, sessions)
    assert assigned.tolist() == [sessions[0], sessions[1], sessions[1], sessions[1]]


def test_production_folds_are_identical_for_both_horizons():
    from src.core.config import walk_forward_config

    settings = {key: value for key, value in walk_forward_config().items() if key != "period"}
    sessions = pd.bdate_range("2019-01-02", periods=635)
    first = walk_forward(sessions, horizon=1, **settings)[0]
    fifth = walk_forward(sessions, horizon=5, **settings)[0]
    pd.testing.assert_index_equal(first.train, fifth.train)
    pd.testing.assert_index_equal(first.test, fifth.test)
    assert len(first.train) == 504 and len(first.test) == 126
    assert sessions.get_loc(first.test[0]) - sessions.get_loc(first.train[-1]) == 6


@pytest.mark.parametrize("horizon", [1, 5])
def test_shared_rows_preserve_historical_membership_and_short_history(panel, horizon):
    early = panel.rows(panel.days[:30], horizon)
    assert "FFF" not in early["ticker"].tolist()
    joined = panel.rows(panel.days[30:31], horizon)
    assert "FFF" in joined["ticker"].tolist()
    assert not panel.rows(panel.days[:1], horizon).empty
    rows = panel.rows(panel.days, horizon)
    assert rows["date"].max() == panel.days[-horizon - 1]


@pytest.mark.parametrize("horizon", [1, 5])
def test_actual_fitting_schema_metrics_backtest_and_no_graph(model, panel, fold, tables, monkeypatch, horizon):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.models.encoder import CompanyEncoder
    from src.models.gat_lstm import GATLSTM

    models, initial = [], []
    original_init = GATLSTM.__init__

    def record(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        models.append(self)
        initial.append(self.encoder.lstm.weight_ih_l0.detach().clone())

    monkeypatch.setattr(GATLSTM, "__init__", record)
    predictions, info = model.run_fold(panel, fold, horizon)
    trained = models[0]
    assert isinstance(trained.encoder, CompanyEncoder)
    assert not trained.has_graph and len(trained.convs) == 0
    assert trained.encoder.lstm.input_size == 14 and trained.head.in_features == 4
    assert not torch.equal(initial[0], trained.encoder.lstm.weight_ih_l0)
    expected = panel.rows(fold.test, horizon)
    pd.testing.assert_frame_equal(predictions[["date", "ticker"]], expected[["date", "ticker"]])
    assert predictions["p_up"].between(0, 1).all()
    assert info["edges_per_day"] == 0 and info["epochs"] == 1
    validate(conform(predictions, "predictions"), "predictions")
    metrics = classification_metrics(predictions["y_true"].to_numpy(), predictions["p_up"].to_numpy())
    assert np.isfinite(metrics["auc"])
    result = backtest(predictions, tables["labels"], horizon)
    assert np.isfinite(result.cumulative_return)
    assert "FFF" not in panel.rows(panel.days[:30], horizon)["ticker"].tolist()

    # Later observations and held-out labels must not change earlier predictions.
    price, news, embeddings = panel.price.copy(), panel.news.copy(), panel.emb.copy()
    price[45:] += 1000
    news[45:, :, 1:] += 1000
    embeddings[panel.emb_at[:, 0] >= 45] += 1000
    labels = {h: values.copy() for h, values in panel.labels.items()}
    for values in labels.values():
        values[35:] = 1 - values[35:]
    changed = replace(panel, price=price, news=news, emb=embeddings, labels=labels)
    repeated, _ = model.run_fold(changed, fold, horizon)
    before_change = predictions["date"] < panel.days[45]
    np.testing.assert_array_equal(predictions.loc[before_change, "p_up"], repeated.loc[before_change, "p_up"])
    for name, weight in trained.state_dict().items():
        torch.testing.assert_close(weight, models[1].state_dict()[name], rtol=0, atol=0)


def test_too_few_training_news_rows_are_an_explicit_error(panel):
    with pytest.raises(ValueError, match="too few"):
        fit_features(panel, panel.days[:2], window=20, embedding_dim=64)
