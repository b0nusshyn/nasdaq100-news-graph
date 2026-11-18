"""Task 3.2 (#9): model skeleton, LSTM encoder -> per-edge-type GAT -> dense -> P(up).

CI installs only requirements-dev.txt (no torch), so these run locally or on Colab.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.encoder import CompanyEncoder  # noqa: E402

N, T, F, H = 5, 20, 7, 16


TYPES = ("sector", "comention", "corr")
HEADS = 2

# Directed edges as (src, dst, edge type index, weight). Company 4 has no edges at all.
EDGES = [(0, 1, 0, 1.0), (1, 0, 0, 1.0), (0, 2, 1, 3.0), (2, 3, 2, -0.4), (3, 2, 2, -0.4)]


def graph(edges=EDGES):
    src, dst, etype, w = zip(*edges, strict=True)
    return (torch.tensor([src, dst]), torch.tensor(etype), torch.tensor(w, dtype=torch.float32))


def model(edge_types=TYPES):
    from src.models.gat_lstm import GATLSTM

    torch.manual_seed(0)
    return GATLSTM(n_features=F, hidden=H, heads=HEADS, edge_types=edge_types).eval()


def test_encoder_summarises_each_company_window():
    enc = CompanyEncoder(n_features=F, hidden=H)
    assert enc(torch.randn(N, T, F)).shape == (N, H)


def test_one_logit_per_company_and_one_attention_row_per_edge():
    logits, attention = model()(torch.randn(N, T, F), *graph())
    assert logits.shape == (N,)
    assert attention.shape == (len(EDGES), HEADS)


def test_attention_rows_follow_input_edge_order():
    # Company 1 receives three sector edges and one comention edge, interleaved with the rest.
    edges = [(0, 1, 0, 1.0), (2, 3, 2, 0.5), (2, 1, 0, 1.0), (4, 1, 1, 2.0), (3, 1, 0, 1.0), (3, 2, 2, 0.5)]
    _, attention = model()(torch.randn(N, T, F), *graph(edges))
    sector_into_1 = attention[[0, 2, 4]]
    assert torch.allclose(sector_into_1.sum(0), torch.ones(HEADS))
    assert not torch.allclose(sector_into_1[0], sector_into_1[1])  # real weights, not uniform
    assert torch.allclose(attention[3], torch.ones(HEADS))  # sole comention edge into 1


def test_a_shock_reaches_neighbours_along_edges_and_nowhere_else():
    m = model()
    x = torch.randn(N, T, F)
    shocked = x.clone()
    shocked[0] += 3.0  # company 0 has edges to 1 (sector) and 2 (comention); 4 is isolated
    with torch.no_grad():
        before, _ = m(x, *graph())
        after, _ = m(shocked, *graph())
    # A NaN is still "a value": the isolated company must get a real prediction.
    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    moved = ~torch.isclose(before, after)
    assert moved.tolist() == [True, True, True, False, False]
    # 3 is two hops away (via 2) and the model has one GAT layer, so it does not move either.


# --------------------------------------------------------------------------- training (#29)

SMALL = {
    "seed": 0, "horizons": [1, 5], "window_days": 5,
    "features": {"price": ["ret_1d", "volatility_20d", "rsi_14", "macd", "macd_signal", "volume_z_20d"],
                 "news": ["has_news", "sentiment", "confidence"], "embeddings": True, "embedding_dim": 4},
    "encoder": {"hidden": 8, "layers": 1, "dropout": 0.0},
    "gat": {"heads": 2, "edge_types": list(TYPES)},
    "train": {"epochs": 3, "batch_days": 8, "lr": 0.01, "weight_decay": 0.0, "grad_clip": 1.0,
              "validation_fraction": 0.1, "early_stopping_patience": 2},
}
SHORT_FOLDS = {"period": {"start": "2021-01-01", "end": "2021-12-31"}, "train_days": 30, "test_days": 10,
               "step_days": 10, "expanding": False, "embargo_days": 5}


@pytest.fixture(scope="module")
def fixture_inputs():
    from src.core.io import read_table
    from src.core.paths import FIXTURES_DIR
    from src.models.panel import build_panel
    from src.models.train import folds_for

    t = {n: read_table(n, FIXTURES_DIR / f"{n}.parquet") for n in ("prices", "labels", "news_features", "edges")}
    panel = build_panel(t["prices"], t["labels"], t["news_features"])
    return panel, t["edges"], folds_for(panel, SHORT_FOLDS)


def test_validation_days_are_embargoed_from_the_days_the_model_fits_on():
    import pandas as pd

    from src.models.panel import split_validation

    days = pd.bdate_range("2021-01-04", periods=100)
    fit, val = split_validation(days, 0.1, 5)
    assert len(val) == 10 and val[-1] == days[-1]
    assert days.get_loc(val[0]) - days.get_loc(fit[-1]) == 6  # five unused days in between


def test_ablation_runs_are_named_after_what_they_drop():
    from src.models.train import model_name

    assert model_name() == "gat_lstm"
    assert model_name(["news", "corr"]) == "gat_lstm_no_corr_no_news"


def test_a_batch_has_the_in_index_companies_of_each_day_and_only_their_edges(fixture_inputs):
    import numpy as np

    from src.models.panel import fit_features
    from src.models.train import fold_edges, make_batch

    panel, edges, folds = fixture_inputs
    features = fit_features(panel, folds[0].train, window=5, embedding_dim=4)
    days = np.array([25, 45])  # FFF joins at day 30
    b = make_batch(days, panel, features, fold_edges(edges, panel, 0, TYPES), horizon=1)
    assert b["x"].shape == (5 + 6, 5, len(features.names))
    assert b["at"][:5, 1].tolist() == [0, 1, 2, 3, 4] and b["at"][5:, 1].tolist() == [0, 1, 2, 3, 4, 5]
    src, dst = b["edge_index"]
    assert ((src < 5) == (dst < 5)).all()  # no edge crosses from one day's graph to another's


def test_a_fold_trains_and_predicts_exactly_the_shared_row_set(fixture_inputs, tmp_path, monkeypatch):
    from src.core.io import read_predictions, write_predictions
    from src.core.paths import DATA_DIR_ENV
    from src.models.train import train_fold

    panel, edges, folds = fixture_inputs
    for h in (1, 5):
        preds, info = train_fold(panel, edges, folds[0], h, SMALL, log=lambda *_: None)
        expected = panel.rows(folds[0].test, h)
        assert preds[["date", "ticker"]].equals(expected[["date", "ticker"]])
        assert (preds["y_true"].to_numpy() == expected["y"].to_numpy()).all()
        assert preds["p_up"].between(0, 1).all() and (preds["fold"] == 0).all() and (preds["horizon"] == h).all()
        assert 1 <= info["epochs"] <= 3
        assert 0 <= info["threshold"] <= 1 and 0 <= info["val_base_rate"] <= 1
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    write_predictions(preds, "gat_lstm")  # validates against SCHEMA.md
    assert len(read_predictions("gat_lstm")) == len(preds)


def test_early_stopping_checks_within_day_auc_several_times_per_epoch_and_logs_the_loss_optimum(fixture_inputs):
    # 3.12 (#58): stop on the headline metric (validation within-day AUC), validate more often
    # than once per epoch, and log where the validation-loss optimum was, to measure divergence.
    from src.models.train import train_fold

    panel, edges, folds = fixture_inputs
    cfg = {**SMALL, "train": {**SMALL["train"], "validate_every_steps": 1}}
    _, info = train_fold(panel, edges, folds[0], 1, cfg, log=lambda *_: None)
    assert info["stop_metric"] == "within_day_auc"
    assert info["checks"] == info["steps"] > info["epochs"]  # one check per optimiser step
    assert 1 <= info["best_step"] <= info["steps"] and 1 <= info["loss_optimal_step"] <= info["steps"]
    for k in ("best_val_within_day_auc", "val_within_day_auc_at_loss_optimum", "val_loss_at_best", "val_mcc"):
        assert k in info
    assert info["best_val_within_day_auc"] >= info["val_within_day_auc_at_loss_optimum"]


def test_tuning_can_skip_the_test_window_entirely(fixture_inputs):
    from src.models.train import train_fold

    panel, edges, folds = fixture_inputs
    preds, info = train_fold(panel, edges, folds[0], 1, SMALL, predict_test=False, log=lambda *_: None)
    assert preds is None and "test_rows" not in info


def test_ablations_drop_news_features_and_edge_types(fixture_inputs):
    from src.models.train import train_fold

    panel, edges, folds = fixture_inputs
    _, full = train_fold(panel, edges, folds[0], 1, SMALL, log=lambda *_: None)
    preds, ablated = train_fold(panel, edges, folds[0], 1, SMALL, ablate=("news", "corr"), log=lambda *_: None)
    assert ablated["features"] == 7 and full["features"] == 7 + 3 + 4  # price + flag; news scalars; PCA
    assert ablated["edges_per_day"] < full["edges_per_day"]
    assert (preds["model"] == "gat_lstm_no_corr_no_news").all()


def test_a_fold_without_edges_of_a_type_is_an_error(fixture_inputs):
    from src.models.train import fold_edges

    panel, edges, _ = fixture_inputs
    with pytest.raises(ValueError, match="rebuild it"):
        fold_edges(edges, panel, 1, TYPES)  # the fixture has comention/corr rows for fold 0 only


def test_edge_of_a_type_the_model_lacks_is_an_error():
    # An ablation model without corr must not silently ignore corr edges handed to it.
    m = model(edge_types=("sector", "comention"))
    with pytest.raises(ValueError, match="edge type"):
        m(torch.randn(N, T, F), *graph())


def test_fold_report_leads_with_within_day_auc_and_reports_the_base_rate_beside_accuracy():
    import pandas as pd

    from src.models.train import fold_report

    # Every probability is above 0.5 but the ranking is perfect: AUC 1, and a threshold chosen
    # inside the range of predictions gives a real MCC instead of an all-up NaN. One day, so the
    # within-day AUC (the headline since 18 Nov) is the same ranking.
    preds = pd.DataFrame({"date": pd.Timestamp("2021-01-04"), "y_true": [0, 0, 1, 1, 1],
                          "p_up": [0.55, 0.56, 0.60, 0.61, 0.62]})
    at_half = fold_report(preds, {"threshold": 0.5})
    assert at_half["share_predicted_up"] == 1 and at_half["accuracy"] == at_half["base_rate"] == 0.6
    chosen = fold_report(preds, {"threshold": 0.60})
    assert list(chosen)[:2] == ["within_day_auc", "auc"] and chosen["within_day_auc"] == chosen["auc"] == 1.0
    assert chosen["mcc"] == 1.0 and chosen["accuracy"] == 1.0


def test_run_record_stores_each_fold_threshold_beside_the_predictions(tmp_path):
    import json

    from src.models.train import write_run_record

    out = write_run_record(tmp_path / "gat_lstm_no_news.parquet", {"seed": 0}, ["news"],
                           [{"fold": 0, "horizon": 1, "threshold": 0.53}])
    record = json.loads(out.read_text())
    assert out.name == "gat_lstm_no_news_run.json" and record["model"] == "gat_lstm_no_news"
    assert record["folds"][0]["threshold"] == 0.53 and "git_commit" in record


# --------------------------------------------------------------------------- Baseline 6 = minus the GAT (#35)


def test_without_edge_types_the_model_is_dense_over_the_encoder():
    from src.models.gat_lstm import GATLSTM

    m = GATLSTM(n_features=F, hidden=H, heads=HEADS, edge_types=()).eval()
    assert not m.has_graph and len(m.convs) == 0 and m.head.in_features == H
    x = torch.randn(N, T, F)
    no_edges = (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0))
    logits, attention = m(x, *no_edges)
    with torch.no_grad():
        assert torch.allclose(logits, m.head(m.encoder(x)).squeeze(-1))
    assert attention.shape == (0, HEADS)


def test_one_seed_gives_both_models_the_same_initial_encoder():
    from src.models.gat_lstm import GATLSTM

    torch.manual_seed(7)
    with_graph = GATLSTM(n_features=F, hidden=H, heads=HEADS, edge_types=TYPES)
    torch.manual_seed(7)
    without = GATLSTM(n_features=F, hidden=H, heads=HEADS, edge_types=())
    for (name, a), (_, b) in zip(with_graph.encoder.state_dict().items(), without.encoder.state_dict().items(),
                                 strict=True):
        assert torch.equal(a, b), name


def test_baseline_6_is_the_proposed_model_with_every_edge_type_ablated(fixture_inputs):
    # Same settings: Baseline 6's config and the all-edges ablation are one experiment, so the
    # ablation table and the baseline table can cite one run (before 4.13 tunes them apart).
    from src.models.train import train_fold

    panel, edges, folds = fixture_inputs
    b6 = SMALL | {"model": "lstm_news", "gat": {**SMALL["gat"], "edge_types": []}}
    a, _ = train_fold(panel, None, folds[0], 1, b6, log=lambda *_: None)
    b, _ = train_fold(panel, edges, folds[0], 1, SMALL, ablate=TYPES, log=lambda *_: None)
    assert (a["model"] == "lstm_news").all()
    assert a[["date", "ticker", "y_true"]].equals(b[["date", "ticker", "y_true"]])
    assert np.array_equal(a["p_up"].to_numpy(), b["p_up"].to_numpy())


def configs():
    from src.core.config import load_config
    from src.core.paths import CONFIGS_DIR

    return {n: load_config(CONFIGS_DIR / f"{n}.yaml") for n in ("gat_lstm", "lstm_news", "lstm_price")}


def test_baseline_configs_share_the_proposed_models_inputs_and_validation():
    # 4.13 may tune lr, hidden, dropout apart; the inputs and the split may never drift.
    c = configs()
    for name in ("lstm_news", "lstm_price"):
        assert c[name]["model"] == name and c[name]["gat"]["edge_types"] == []
        for key in ("horizons", "window_days"):
            assert c[name][key] == c["gat_lstm"][key], (name, key)
        assert c[name]["features"]["price"] == c["gat_lstm"]["features"]["price"], name
        assert c[name]["train"]["validation_fraction"] == c["gat_lstm"]["train"]["validation_fraction"], name
        assert set(c[name]) == set(c["gat_lstm"]) and set(c[name]["train"]) == set(c["gat_lstm"]["train"]), name
    assert c["lstm_news"]["features"] == c["gat_lstm"]["features"]
    assert c["lstm_price"]["features"]["news"] == [] and c["lstm_price"]["features"]["embeddings"] is False


@pytest.mark.parametrize("cls_name, news", [("LSTMNews", True), ("LSTMPrice", False)])
def test_lstm_baselines_run_through_the_proposed_models_loop(fixture_inputs, tmp_path, cls_name, news):
    import yaml

    from src.models import baselines

    panel, _, folds = fixture_inputs
    cls = getattr(baselines, cls_name)
    feat = SMALL["features"] if news else {**SMALL["features"], "news": [], "embeddings": False}
    cfg = SMALL | {"model": cls.name, "gat": {**SMALL["gat"], "edge_types": []}, "features": feat}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    preds, info = cls(path).run_fold(panel, folds[0], 5)
    assert preds[["date", "ticker"]].equals(panel.rows(folds[0].test, 5)[["date", "ticker"]])
    assert (preds["model"] == cls.name).all() and 0 <= info["threshold"] <= 1
    assert info["features"] == (14 if news else 7)


def test_an_lstm_baseline_refuses_a_config_with_a_graph(tmp_path):
    import yaml

    from src.models.baselines.lstm_news import LSTMNews

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(SMALL | {"model": "lstm_news"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a lstm_news config"):
        LSTMNews(path)


def test_deal_arb_ablation_drops_only_the_named_corr_pairs_in_both_directions(fixture_inputs):
    # 3.9 (#41): the deal-arbitrage pairs lose their corr edges; their sector and comention edges stay.
    from src.models.train import fold_edges

    panel, edges, _ = fixture_inputs
    full = fold_edges(edges, panel, 0, TYPES)
    cut = fold_edges(edges, panel, 0, TYPES, drop_corr_pairs=(("BBB", "AAA"),))
    a, b, corr = panel.tickers.get_loc("AAA"), panel.tickers.get_loc("BBB"), TYPES.index("corr")

    def pair_edges(g, t):
        either_way = ((g["src"] == a) & (g["dst"] == b)) | ((g["src"] == b) & (g["dst"] == a))
        return int((either_way & (g["type"] == t)).sum())

    assert pair_edges(full, corr) > 0 and pair_edges(cut, corr) == 0
    for t in (TYPES.index("sector"), TYPES.index("comention")):
        assert pair_edges(cut, t) == pair_edges(full, t)
    assert len(full["src"]) - len(cut["src"]) == pair_edges(full, corr)


def test_deal_arb_is_an_ablation_name_and_its_pairs_are_the_three_from_3_4():
    from src.models.train import DEAL_ARB, DEAL_ARB_PAIRS, model_name

    assert model_name([DEAL_ARB]) == "gat_lstm_no_deal_arb"
    assert set(map(frozenset, DEAL_ARB_PAIRS)) == {frozenset(p) for p in (("ADI", "MXIM"), ("AMD", "XLNX"),
                                                                          ("AZN", "ALXN"))}
