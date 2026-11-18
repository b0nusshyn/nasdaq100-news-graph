"""Task 3.10 (#42): attention weights and the graph's contribution, per prediction."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tests.test_models import EDGES, SMALL, TYPES, fixture_inputs, graph, model  # noqa: E402, F401

T, F = 20, 7


def x():
    torch.manual_seed(1)
    return torch.randn(5, T, F)


def test_explain_returns_exactly_the_forward_logits_and_attention():
    m, inputs = model(), (x(), *graph())
    logits, attention = m(*inputs)
    out = m.explain(*inputs)
    assert torch.equal(out["logits"], logits) and torch.equal(out["attention"], attention)


def test_graph_off_is_the_dense_layer_on_the_company_state_and_a_zero_graph_signal():
    m, inputs = model(), (x(), *graph())
    out = m.explain(*inputs)
    with torch.no_grad():
        h = m.encoder(inputs[0])
        expected = m.head(torch.cat([h, torch.zeros_like(h)], dim=-1)).squeeze(-1)
    assert torch.allclose(out["logits_no_graph"], expected)
    assert set(out) == {"logits", "logits_no_graph", "attention", *(f"drop_{t}" for t in TYPES)}


def test_with_one_edge_type_dropping_it_is_switching_the_graph_off():
    one = [e for e in EDGES if e[2] == 0]
    m = model(("sector",))
    out = m.explain(x(), *graph(one))
    assert torch.allclose(out["drop_sector"], out["logits_no_graph"])


def test_explain_needs_a_graph():
    with pytest.raises(ValueError, match="needs a model with a graph"):
        empty = (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0))
        model(()).explain(x(), *empty)


@pytest.fixture(scope="module")
def explained(fixture_inputs):  # noqa: F811
    from src.models.attention import explain_days
    from src.models.train import train_fold

    panel, edges, folds = fixture_inputs
    preds, _, fitted = train_fold(panel, edges, folds[0], 1, SMALL, log=lambda *_: None, return_model=True)
    contrib, edge_rows = explain_days(fitted, panel, fitted.test_idx)
    return panel, fitted, preds, contrib, edge_rows


def test_explained_predictions_are_the_trained_predictions_bit_for_bit(explained):
    _, _, preds, contrib, _ = explained
    assert contrib[["date", "ticker"]].reset_index(drop=True).equals(preds[["date", "ticker"]])
    assert (contrib["p_up"].to_numpy() == preds["p_up"].to_numpy()).all()
    assert (contrib["y_true"].to_numpy() == preds["y_true"].to_numpy()).all()
    assert np.allclose(contrib["graph_logit"], contrib["logit"] - contrib["logit_no_graph"])
    assert np.allclose(contrib.groupby("date")["graph_logit_within_day"].mean(), 0, atol=1e-5)
    assert np.allclose(contrib["graph_logit"] - contrib["graph_logit_within_day"],
                       contrib.groupby("date")["graph_logit"].transform("mean"), atol=1e-5)


def test_each_predictions_attention_sums_to_one_per_edge_type_and_head(explained):
    _, fitted, _, contrib, e = explained
    heads = [f"alpha_{k}" for k in range(fitted.model.heads)]
    sums = e.groupby(["date", "dst", "edge_type"])[heads].sum()
    assert np.allclose(sums.to_numpy(), 1, atol=1e-5)
    deg = e.groupby(["date", "dst", "edge_type"]).size().unstack(fill_value=0)
    merged = contrib.set_index(["date", "ticker"])
    for t in deg.columns:
        assert (merged.loc[deg.index, f"in_degree_{t}"].to_numpy() == deg[t].to_numpy()).all()
    predicted = set(zip(contrib["date"], contrib["ticker"], strict=True))
    assert all(k in predicted for k in zip(e["date"], e["dst"], strict=True))


def test_source_news_days_count_the_sources_news_over_the_input_window(explained):
    panel, fitted, _, _, e = explained
    w = fitted.features.window
    sample = e.sample(min(len(e), 200), random_state=0)
    d, k = panel.day_index(sample["date"]), panel.tickers.get_indexer(sample["src"])
    expected = [int((panel.news[max(di - w + 1, 0): di + 1, ki, 0] > 0).sum()) for di, ki in zip(d, k, strict=True)]
    assert sample["src_news_days"].tolist() == expected
    assert (sample["src_has_news"].to_numpy() == (panel.news[d, k, 0] > 0)).all()


def test_concentration_is_one_for_uniform_attention_zero_for_one_hot_and_skips_single_edges():
    from src.models.attention import node_attention

    rows = [("A", "sector", [0.5, 0.5]), ("A", "sector", [0.5, 0.5]),
            ("B", "corr", [1.0, 1.0]), ("B", "corr", [0.0, 0.0]), ("B", "corr", [0.0, 0.0]),
            ("C", "sector", [1.0, 1.0])]
    e = pd.DataFrame({"horizon": 1, "date": pd.Timestamp("2021-06-01"), "dst": [r[0] for r in rows],
                      "edge_type": [r[1] for r in rows], "alpha_0": [r[2][0] for r in rows],
                      "alpha_1": [r[2][1] for r in rows]})
    e["alpha_mean"] = (e["alpha_0"] + e["alpha_1"]) / 2
    n = node_attention(e, heads=2).set_index("ticker")
    assert set(n.index) == {"A", "B"}
    assert n.loc["A", "entropy"] == pytest.approx(1) and n.loc["A", "top_share"] == pytest.approx(0.5)
    assert n.loc["B", "entropy"] == pytest.approx(0, abs=1e-9) and n.loc["B", "uniform_share"] == pytest.approx(1 / 3)


def test_candidates_are_the_largest_within_day_contributions_with_the_most_attended_edge_of_the_type_that_carried_it():
    from src.models.attention import top_contributions

    day = pd.Timestamp("2021-06-01")
    contrib = pd.DataFrame({"horizon": 1, "date": day, "ticker": ["A", "B", "C"],
                            "graph_logit": [0.9, -0.5, 0.2], "graph_logit_within_day": [0.01, -0.5, 0.2],
                            "graph_logit_sector": [0.01, -0.1, 0.2], "graph_logit_corr": [0.0, -0.45, 0.0]})
    edges = pd.DataFrame({"horizon": 1, "date": day, "src": ["X", "Y", "Z"], "dst": ["B", "B", "B"],
                          "edge_type": ["corr", "corr", "sector"], "weight": [0.3, 0.6, 1.0],
                          "alpha_mean": [0.2, 0.8, 1.0], "src_has_news": [True, False, True],
                          "src_news_days": [3, 0, 1]})
    top = top_contributions(contrib, lambda _: edges, ("sector", "corr"), n=2)
    assert top["ticker"].tolist() == ["B", "C"]
    b = top.iloc[0]
    assert (b["via"], b["top_src"], b["top_alpha"], b["top_weight"]) == ("corr", "Y", 0.8, 0.6)
