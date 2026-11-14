"""Task 3.2 (#9): model skeleton, LSTM encoder -> per-edge-type GAT -> dense -> P(up).

CI installs only requirements-dev.txt (no torch), so these run locally or on Colab.
"""

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


def fixture_tables():
    from src.core.io import read_table
    from src.core.paths import FIXTURES_DIR

    return {n: read_table(n, FIXTURES_DIR / f"{n}.parquet") for n in ("prices", "news_features", "edges")}


def test_fixture_day_runs_end_to_end_for_in_index_companies_only():
    from src.models.snapshot import build_snapshot

    tables = fixture_tables()
    dates = sorted(tables["prices"]["date"].unique())
    before_join, after_join = dates[25], dates[-1]  # FFF joins the index at day 30

    early = build_snapshot(before_join, **tables, fold=0, window=T, edge_types=TYPES)
    assert early.tickers == ["AAA", "BBB", "CCC", "DDD", "EEE"]

    snap = build_snapshot(after_join, **tables, fold=0, window=T, edge_types=TYPES)
    assert snap.tickers == ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    n = len(snap.tickers)
    assert snap.x.shape == (n, T, 4)  # trailing return, has_news, sentiment, confidence
    assert int(snap.edge_index.max()) < n

    from src.models.gat_lstm import GATLSTM

    m = GATLSTM(n_features=4, hidden=H, heads=HEADS, edge_types=TYPES).eval()
    logits, attention = m(snap.x, snap.edge_index, snap.edge_type, snap.edge_weight)
    p_up = torch.sigmoid(logits)
    assert p_up.shape == (n,) and ((p_up >= 0) & (p_up <= 1)).all()
    assert attention.shape == (snap.edge_index.shape[1], HEADS)


def test_edge_of_a_type_the_model_lacks_is_an_error():
    # An ablation model without corr must not silently ignore corr edges handed to it.
    m = model(edge_types=("sector", "comention"))
    with pytest.raises(ValueError, match="edge type"):
        m(torch.randn(N, T, F), *graph())
