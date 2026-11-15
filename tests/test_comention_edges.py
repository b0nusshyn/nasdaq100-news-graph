"""Task 3.3 (#21): `comention` edges, one set per walk-forward fold.

Weight = Jaccard of the two firms' article sets over the fold's training window, each article
weighted 1 / (k - 1) for k companies named; each ticker keeps its top-k peers.
"""

import pandas as pd
import pytest

from src.core.schema import COMENTION_MIN_COUNT, COMENTION_TOP_K, validate
from src.core.tickers import company, same_company
from src.eval.splitter import Fold
from src.graph.build_edges import comention_edges, comention_pairs, top_k_both_ways

DAYS = pd.bdate_range("2021-01-04", periods=20)
TRAIN, TEST = DAYS[:10], DAYS[15:]
FOLD = Fold(index=0, train=TRAIN, test=TEST)


def articles(*mentions, day=TRAIN[0]):
    """One article per tuple of tickers, all on `day`."""
    rows = [(f"{day:%Y%m%d}-{i}", day, t) for i, tickers in enumerate(mentions) for t in tickers]
    return pd.DataFrame(rows, columns=["article_id", "date", "ticker"])


def edge_weights(edges: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {(s, d): pytest.approx(w) for s, d, w in zip(edges["src"], edges["dst"], edges["weight"], strict=True)}


def pair(p: pd.DataFrame, a: str, b: str) -> pd.Series:
    return p.set_index(["a", "b"]).loc[(a, b)]


# --------------------------------------------------------------------------- weight


def test_weight_is_jaccard_of_the_two_firms_article_sets():
    # AMD in 3 articles, NVDA in 2, together in 2: 2 / (3 + 2 - 2).
    at = articles(("AMD", "NVDA"), ("AMD", "NVDA"), ("AMD",))
    assert edge_weights(comention_edges(at, [FOLD], min_count=1)) == {
        ("AMD", "NVDA"): pytest.approx(2 / 3),
        ("NVDA", "AMD"): pytest.approx(2 / 3),
    }


def test_heavy_coverage_alone_does_not_make_a_strong_edge():
    # AAPL-MSFT co-occur more often than EA-TTWO, but are each covered far more on their own.
    at = articles(*[("AAPL", "MSFT")] * 4, *[("AAPL",)] * 40, *[("MSFT",)] * 40, *[("EA", "TTWO")] * 3)
    edges = comention_edges(at, [FOLD], min_count=1).set_index(["src", "dst"])["weight"]
    assert edges[("EA", "TTWO")] > edges[("AAPL", "MSFT")]


def test_roundup_article_is_weak_evidence_about_each_pair():
    # A 4-firm roundup counts 1/3 towards each of its pairs and towards each firm's total.
    p = comention_pairs(articles(("AMD", "INTC", "NVDA", "QCOM"), ("AMD", "NVDA")), TRAIN)
    amd_nvda = pair(p, "AMD", "NVDA")
    assert amd_nvda["n_articles"] == 2
    assert amd_nvda["co"] == pytest.approx(1 + 1 / 3)
    assert amd_nvda["n_a"] == pytest.approx(1 + 1 / 3)
    assert pair(p, "INTC", "QCOM")["co"] == pytest.approx(1 / 3)


def test_plain_counting_is_available_for_ablation():
    p = comention_pairs(articles(("AMD", "INTC", "NVDA", "QCOM")), TRAIN, inverse=False)
    assert set(p["co"]) == {1.0}


def test_max_companies_drops_articles_naming_more_companies():
    at = articles(("AMD", "INTC", "NVDA", "QCOM"), ("EA", "TTWO"))
    p = comention_pairs(at, TRAIN, max_companies=3)
    assert set(zip(p["a"], p["b"], strict=True)) == {("EA", "TTWO")}


def test_weights_stay_in_unit_interval():
    at = articles(("A", "B", "C"), ("A", "B"), ("B",), ("C", "D", "E", "F"), ("A", "F"))
    assert comention_pairs(at, TRAIN)["jaccard"].between(0, 1).all()


# --------------------------------------------------------------------------- share classes


def test_share_classes_belong_to_one_company():
    assert company("GOOGL") == company("GOOG")
    assert same_company("FOX", "FOXA") and same_company("LBTYK", "LBTYA")
    assert not same_company("GOOG", "MSFT")
    assert company("MSFT") == "MSFT"


def test_share_classes_of_one_company_are_never_comention_peers():
    # NLP tags every share class together, so GOOG-GOOGL would have Jaccard 1 by construction.
    at = articles(*[("GOOG", "GOOGL")] * 5, ("FOX", "FOXA", "NFLX"))
    edges = comention_edges(at, [FOLD], min_count=1)
    assert not any(same_company(s, d) for s, d in zip(edges["src"], edges["dst"], strict=True))


def test_share_classes_count_as_one_company_in_an_article():
    # "Alphabet and Microsoft" names two companies, not three: full weight, not 1/2.
    p = comention_pairs(articles(("GOOG", "GOOGL", "MSFT")), TRAIN)
    assert pair(p, "GOOG", "MSFT")["co"] == pytest.approx(1.0)
    assert pair(p, "GOOGL", "MSFT")["co"] == pytest.approx(1.0)


def test_max_companies_counts_share_classes_once():
    p = comention_pairs(articles(("GOOG", "GOOGL", "MSFT")), TRAIN, max_companies=2)
    assert len(p) == 2


# --------------------------------------------------------------------------- look-ahead


def test_only_training_window_articles_count():
    at = pd.concat([
        articles(("AMD", "NVDA")),
        articles(("AMD", "NVDA"), ("AMD", "INTC"), day=TEST[0]),  # test-window news
        articles(("AMD", "QCOM"), day=DAYS[12]),  # embargo gap between train and test
    ])
    edges = comention_edges(at, [FOLD], min_count=1)
    assert edge_weights(edges) == {("AMD", "NVDA"): pytest.approx(1.0), ("NVDA", "AMD"): pytest.approx(1.0)}


def test_each_fold_is_built_from_its_own_training_window():
    later = Fold(index=1, train=DAYS[5:15], test=DAYS[15:])
    at = pd.concat([articles(("AMD", "NVDA")), articles(("EA", "TTWO"), day=DAYS[12])])
    edges = comention_edges(at, [FOLD, later], min_count=1)
    by_fold = {f: set(g["src"]) for f, g in edges.groupby("fold")}
    assert by_fold == {0: {"AMD", "NVDA"}, 1: {"EA", "TTWO"}}


# --------------------------------------------------------------------------- sparsity


def test_pairs_below_min_count_are_dropped():
    at = articles(("AMD", "NVDA"), ("AMD", "NVDA"), ("EA", "TTWO"))
    edges = comention_edges(at, [FOLD], min_count=2)
    assert set(edges["src"]) == {"AMD", "NVDA"}


def test_a_pair_picked_from_either_side_is_kept_both_ways():
    pairs = pd.DataFrame({
        "a": ["HUB", "HUB", "HUB", "HUB", "P3", "P4"],
        "b": ["P1", "P2", "P3", "P4", "Q3", "Q4"],
        "weight": [0.4, 0.3, 0.2, 0.1, 0.9, 0.9],
    })
    kept = top_k_both_ways(pairs, k=1)
    # HUB's own pick is P1; P2's only peer is HUB, so P2 picks it; P3 and P4 prefer Q3, Q4.
    assert set(kept.loc[kept["src"] == "HUB", "dst"]) == {"P1", "P2"}
    assert set(kept.loc[kept["dst"] == "HUB", "src"]) == {"P1", "P2"}
    assert dict(zip(zip(kept["src"], kept["dst"], strict=True), kept["weight"], strict=True))[("P2", "HUB")] == 0.3


def test_top_k_bounds_the_edge_count():
    # Fully connected 8 firms (28 pairs): with k=2 at most 8 * 2 undirected edges survive, and
    # every firm keeps its own two strongest peers.
    names = list("ABCDEFGH")
    rows = [(a, b, 10 * i + j) for i, a in enumerate(names) for j, b in enumerate(names) if a < b]
    pairs = pd.DataFrame(rows, columns=["a", "b", "weight"])
    kept = top_k_both_ways(pairs, k=2)
    assert len(kept) // 2 <= len(names) * 2 < len(pairs)
    for name in names:
        mine = pd.concat([pairs.loc[pairs["a"] == name, ["b", "weight"]].set_axis(["peer", "weight"], axis=1),
                          pairs.loc[pairs["b"] == name, ["a", "weight"]].set_axis(["peer", "weight"], axis=1)])
        best = set(mine.nlargest(2, "weight")["peer"])
        assert best <= set(kept.loc[kept["src"] == name, "dst"])


def test_ties_are_broken_deterministically():
    pairs = pd.DataFrame({"a": ["HUB"] * 3, "b": ["X", "Y", "Z"], "weight": [0.5, 0.5, 0.5]})
    first = top_k_both_ways(pairs, k=1)
    assert first.equals(top_k_both_ways(pairs.iloc[::-1].reset_index(drop=True), k=1))


def test_top_k_none_keeps_every_pair_above_min_count():
    at = articles(*[("HUB", f"P{i}") for i in range(15)])
    assert len(comention_edges(at, [FOLD], min_count=1, top_k=None)) == 30


def test_defaults_are_the_schema_decision():
    assert (COMENTION_TOP_K, COMENTION_MIN_COUNT) == (10, 2)


# --------------------------------------------------------------------------- contract


def test_edges_are_symmetric_without_self_loops_and_match_the_contract():
    at = articles(("AMD", "NVDA", "INTC"), ("AMD", "NVDA"), ("GOOG", "GOOGL", "MSFT"), ("AAPL", "MSFT"))
    edges = comention_edges(at, [FOLD], min_count=1)
    validate(edges, "edges")
    assert set(edges["edge_type"]) == {"comention"}
    assert set(edges["fold"]) == {0}
    assert not (edges["src"] == edges["dst"]).any()
    assert edge_weights(edges) == edge_weights(edges.rename(columns={"src": "dst", "dst": "src"}))


def test_no_articles_gives_no_edges():
    edges = comention_edges(articles(("AMD",)), [FOLD])
    validate(edges, "edges")
    assert edges.empty
