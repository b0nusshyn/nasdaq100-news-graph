"""Task 3.4 (#22): `corr` edges, one set per walk-forward fold.

Weight = partial correlation of two firms' daily returns given the equal-weight market, over the
fold's training window: each stock's market beta is fitted on those days and removed before
correlating. Each ticker keeps its top-k peers.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.schema import CORR_MIN_OVERLAP, CORR_RESIDUAL, CORR_TOP_K, validate
from src.core.tickers import same_company
from src.eval.splitter import Fold
from src.graph.build_edges import corr_edges, corr_pairs, market_model

DAYS = pd.bdate_range("2021-01-04", periods=400)
TRAIN, TEST = DAYS[1:301], DAYS[306:]
FOLD = Fold(index=0, train=TRAIN, test=TEST)


def panel(returns: pd.DataFrame, in_index: pd.DataFrame | bool = True) -> pd.DataFrame:
    """`prices` rows (date, ticker, close, in_index) whose close-to-close returns are `returns`.

    The first row of `returns` is the day before the window: its value is ignored and its close
    anchors the series.
    """
    close = 100 * (1 + returns.fillna(0)).cumprod()
    close = close.where(returns.notna() | (returns.index == returns.index[0])[:, None])
    flags = in_index if isinstance(in_index, pd.DataFrame) else pd.DataFrame(in_index, returns.index, returns.columns)
    long = close.stack().rename("close").rename_axis(["date", "ticker"]).reset_index()
    long["in_index"] = [bool(flags.at[d, t]) for d, t in zip(long["date"], long["ticker"], strict=True)]
    return long


def pair(p: pd.DataFrame, a: str, b: str) -> pd.Series:
    return p.set_index(["a", "b"]).loc[(a, b)]


def factor_returns(betas: dict[str, float], days=DAYS, seed=0, noise=0.01) -> pd.DataFrame:
    """r_i = beta_i * f + e_i with one common factor f (daily vol 1%) and independent noise."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, len(days))
    cols = {t: b * f + rng.normal(0, noise, len(days)) for t, b in betas.items()}
    return pd.DataFrame(cols, index=days)


# --------------------------------------------------------------------------- market model


def test_fitted_beta_recovers_a_stocks_market_sensitivity():
    # 60 stocks with betas averaging 1, so the equal-weight market tracks the factor; one of them
    # has beta 1.5 and one 0.4. Noise 0.3% over 399 days puts the sampling error of beta near 0.015.
    betas = {f"S{i:02d}": b for i, b in enumerate(np.linspace(0.5, 1.5, 58))} | {"HIGH": 1.5, "LOW": 0.4}
    _, beta = market_model(panel(factor_returns(betas, noise=0.003)), DAYS[1:])
    assert beta["HIGH"] == pytest.approx(1.5, abs=0.05)
    assert beta["LOW"] == pytest.approx(0.4, abs=0.05)


def industry_panel(seed=0) -> pd.DataFrame:
    """40 stocks on one market factor. A and B (beta 1) also share an industry shock as large as
    their own noise, so their partial correlation given the market is 1/2. H1 and H2 (beta 1.7)
    share nothing but the market, so theirs is 0."""
    rng = np.random.default_rng(seed + 1000)  # not `seed`: the shock must be independent of the factor
    r = factor_returns({f"S{i:02d}": 1.0 for i in range(36)} | {"A": 1.0, "B": 1.0, "H1": 1.7, "H2": 1.7},
                       seed=seed, noise=0.004)
    shock = rng.normal(0, 0.004, len(DAYS))
    r["A"] += shock
    r["B"] += shock
    return r


# --------------------------------------------------------------------------- weight


def test_weight_is_partial_correlation_given_the_market():
    p = corr_pairs(panel(industry_panel()), DAYS[1:])
    assert pair(p, "A", "B")["corr"] == pytest.approx(0.5, abs=0.08)
    assert pair(p, "H1", "H2")["corr"] == pytest.approx(0.0, abs=0.1)


def test_raw_returns_rank_pairs_by_shared_beta_instead():
    # Kept for the ablation: on raw returns the two high-beta names look more related than A-B.
    p = corr_pairs(panel(industry_panel()), DAYS[1:], residual=False)
    assert pair(p, "H1", "H2")["corr"] > pair(p, "A", "B")["corr"] > 0.8


# --------------------------------------------------------------------------- contract


def test_each_ticker_keeps_its_top_peer_both_ways_and_matches_the_contract():
    edges = corr_edges(panel(industry_panel()), [FOLD], top_k=1)
    validate(edges, "edges")
    assert set(edges["edge_type"]) == {"corr"} and set(edges["fold"]) == {0}
    assert not (edges["src"] == edges["dst"]).any()
    kept = set(zip(edges["src"], edges["dst"], strict=True))
    assert {("A", "B"), ("B", "A")} <= kept
    assert all((d, s) in kept for s, d in kept)
    assert edges.set_index(["src", "dst"]).loc[("A", "B"), "weight"] == pytest.approx(0.5, abs=0.1)


def test_no_folds_gives_no_edges():
    edges = corr_edges(panel(industry_panel()), [])
    validate(edges, "edges")
    assert edges.empty


def test_defaults_are_the_schema_decision():
    assert (CORR_TOP_K, CORR_MIN_OVERLAP, CORR_RESIDUAL) == (10, 126, True)


# --------------------------------------------------------------------------- look-ahead


def top_peer(edges: pd.DataFrame, ticker: str, fold: int = 0) -> str:
    mine = edges[(edges["fold"] == fold) & (edges["src"] == ticker)]
    return mine.sort_values("weight", ascending=False)["dst"].iloc[0]


def test_only_training_window_returns_count():
    # After training ends (embargo and test window) A moves with C instead of B.
    r = industry_panel().rename(columns={"S35": "C"})
    late = r.index > TRAIN[-1]
    shock = np.random.default_rng(7).normal(0, 0.02, late.sum())
    r.loc[late, "A"] += shock
    r.loc[late, "C"] += shock
    edges = corr_edges(panel(r), [FOLD], top_k=1)
    assert top_peer(edges, "A") == "B"


def test_each_fold_is_built_from_its_own_training_window():
    # A pairs with B in the first 300 days and with C (a renamed S35) afterwards.
    r = industry_panel().rename(columns={"S35": "C"})
    later = DAYS > TRAIN[-1]
    r.loc[later, "B"] = r.loc[later, "S00"]
    shock = np.random.default_rng(7).normal(0, 0.004, later.sum())
    r.loc[later, "A"] += shock
    r.loc[later, "C"] += shock
    second = Fold(index=1, train=DAYS[later][:-5], test=DAYS[later][-5:])
    edges = corr_edges(panel(r), [FOLD, second], top_k=1, min_overlap=60)
    assert (top_peer(edges, "A", 0), top_peer(edges, "A", 1)) == ("B", "C")


# --------------------------------------------------------------------------- candidates


def membership(r: pd.DataFrame, **out: pd.Series) -> pd.DataFrame:
    """in_index flags: every ticker in the index every day, except the masks given (True = in)."""
    flags = pd.DataFrame(True, r.index, r.columns)
    for t, mask in out.items():
        flags[t] = mask
    return flags


def test_a_ticker_never_in_the_index_during_training_gets_no_edges():
    r = industry_panel()
    joins_in_test = pd.Series(r.index >= TEST[0], index=r.index)
    edges = corr_edges(panel(r, membership(r, B=joins_in_test)), [FOLD])
    assert "B" not in set(edges["src"]) | set(edges["dst"])


def test_a_late_joiner_uses_its_returns_from_before_it_joined():
    # B joins the index on the last training day; all 300 training-day returns still count.
    r = industry_panel()
    joins_late = pd.Series(r.index >= TRAIN[-1], index=r.index)
    p = corr_pairs(panel(r, membership(r, B=joins_late)), TRAIN)
    assert pair(p, "A", "B")["n_obs"] == len(TRAIN)
    assert top_peer(corr_edges(panel(r, membership(r, B=joins_late)), [FOLD], top_k=1), "A") == "B"


def test_pairs_with_too_few_shared_returns_are_dropped():
    # B's first close is 101 training days from the end, so it has 100 returns: below 126.
    r = industry_panel()
    r.loc[r.index < TRAIN[-101], "B"] = np.nan
    assert pair(corr_pairs(panel(r), TRAIN), "A", "B")["n_obs"] == 100
    assert "B" not in set(corr_edges(panel(r), [FOLD])["src"])
    assert "B" in set(corr_edges(panel(r), [FOLD], min_overlap=60)["src"])


# --------------------------------------------------------------------------- share classes


def test_share_classes_of_one_company_are_never_corr_peers():
    # GOOG and GOOGL trade almost identically, so each would be the other's top peer.
    r = industry_panel()
    r["GOOG"] = r["S00"]
    r["GOOGL"] = r["S00"] + np.random.default_rng(3).normal(0, 1e-4, len(r))
    edges = corr_edges(panel(r), [FOLD])
    assert not any(same_company(s, d) for s, d in zip(edges["src"], edges["dst"], strict=True))


# --------------------------------------------------------------------------- report

# The committed fixtures cover 60 trading days, too short for configs/walk_forward.yaml.
SHORT_FOLDS = {"period": {"start": "2021-01-01", "end": "2021-12-31"}, "train_days": 30, "test_days": 10,
               "step_days": 10, "expanding": False, "embargo_days": 5}


def test_report_builds_on_the_committed_fixtures(monkeypatch):
    from src.core.io import read_table
    from src.core.paths import DATA_DIR_ENV, FIXTURES_DIR
    from src.graph.corr_report import build

    monkeypatch.setenv(DATA_DIR_ENV, str(FIXTURES_DIR))
    report, top = build(read_table("prices", columns=["date", "ticker", "close", "in_index"]), SHORT_FOLDS,
                        min_overlap=20)
    for section in ("## 1 ·", "## 2 ·", "## 3 ·", "## 4 ·", "## 5 ·"):
        assert section in report
    assert not top.empty and list(top.columns) == ["fold", "src", "dst", "weight"]
