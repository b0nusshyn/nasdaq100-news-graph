"""Company graph edges. Owner: Modelling. Issues #8, #21, #22, #28.

Output: `edges.parquet` (SCHEMA.md). `sector` edges are static (fold = -1); `comention` and
`corr` weights are computed per walk-forward fold from that fold's training window only, so
this runs once per fold and each run replaces that fold's rows.

    python -m src.graph.build_edges --types sector comention corr --fold 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.schema import (
    ALL_FOLDS,
    COMENTION_MIN_COUNT,
    COMENTION_TOP_K,
    CORR_MIN_OVERLAP,
    CORR_RESIDUAL,
    CORR_TOP_K,
    EDGE_TYPES,
    conform,
)
from src.core.tickers import canonical_ticker, company, same_company
from src.core.todo import todo
from src.eval.splitter import Fold

# Static input, not a SCHEMA.md artifact: committed with the code so every fold and every
# teammate builds the same sector edges.
GICS_CSV = Path(__file__).with_name("gics.csv")
GICS_LEVELS = ("gics_sector", "gics_industry", "gics_sub_industry")


def load_gics(path: Path = GICS_CSV) -> pd.DataFrame:
    """The hand-checked GICS table, one row per company, keyed by current ticker."""
    gics = pd.read_csv(path, dtype="string")
    gics["ticker"] = gics["ticker"].map(canonical_ticker)
    dupes = sorted(set(gics.loc[gics["ticker"].duplicated(), "ticker"]))
    if dupes:
        raise ValueError(f"{path}: companies listed more than once: {dupes}")
    blank = gics.loc[gics[list(GICS_LEVELS)].isna().any(axis=1), "ticker"].tolist()
    if blank:
        raise ValueError(f"{path}: missing GICS classification for {blank}")
    return gics


def check_gics_coverage(gics: pd.DataFrame, membership: pd.DataFrame) -> None:
    """Raise if any ticker that was ever a member has no GICS row (it would drop out of the graph)."""
    missing = sorted(set(membership["ticker"]) - set(gics["ticker"]))
    if missing:
        raise ValueError(f"no GICS classification for former or current members: {missing}")


def sector_edges(gics: pd.DataFrame, level: str = "gics_industry") -> pd.DataFrame:
    """Both directions between every pair sharing a GICS `level`; weight 1, fold -1."""
    tickers = gics[["ticker", level]]
    pairs = tickers.merge(tickers, on=level, suffixes=("_src", "_dst"))
    pairs = pairs[pairs["ticker_src"] != pairs["ticker_dst"]]
    edges = pd.DataFrame({
        "fold": ALL_FOLDS,
        "src": pairs["ticker_src"],
        "dst": pairs["ticker_dst"],
        "edge_type": "sector",
        "weight": 1.0,
    })
    return conform(edges, "edges")


def comention_pairs(
    article_tickers: pd.DataFrame,
    days: pd.DatetimeIndex,
    *,
    max_companies: int | None = None,
    inverse: bool = True,
) -> pd.DataFrame:
    """Co-mention statistics for every unordered pair (`a` < `b`) over the trading `days` given.

    Only articles dated inside `days` count: pass a fold's training window, never more. k is the
    number of distinct *companies* an article names (share classes count once). Each article
    weighs w = 1 / (k - 1), so a roundup naming twenty firms is weak evidence about any one
    pair; with `inverse=False` every article weighs 1. Articles naming more than `max_companies`
    companies are dropped. Two share classes of one company are never a pair.

    Columns: `n_articles` (raw count of articles naming both), `co` (their summed weight),
    `n_a` / `n_b` (summed weight of all articles naming each firm) and
    `jaccard` = co / (n_a + n_b - co), in [0, 1] because co <= min(n_a, n_b).
    """
    at = article_tickers.loc[article_tickers["date"].isin(days), ["article_id", "ticker"]]
    at = at.drop_duplicates().assign(company=lambda d: d["ticker"].map(company))
    k = at.groupby("article_id")["company"].transform("nunique")
    if max_companies is not None:
        at, k = at[k <= max_companies], k[k <= max_companies]
    at = at.assign(w=1.0 / (k - 1).clip(lower=1) if inverse else 1.0)

    n = at.groupby("ticker")["w"].sum()
    multi = at[k > 1]
    pairs = multi.merge(multi[["article_id", "ticker", "company"]], on="article_id", suffixes=("_a", "_b"))
    pairs = pairs[(pairs["ticker_a"] < pairs["ticker_b"]) & (pairs["company_a"] != pairs["company_b"])]
    out = (
        pairs.groupby(["ticker_a", "ticker_b"])["w"]
        .agg(n_articles="size", co="sum")
        .reset_index()
        .rename(columns={"ticker_a": "a", "ticker_b": "b"})
    )
    out["n_a"] = out["a"].map(n).astype(float)
    out["n_b"] = out["b"].map(n).astype(float)
    out["jaccard"] = out["co"] / (out["n_a"] + out["n_b"] - out["co"])
    return out


def top_k_both_ways(pairs: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """Undirected `a`, `b`, `weight` -> directed `src`, `dst`, `weight` rows, both directions.

    Each ticker ranks its peers by `weight` (ties by peer name, so the result is deterministic)
    and keeps the top `k`; a pair picked by either side is kept, in both directions. `k=None`
    keeps every pair. Shared by `comention` and `corr` edges (SCHEMA.md).
    """
    cols = ["src", "dst", "weight"]
    both = pd.concat(
        [pairs[["a", "b", "weight"]].set_axis(cols, axis=1), pairs[["b", "a", "weight"]].set_axis(cols, axis=1)],
        ignore_index=True,
    )
    both = both.sort_values(["src", "weight", "dst"], ascending=[True, False, True], kind="stable")
    if k is not None:
        keep = set(_undirected(both.groupby("src").head(k)))
        both = both[np.array([p in keep for p in _undirected(both)], dtype=bool)]
    return both.sort_values(["src", "dst"]).reset_index(drop=True)


def _undirected(edges: pd.DataFrame) -> list[tuple[str, str]]:
    """(lower, higher) ticker of each `src`, `dst` row, so both directions share one key."""
    src, dst = edges["src"].tolist(), edges["dst"].tolist()
    return [(s, d) if s < d else (d, s) for s, d in zip(src, dst, strict=True)]


def comention_edges(
    article_tickers: pd.DataFrame,
    folds: list[Fold],
    *,
    top_k: int | None = COMENTION_TOP_K,
    min_count: int = COMENTION_MIN_COUNT,
    max_companies: int | None = None,
    inverse: bool = True,
) -> pd.DataFrame:
    """Per fold: weight = co-mention Jaccard over the training window (`comention_pairs`).

    A pair needs `min_count` training-window articles naming both firms to be a candidate; each
    ticker then keeps its `top_k` peers by weight (`top_k_both_ways`). Defaults are SCHEMA.md
    decision 6; the keyword arguments exist for the 3.9 ablations.
    """
    parts = []
    for fold in folds:
        p = comention_pairs(article_tickers, fold.train, max_companies=max_companies, inverse=inverse)
        p = p[p["n_articles"] >= min_count].rename(columns={"jaccard": "weight"})
        parts.append(top_k_both_ways(p, top_k).assign(fold=fold.index, edge_type="comention"))
    columns = ["fold", "src", "dst", "edge_type", "weight"]
    edges = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return conform(edges, "edges")


def market_model(prices: pd.DataFrame, days: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """Daily returns over the trading `days` given, with the equal-weight market removed.

    Pass a fold's training window, never more. Candidates are the tickers `in_index` on at least
    one of `days`; each uses every return it has on those days, in the index or not (all past
    data). Returns are close-to-close on the `prices` calendar, so the first day's return uses
    the previous close. The market m_t is the mean return of the tickers in the index on day t;
    each candidate's beta is the OLS slope of its returns on m over its own observed days.

    Returns (`residual`, `beta`): residual r_i - alpha_i - beta_i * m (dates x tickers, NaN where
    a ticker has no return), and beta per ticker.
    """
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    member = prices.pivot(index="date", columns="ticker", values="in_index").reindex_like(close)
    member = member.fillna(False).astype(bool)
    ret = close.pct_change(fill_method=None)
    ret, member = ret[ret.index.isin(days)], member[member.index.isin(days)]
    market = ret.where(member).mean(axis=1)

    ret = ret.loc[:, member.any()]
    seen = ret.notna()
    m = pd.DataFrame(np.where(seen, market.to_numpy()[:, None], np.nan), ret.index, ret.columns)
    dm, dr = m - m.mean(), ret - ret.mean()
    beta = (dm * dr).sum() / (dm**2).sum()
    alpha = ret.mean() - beta * m.mean()
    residual = ret - alpha - m.mul(beta, axis=1)
    return residual, beta


def corr_pairs(prices: pd.DataFrame, days: pd.DatetimeIndex, *, residual: bool = CORR_RESIDUAL) -> pd.DataFrame:
    """Return correlation for every unordered pair (`a` < `b`) of candidates over the trading `days`.

    Candidates and returns as in `market_model`; pass a fold's training window, never more. With
    `residual` (the default) the weight is the partial correlation given the equal-weight
    market: each stock's fitted market beta is removed before correlating, so two high-beta
    names are not related merely because both move a lot with the market. `residual=False`
    correlates raw returns (ablation). Two share classes of one company are never a pair.

    Columns: `n_obs` (days on which both have a return) and `corr` (Pearson over those days).
    """
    resid, _ = market_model(prices, days)
    if not residual:
        close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
        resid = close.pct_change(fill_method=None).reindex(index=resid.index, columns=resid.columns)
    seen = resid.notna().astype(float)
    n_obs = seen.T @ seen
    corr = resid.corr(min_periods=2)
    tickers = resid.columns.tolist()
    ia, ib = np.triu_indices(len(tickers), k=1)
    out = pd.DataFrame({
        "a": np.array(tickers, dtype=object)[ia],
        "b": np.array(tickers, dtype=object)[ib],
        "n_obs": n_obs.to_numpy()[ia, ib].astype(int),
        "corr": corr.to_numpy()[ia, ib],
    })
    keep = out["corr"].notna() & ~np.array([same_company(a, b) for a, b in zip(out["a"], out["b"], strict=True)])
    return out[keep].reset_index(drop=True)


def corr_edges(
    prices: pd.DataFrame,
    folds: list[Fold],
    *,
    top_k: int | None = CORR_TOP_K,
    min_overlap: int = CORR_MIN_OVERLAP,
    residual: bool = CORR_RESIDUAL,
) -> pd.DataFrame:
    """Per fold: weight = training-window return correlation (`corr_pairs`), stored both ways.

    A pair needs `min_overlap` training days on which both firms have a return to be a
    candidate; each ticker then keeps its `top_k` peers by weight (`top_k_both_ways`). Share
    classes of one company never pair. Defaults are SCHEMA.md decision 7; the keyword arguments
    exist for the 3.9 ablations.
    """
    parts = []
    for fold in folds:
        p = corr_pairs(prices, fold.train, residual=residual)
        p = p[p["n_obs"] >= min_overlap].rename(columns={"corr": "weight"})
        parts.append(top_k_both_ways(p, top_k).assign(fold=fold.index, edge_type="corr"))
    columns = ["fold", "src", "dst", "edge_type", "weight"]
    edges = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return conform(edges, "edges")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--types", nargs="+", choices=EDGE_TYPES, default=list(EDGE_TYPES))
    p.add_argument("--fold", type=int, required=True, help="walk-forward fold index (0-based)")
    p.parse_args()
    todo(28, "build_edges")


if __name__ == "__main__":
    main()
