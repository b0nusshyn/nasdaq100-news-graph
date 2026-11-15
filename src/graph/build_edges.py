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

from src.core.schema import ALL_FOLDS, COMENTION_MIN_COUNT, COMENTION_TOP_K, CORR_TOP_K, EDGE_TYPES, conform
from src.core.tickers import canonical_ticker, company
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


def corr_edges(prices: pd.DataFrame, folds: list[Fold], top_k: int = CORR_TOP_K) -> pd.DataFrame:
    """Per fold: each ticker's top-k most correlated peers over the training window, stored both ways.

    Exclude share classes of one company (`src.core.tickers.same_company`): their returns are
    near-identical, so they would take each other's top slot. Select with `top_k_both_ways`.
    """
    todo(22, "correlation edges")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--types", nargs="+", choices=EDGE_TYPES, default=list(EDGE_TYPES))
    p.add_argument("--fold", type=int, required=True, help="walk-forward fold index (0-based)")
    p.parse_args()
    todo(28, "build_edges")


if __name__ == "__main__":
    main()
