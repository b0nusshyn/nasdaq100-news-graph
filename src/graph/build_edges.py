"""Company graph edges. Owner: Modelling. Issues #8, #21, #22, #28.

Output: `edges.parquet` (SCHEMA.md). `sector` edges are static (fold = -1); `comention` and
`corr` weights are computed per walk-forward fold from that fold's training window only, so
this runs once per fold and each run replaces that fold's rows.

    python -m src.graph.build_edges --types sector comention corr --fold 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.core.schema import ALL_FOLDS, CORR_TOP_K, EDGE_TYPES, conform
from src.core.tickers import canonical_ticker
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


def comention_edges(article_tickers: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Per fold: weight = number of training-window articles naming both firms."""
    todo(21, "co-mention edges")


def corr_edges(prices: pd.DataFrame, folds: list[Fold], top_k: int = CORR_TOP_K) -> pd.DataFrame:
    """Per fold: each ticker's top-k most correlated peers over the training window, stored both ways."""
    todo(22, "correlation edges")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--types", nargs="+", choices=EDGE_TYPES, default=list(EDGE_TYPES))
    p.add_argument("--fold", type=int, required=True, help="walk-forward fold index (0-based)")
    p.parse_args()
    todo(28, "build_edges")


if __name__ == "__main__":
    main()
