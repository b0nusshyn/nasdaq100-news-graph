"""Company graph edges. Owner: Modelling. Issues #8, #21, #22, #28.

Output: `edges.parquet` (SCHEMA.md). `sector` edges are static (fold = -1); `comention` and
`corr` weights are computed per walk-forward fold from that fold's training window only, so
this runs once per fold and each run replaces that fold's rows.

    python -m src.graph.build_edges --types sector comention corr --fold 0
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.core.schema import CORR_TOP_K, EDGE_TYPES
from src.core.todo import todo
from src.eval.splitter import Fold


def sector_edges(constituents: pd.DataFrame) -> pd.DataFrame:
    """Both directions between every pair sharing a GICS sector; weight 1, fold -1."""
    todo(8, "sector edges")


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
