"""Forward returns and up/down labels per horizon, on the trading-day calendar.

Output: `labels.parquet` (SCHEMA.md). Owner: Data. Issue #15.

    python -m src.data.build_labels --horizons 1 5
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.core.schema import HORIZONS
from src.core.todo import todo


def make_labels(prices: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """ret_{h}d = close[t+h] / close[t] - 1 (per ticker, trading days); y_{h}d = ret > 0.

    The last h rows of each ticker have no label (NaN / <NA>).
    """
    todo(15, "build labels")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.parse_args()
    todo(15, "build_labels")


if __name__ == "__main__":
    main()
