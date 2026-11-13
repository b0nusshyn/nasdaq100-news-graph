"""Technical indicators used as LSTM input. Computed on the fly, not stored. Owner: Data. Issue #19.

Every indicator at date t uses prices up to and including t only.
"""

from __future__ import annotations

import pandas as pd

from src.core.todo import todo

FEATURES = ("ret_1d", "volatility_20d", "rsi_14", "macd", "macd_signal", "volume_z_20d")


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Return (date, ticker, *FEATURES), one row per prices row."""
    todo(19, "technical indicators")
