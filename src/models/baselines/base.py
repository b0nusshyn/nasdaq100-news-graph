"""Common interface for every baseline. Owner: Evaluation.

`src.eval.run_baselines` drives each baseline through the same folds as the proposed model.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Baseline(Protocol):
    name: str

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        """`panel` is indexed by (date, ticker) and holds prices, indicators, news features and labels.
        Fit only on rows whose date is in `train_dates`."""

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        """P(up) indexed by (date, ticker) for rows in `test_dates`. Earlier rows of `panel` may be
        read as history (e.g. LSTM windows) but never labels from the test window."""
