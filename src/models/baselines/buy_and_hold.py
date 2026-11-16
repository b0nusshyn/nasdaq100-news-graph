"""Baseline: 1. buy-and-hold, always predicts up (does any signal exist?). Owner: Evaluation. Issue #25."""

from __future__ import annotations

from numbers import Integral

import pandas as pd


class BuyAndHold:
    name = "buy_and_hold"

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        """Validate the horizon; this constant baseline has nothing to learn."""
        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon not in (1, 5):
            raise ValueError("horizon must be 1 or 5")

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        """Predict up for in-index test rows, preserving their order.

        The caller attaches labels and removes missing outcomes for evaluation.
        """
        test_rows = panel.index.get_level_values("date").isin(test_dates) & panel["in_index"]
        return pd.Series(1.0, index=panel.index[test_rows], dtype="float32", name="p_up")
