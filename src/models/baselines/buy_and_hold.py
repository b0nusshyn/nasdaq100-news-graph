"""Baseline: 1. buy-and-hold, always predicts up (does any signal exist?). Owner: Evaluation. Issue #25."""

from __future__ import annotations

import pandas as pd

from src.core.todo import todo


class BuyAndHold:
    name = "buy_and_hold"

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        todo(25, "buy_and_hold.fit")

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        todo(25, "buy_and_hold.predict_proba")
