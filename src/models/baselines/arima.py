"""Baseline: 2. ARIMA on each ticker's returns, price only. Owner: Evaluation. Issue #25."""

from __future__ import annotations

import pandas as pd

from src.core.todo import todo


class Arima:
    name = "arima"

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        todo(25, "arima.fit")

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        todo(25, "arima.predict_proba")
