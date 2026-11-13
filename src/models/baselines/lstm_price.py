"""Baseline: 5. LSTM on prices/indicators only (isolates the contribution of news). Owner: Evaluation. Issue #33."""

from __future__ import annotations

import pandas as pd

from src.core.todo import todo


class LSTMPrice:
    name = "lstm_price"

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        todo(33, "lstm_price.fit")

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        todo(33, "lstm_price.predict_proba")
