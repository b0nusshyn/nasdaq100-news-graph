"""Baseline 6: LSTM + news, no graph. Owner: Evaluation (encoder from Modelling). Issue #38.

The decisive comparison: the proposed model with the GAT removed. Use
`src.models.encoder.CompanyEncoder` unchanged, so the only difference is the graph.
"""

from __future__ import annotations

import pandas as pd

from src.core.todo import todo


class LSTMNews:
    name = "lstm_news"

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        todo(38, "lstm_news.fit")

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        todo(38, "lstm_news.predict_proba")
