"""Baseline: 3. logistic regression on indicators. Owner: Evaluation. Issue #25."""

from __future__ import annotations

import pandas as pd

from src.core.todo import todo


class Logistic:
    name = "logistic"

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        todo(25, "logistic.fit")

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        todo(25, "logistic.predict_proba")
