"""Baseline: 4. XGBoost on indicators (+ news aggregates). Owner: Evaluation. Issue #32."""

from __future__ import annotations

import pandas as pd

from src.core.todo import todo


class XGBoost:
    name = "xgboost"

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        todo(32, "xgboost.fit")

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        todo(32, "xgboost.predict_proba")
