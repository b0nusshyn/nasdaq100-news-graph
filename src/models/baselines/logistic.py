"""Pooled logistic regression on the project's causal indicators (Issue #25).

Inputs: ret_1d, volatility_20d, rsi_14, macd, macd_signal, volume_z_20d.
Compute them with compute_indicators before building the baseline panel.
"""

from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np
import pandas as pd

from src.data.indicators import FEATURES


class Logistic:
    name = "logistic"
    features = FEATURES  # returns, volatility, RSI, MACD, MACD signal, volume z-score

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        """Fit median imputation, scaling and L2 regression on eligible training rows."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon not in (1, 5):
            raise ValueError("horizon must be 1 or 5")
        if train_dates.empty:
            raise ValueError("train_dates must not be empty")

        training_rows = panel.index.get_level_values("date").isin(train_dates) & panel["in_index"]
        training = panel.loc[training_rows].dropna(subset=[f"y_{horizon}d"])
        labels = training[f"y_{horizon}d"]
        if training.empty or not labels.isin([0, 1]).all():
            raise ValueError("training requires nonmissing binary labels")

        self.train_end = train_dates.max()
        self.pipeline = None
        self.fallback_probability = float(labels.mean())
        features = training.loc[:, self.features].replace([np.inf, -np.inf], np.nan)
        if labels.nunique() == 1 or not features.notna().any().any():
            warnings.warn("Logistic: using training class frequency; insufficient features or classes", stacklevel=2)
            return

        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000),
        )
        self.pipeline.fit(features, labels.astype(int))

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        """Return P(up) for all in-index test rows without reading their labels."""
        if not hasattr(self, "train_end"):
            raise ValueError("call fit before predict_proba")
        if not test_dates.empty and test_dates.min() <= self.train_end:
            raise ValueError("test dates must follow the training window")

        test_rows = panel.index.get_level_values("date").isin(test_dates) & panel["in_index"]
        test = panel.loc[test_rows]
        probabilities = pd.Series(self.fallback_probability, index=test.index, dtype="float32", name="p_up")
        if self.pipeline is not None and not test.empty:
            features = test.loc[:, self.features].replace([np.inf, -np.inf], np.nan)
            probabilities.iloc[:] = self.pipeline.predict_proba(features)[:, 1].astype("float32")
        return probabilities
