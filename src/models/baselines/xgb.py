"""Pooled XGBoost on the six causal technical indicators (Issue #32).

Inputs: ret_1d, volatility_20d, rsi_14, macd, macd_signal, volume_z_20d.
Prepare them with compute_indicators. Trees handle missing values directly;
no scaling or imputation is fitted separately.
"""

from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np
import pandas as pd

from src.data.indicators import FEATURES


class XGBoost:
    name = "xgboost"
    features = FEATURES
    minimum_samples = 20

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        """Fit on eligible labelled rows; unusable samples use training class frequency."""
        valid_horizon = isinstance(horizon, Integral) and not isinstance(horizon, bool)
        if not valid_horizon or horizon not in (1, 5):
            raise ValueError("horizon must be 1 or 5")

        if train_dates.empty or train_dates.hasnans:
            raise ValueError("train_dates must be nonempty and contain no missing dates")
        if not train_dates.is_unique or not train_dates.is_monotonic_increasing:
            raise ValueError("train_dates must be sorted and unique")
        if panel.index.names != ["date", "ticker"] or not panel.index.is_unique:
            raise ValueError("panel must have a unique (date, ticker) index")

        missing_features = sorted(set(self.features) - set(panel.columns))
        if missing_features:
            raise ValueError(f"panel is missing indicator columns: {missing_features}")

        dates = panel.index.get_level_values("date")
        training_rows = dates.isin(train_dates) & panel["in_index"]
        label_column = f"y_{horizon}d"
        training = panel.loc[training_rows].dropna(subset=[label_column])
        labels = training[label_column]
        if not labels.isin([0, 1]).all():
            raise ValueError("training labels must contain only 0 and 1")

        self.train_end = None
        self.model = None
        self.fallback_probability = float(labels.mean()) if not labels.empty else 0.5

        training_features = training.loc[:, self.features].replace([np.inf, -np.inf], np.nan)
        too_few_samples = len(training) < self.minimum_samples
        single_class = labels.nunique() < 2
        no_features = not training_features.notna().any().any()
        if too_few_samples or single_class or no_features:
            warnings.warn(
                "XGBoost: insufficient rows, classes or features; using training class frequency "
                "(0.5 when no labels are available)",
                UserWarning,
                stacklevel=2,
            )
            self.train_end = train_dates[-1]
            return

        # Import only when training is needed, so lightweight CLI imports still work.
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            reg_lambda=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )
        classifier.fit(training_features, labels.astype(int))
        self.model = classifier
        self.train_end = train_dates[-1]

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        """Return float32 P(up) for every eligible test row, without reading labels."""
        if getattr(self, "train_end", None) is None:
            raise ValueError("call fit before predict_proba")
        if not test_dates.empty and test_dates.min() <= self.train_end:
            raise ValueError("test dates must follow the training window")
        if panel.index.names != ["date", "ticker"] or not panel.index.is_unique:
            raise ValueError("panel must have a unique (date, ticker) index")

        dates = panel.index.get_level_values("date")
        test_rows = dates.isin(test_dates) & panel["in_index"]
        test = panel.loc[test_rows]
        probabilities = pd.Series(self.fallback_probability, index=test.index, dtype="float32", name="p_up")
        if test.empty or self.model is None:
            return probabilities

        test_features = test.loc[:, self.features].replace([np.inf, -np.inf], np.nan)
        probabilities.iloc[:] = self.model.predict_proba(test_features)[:, 1].astype("float32")
        return probabilities
