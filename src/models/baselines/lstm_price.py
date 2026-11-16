"""Price-only LSTM on the six causal indicators (Evaluation, Issue #33).

Windows end at the prediction day's close. Earlier nonmember observations may
supply context, but only in-index endpoints are trained or predicted. Missing
rows break a window; missing feature values use training-only median imputation.
"""

from __future__ import annotations

import warnings
from numbers import Integral, Real

import numpy as np
import pandas as pd

from src.data.indicators import FEATURES


class LSTMPrice:
    name = "lstm_price"
    features = FEATURES

    def __init__(
        self,
        *,
        window_days: int = 20,
        hidden: int = 16,
        epochs: int = 5,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        seed: int = 42,
    ) -> None:
        """Use a small CPU model; shorter settings support inexpensive unit tests."""
        for name, value in (
            ("window_days", window_days),
            ("hidden", hidden),
            ("epochs", epochs),
            ("batch_size", batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(learning_rate, Real) or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")

        self.window_days = window_days
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def _validate_panel(self, panel: pd.DataFrame) -> None:
        """Check the shared index and required input columns without reading labels."""
        if panel.index.names != ["date", "ticker"] or not panel.index.is_unique:
            raise ValueError("panel must have a unique (date, ticker) index")
        dates = panel.index.get_level_values("date")
        if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans:
            raise ValueError("panel dates must be nonmissing datetime dates")
        if dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("panel dates must be timezone-naive dates at midnight")
        missing = (set(self.features) | {"in_index"}) - set(panel.columns)
        if missing:
            raise ValueError(f"panel is missing columns: {sorted(missing)}")
        if panel["in_index"].isna().any():
            raise ValueError("in_index must not contain missing values")

    def _prepare_sequences(
        self,
        panel: pd.DataFrame,
        endpoints: pd.MultiIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return chronological windows and their positions in the endpoint index.

        The panel must include the trading calendar and available historical context.
        Never compress a missing ticker/session into a shorter observed-row window.
        """
        empty = np.empty((0, self.window_days, len(self.features)), dtype="float32")
        if endpoints.empty:
            return empty, np.array([], dtype=int)

        calendar = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
        calendar = calendar[calendar <= endpoints.get_level_values("date").max()]
        endpoint_tickers = endpoints.get_level_values("ticker")
        sequences = []
        positions = []

        for ticker in endpoint_tickers.unique():
            history = panel.xs(ticker, level="ticker")
            observed = calendar.isin(history.index)
            values = history.reindex(calendar).loc[:, self.features]
            values = values.replace([np.inf, -np.inf], np.nan).fillna(self.feature_medians)
            values = ((values - self.feature_means) / self.feature_scales).to_numpy(dtype="float32")

            for position in np.flatnonzero(endpoint_tickers == ticker):
                date = endpoints[position][0]
                stop = calendar.get_loc(date) + 1
                start = stop - self.window_days
                if start < 0 or not observed[start:stop].all():
                    continue
                sequences.append(values[start:stop])
                positions.append(position)

        if not sequences:
            return empty, np.array([], dtype=int)
        return np.stack(sequences), np.asarray(positions, dtype=int)

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        """Fit preprocessing and a pooled LSTM using only eligible training endpoints."""
        valid_horizon = isinstance(horizon, Integral) and not isinstance(horizon, bool)
        if not valid_horizon or horizon not in (1, 5):
            raise ValueError("horizon must be 1 or 5")
        if train_dates.empty or train_dates.hasnans:
            raise ValueError("train_dates must be nonempty and contain no missing dates")
        if not train_dates.is_unique or not train_dates.is_monotonic_increasing:
            raise ValueError("train_dates must be sorted and unique")
        self._validate_panel(panel)

        dates = panel.index.get_level_values("date")
        eligible = dates.isin(train_dates) & panel["in_index"]
        label_column = f"y_{horizon}d"
        training = panel.loc[eligible].dropna(subset=[label_column]).sort_index()
        labels = training[label_column]
        if not labels.isin([0, 1]).all():
            raise ValueError("training labels must contain only 0 and 1")

        self.model = None
        self.train_end = None
        self.fallback_probability = float(labels.mean()) if not labels.empty else 0.5
        feature_values = training.loc[:, self.features].replace([np.inf, -np.inf], np.nan)
        self.feature_medians = feature_values.median().fillna(0.0)
        imputed = feature_values.fillna(self.feature_medians)
        self.feature_means = imputed.mean().fillna(0.0)
        self.feature_scales = imputed.std(ddof=0).replace(0.0, 1.0).fillna(1.0)

        sequences, positions = self._prepare_sequences(panel, training.index)
        targets = labels.iloc[positions].to_numpy(dtype="float32")
        has_two_classes = np.unique(targets).size == 2
        has_features = feature_values.notna().any().any()
        if len(sequences) == 0 or not has_two_classes or not has_features:
            warnings.warn(
                "LSTMPrice: insufficient history, classes or features; using training class frequency "
                "(0.5 when no labels are available)",
                UserWarning,
                stacklevel=2,
            )
            self.train_end = train_dates[-1]
            return

        import torch
        from torch import nn

        from src.models.encoder import CompanyEncoder

        inputs = torch.from_numpy(sequences)
        target_tensor = torch.from_numpy(targets)
        # Keep initialization reproducible without changing another model's random state.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(self.seed)
            model = nn.Sequential(
                CompanyEncoder(len(self.features), hidden=self.hidden, layers=1, dropout=0.0),
                nn.Linear(self.hidden, 1),
            ).cpu()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_function = nn.BCEWithLogitsLoss()
            model.train()

            for _ in range(self.epochs):
                for start in range(0, len(inputs), self.batch_size):
                    stop = start + self.batch_size
                    optimizer.zero_grad()
                    logits = model(inputs[start:stop]).squeeze(-1)
                    loss = loss_function(logits, target_tensor[start:stop])
                    if not torch.isfinite(loss):
                        raise ValueError("LSTM training produced a non-finite loss")
                    loss.backward()
                    optimizer.step()

        self.model = model.eval()
        self.train_end = train_dates[-1]

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        """Predict every eligible test row; incomplete windows receive the training prior."""
        if getattr(self, "train_end", None) is None:
            raise ValueError("call fit before predict_proba")
        if not test_dates.empty and test_dates.min() <= self.train_end:
            raise ValueError("test dates must follow the training window")
        self._validate_panel(panel)

        dates = panel.index.get_level_values("date")
        eligible = dates.isin(test_dates) & panel["in_index"]
        endpoints = panel.index[eligible]
        probabilities = pd.Series(self.fallback_probability, index=endpoints, dtype="float32", name="p_up")
        if endpoints.empty or self.model is None:
            return probabilities

        sequences, positions = self._prepare_sequences(panel, endpoints)
        missing_windows = len(endpoints) - len(positions)
        if missing_windows:
            warnings.warn(
                f"LSTMPrice: {missing_windows} incomplete windows; using training class frequency",
                UserWarning,
                stacklevel=2,
            )

        import torch

        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                stop = start + self.batch_size
                inputs = torch.from_numpy(sequences[start:stop])
                values = torch.sigmoid(self.model(inputs).squeeze(-1)).numpy()
                if not np.isfinite(values).all():
                    raise ValueError("LSTM prediction produced non-finite probabilities")
                probabilities.iloc[positions[start:stop]] = values
        return probabilities
