"""Per-ticker ARIMA(1,0,0) on log returns, with Gaussian forecast probabilities.

Forecast uncertainty conditions on fixed fitted parameters. Constant training
returns use their deterministic sign; short or failed fits use a neutral 0.5.
"""

from __future__ import annotations

import warnings
from math import erf, sqrt
from numbers import Integral

import numpy as np
import pandas as pd


def _up_probability(forecast, coefficient: float) -> float:
    """P(sum of future log returns > 0), including AR(1) cross-day covariance."""
    means = np.asarray(forecast.predicted_mean)
    variances = np.asarray(forecast.var_pred_mean)
    total_variance = float(variances.sum())
    # AR(1): covariance(i, j) = coefficient ** (j - i) * variance(i).
    for earlier in range(len(means)):
        for later in range(earlier + 1, len(means)):
            total_variance += 2 * coefficient ** (later - earlier) * variances[earlier]
    total_mean = float(means.sum())
    if not np.isfinite(total_mean) or not np.isfinite(total_variance) or total_variance < 0:
        raise ValueError("invalid forecast mean or variance")
    if total_variance == 0:
        return float(total_mean > 0)
    return 0.5 * (1 + erf(total_mean / sqrt(2 * total_variance)))


class Arima:
    name = "arima"
    minimum_returns = 20

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        """Estimate each ticker once per fold; unusable histories fall back to P(up)=0.5."""
        from statsmodels.tsa.arima.model import ARIMA

        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon not in (1, 5):
            raise ValueError("horizon must be 1 or 5")
        if train_dates.empty or not train_dates.is_unique or not train_dates.is_monotonic_increasing:
            raise ValueError("train_dates must be nonempty, sorted and unique")
        if not panel.index.is_unique:
            raise ValueError("panel must have unique date/ticker rows")

        self.horizon = horizon
        self.train_end = train_dates[-1]
        self.models = {}
        self.constant_returns = {}
        self.last_prices = {}
        training = panel.loc[panel.index.get_level_values("date").isin(train_dates)]
        for ticker, group in training.groupby(level="ticker"):
            history = group.droplevel("ticker").reindex(train_dates)
            close = history["close"].where(history["close"].gt(0) & np.isfinite(history["close"]))
            returns = np.log(close).diff().where(history["in_index"].eq(True))
            self.last_prices[ticker] = close.iloc[-1]
            observed = returns.dropna()
            if len(observed) < self.minimum_returns:
                warnings.warn(f"ARIMA {ticker}: insufficient history; using P(up)=0.5", stacklevel=2)
                continue
            if observed.std() < 1e-12:
                self.constant_returns[ticker] = float(observed.mean())
                continue
            try:
                result = ARIMA(returns.to_numpy(), order=(1, 0, 0), trend="c").fit()
                if not result.mle_retvals["converged"] or not np.isfinite(result.params).all():
                    raise ValueError("fit did not converge")
                self.models[ticker] = result
            except (ValueError, np.linalg.LinAlgError) as error:
                warnings.warn(f"ARIMA {ticker}: {error}; using P(up)=0.5", stacklevel=2)

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        """Update states with observed closes, never refitting parameters or reading labels."""
        if not hasattr(self, "train_end"):
            raise ValueError("call fit before predict_proba")
        if not test_dates.empty and test_dates.min() <= self.train_end:
            raise ValueError("test dates must follow the training window")
        dates = panel.index.get_level_values("date")
        test_rows = dates.isin(test_dates) & panel["in_index"]
        probabilities = pd.Series(0.5, index=panel.index[test_rows], dtype="float32", name="p_up")
        if probabilities.empty:
            return probabilities

        calendar = pd.DatetimeIndex(dates.unique()).sort_values()
        calendar = calendar[(calendar > self.train_end) & (calendar <= test_dates.max())]
        for ticker in probabilities.index.get_level_values("ticker").unique():
            result = self.models.get(ticker)
            constant = self.constant_returns.get(ticker)
            if result is None and constant is None:
                continue
            history = panel.xs(ticker, level="ticker").reindex(calendar)
            previous_close = self.last_prices[ticker]
            for date, close in history["close"].items():
                valid_close = np.isfinite(close) and close > 0
                valid_previous = np.isfinite(previous_close) and previous_close > 0
                daily_return = np.log(close / previous_close) if valid_close and valid_previous else np.nan
                previous_close = close
                try:
                    if result is not None:
                        result = result.extend([daily_return])
                    if (date, ticker) not in probabilities.index or not valid_close:
                        continue
                    if result is None:
                        probability = float(constant > 0)
                    else:
                        forecast = result.get_forecast(steps=self.horizon)
                        probability = _up_probability(forecast, float(result.arparams[0]))
                    probabilities.loc[(date, ticker)] = probability
                except (ValueError, np.linalg.LinAlgError) as error:
                    warnings.warn(f"ARIMA {ticker}: forecast failed ({error}); using P(up)=0.5", stacklevel=2)
                    break
        return probabilities
