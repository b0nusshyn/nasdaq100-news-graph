from math import erf, sqrt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from src.core.io import read_table  # noqa: E402
from src.core.paths import FIXTURES_DIR  # noqa: E402
from src.models.baselines.arima import Arima, _up_probability  # noqa: E402


@pytest.fixture
def panel():
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    return prices.loc[prices["ticker"] == "AAA"].set_index(["date", "ticker"])


@pytest.mark.parametrize("horizon", [1, 5])
def test_future_prices_do_not_change_earlier_predictions(panel, horizon):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    model = Arima()
    model.fit(panel, dates[:35], horizon)
    original = panel.copy(deep=True)
    output = model.predict_proba(panel, dates[40:45])
    changed = panel.copy()
    changed.loc[changed.index.get_level_values("date") > dates[42], "close"] *= 3
    actual = model.predict_proba(changed, dates[40:45])
    pd.testing.assert_series_equal(output.iloc[:3], actual.iloc[:3])
    pd.testing.assert_series_equal(output, model.predict_proba(panel, dates[40:45]))
    pd.testing.assert_frame_equal(panel, original)
    assert output.between(0, 1).all()
    assert str(output.dtype) == "float32"

    second = Arima()
    second.fit(changed, dates[:35], horizon)
    np.testing.assert_allclose(model.models["AAA"].params, second.models["AAA"].params)


@pytest.mark.parametrize("coefficient", [-0.4, 0.0, 0.7])
def test_probability_includes_cross_day_covariance(coefficient):
    horizon = 5
    innovation_variance = 0.01
    variances = [
        innovation_variance * sum(coefficient ** (2 * lag) for lag in range(day + 1)) for day in range(horizon)
    ]
    forecast = SimpleNamespace(predicted_mean=np.full(horizon, 0.02), var_pred_mean=variances)
    # Each innovation affects every remaining return in the cumulative forecast.
    expected_variance = innovation_variance * sum(
        sum(coefficient**lag for lag in range(remaining)) ** 2 for remaining in range(1, horizon + 1)
    )
    expected = 0.5 * (1 + erf(0.1 / sqrt(2 * expected_variance)))
    assert _up_probability(forecast, coefficient) == pytest.approx(expected)


@pytest.mark.parametrize("growth, expected", [(0.0, 0.0), (0.01, 1.0), (-0.01, 0.0)])
def test_constant_returns(panel, growth, expected):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    panel["close"] = 100 * np.exp(growth * np.arange(len(panel)))
    model = Arima()
    model.fit(panel, dates[:35], 5)
    assert model.predict_proba(panel, dates[40:45]).eq(expected).all()


def test_insufficient_history_and_unknown_ticker(panel):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    model = Arima()
    with pytest.warns(UserWarning, match="insufficient history"):
        model.fit(panel, dates[:10], 1)
    assert model.predict_proba(panel, dates[15:20]).eq(0.5).all()
    model.fit(panel, dates[:35], 1)
    unknown = panel.reset_index().assign(ticker="NEW").set_index(["date", "ticker"])
    assert model.predict_proba(unknown, dates[40:45]).eq(0.5).all()


def test_failed_fit_is_explicit(panel, monkeypatch):
    from statsmodels.tsa.arima.model import ARIMA

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic fit failure")

    monkeypatch.setattr(ARIMA, "fit", fail)
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    model = Arima()
    with pytest.warns(UserWarning, match="synthetic fit failure"):
        model.fit(panel, dates[:35], 1)
    assert model.predict_proba(panel, dates[40:45]).eq(0.5).all()


def test_missing_close_does_not_become_a_multi_day_return(panel):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    model = Arima()
    model.fit(panel, dates[:35], 1)
    panel.loc[(dates[41], "AAA"), "close"] = np.nan
    output = model.predict_proba(panel, dates[40:45])
    assert output.loc[(dates[41], "AAA")] == 0.5
    assert output.between(0, 1).all()
    assert model.predict_proba(panel, pd.DatetimeIndex([])).empty


@pytest.mark.parametrize("horizon", [True, 0, 2, 1.0])
def test_invalid_horizon(panel, horizon):
    with pytest.raises(ValueError, match="horizon"):
        Arima().fit(panel, pd.DatetimeIndex([]), horizon)


def test_predict_before_fit_and_overlapping_dates(panel):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    with pytest.raises(ValueError, match="call fit"):
        Arima().predict_proba(panel, dates[-5:])
    model = Arima()
    model.fit(panel, dates[:35], 1)
    with pytest.raises(ValueError, match="follow"):
        model.predict_proba(panel, dates[:5])


def test_prediction_does_not_refit_and_missing_sessions_are_not_bridged(panel, monkeypatch):
    from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    model = Arima()
    model.fit(panel, dates[:35], 1)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("prediction must not estimate model parameters")

    observed_returns = []
    original_extend = ARIMAResults.extend

    def record_extend(self, returns, **kwargs):
        observed_returns.extend(returns)
        return original_extend(self, returns, **kwargs)

    monkeypatch.setattr(ARIMA, "fit", forbidden_fit)
    monkeypatch.setattr(ARIMAResults, "extend", record_extend)
    panel.loc[(dates[41], "AAA"), "close"] = np.nan
    model.predict_proba(panel, dates[40:45])
    assert np.isnan(observed_returns[6])  # Missing day 41.
    assert np.isnan(observed_returns[7])  # Day 42 has no previous close.
    assert len(observed_returns) == 10  # Embargo plus test days, never later days.


def test_nonconverged_fit_and_forecast_failure_are_explicit(panel, monkeypatch):
    from statsmodels.tsa.arima.model import ARIMA

    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    model = Arima()
    model.fit(panel, dates[:35], 1)

    class FailedForecast:
        def extend(self, returns):
            raise ValueError("synthetic forecast failure")

    model.models["AAA"] = FailedForecast()
    with pytest.warns(UserWarning, match="forecast failed"):
        assert model.predict_proba(panel, dates[40:45]).eq(0.5).all()

    failed_result = SimpleNamespace(mle_retvals={"converged": False})
    monkeypatch.setattr(ARIMA, "fit", lambda self: failed_result)
    with pytest.warns(UserWarning, match="did not converge"):
        model.fit(panel, dates[:35], 1)
    assert model.predict_proba(panel, dates[40:45]).eq(0.5).all()


@pytest.mark.parametrize("mean, expected", [(0.0, 0.0), (-0.1, 0.0), (0.1, 1.0)])
def test_zero_forecast_variance_uses_strictly_positive_return(mean, expected):
    forecast = SimpleNamespace(predicted_mean=[mean], var_pred_mean=[0.0])
    assert _up_probability(forecast, 0.0) == expected
