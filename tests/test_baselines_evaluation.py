import warnings

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")

from src.core.io import read_table  # noqa: E402
from src.core.paths import FIXTURES_DIR  # noqa: E402
from src.core.schema import conform, validate  # noqa: E402
from src.data.indicators import compute_indicators  # noqa: E402
from src.eval.backtest import backtest  # noqa: E402
from src.eval.buy_and_hold import buy_and_hold_benchmark  # noqa: E402
from src.eval.metrics import classification_metrics  # noqa: E402
from src.eval.splitter import walk_forward  # noqa: E402
from src.models.baselines.arima import Arima  # noqa: E402
from src.models.baselines.buy_and_hold import BuyAndHold  # noqa: E402
from src.models.baselines.logistic import Logistic  # noqa: E402


@pytest.mark.parametrize("horizon", [1, 5])
def test_same_rows_schema_metrics_and_backtest_across_folds(horizon):
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    labels = read_table("labels", FIXTURES_DIR / "labels.parquet")
    panel = (
        prices.merge(labels, on=["date", "ticker"])
        .merge(compute_indicators(prices), on=["date", "ticker"])
        .set_index(["date", "ticker"])
    )
    dates = pd.DatetimeIndex(prices["date"].unique()).sort_values()
    original = panel.copy(deep=True)
    folds = walk_forward(dates[:-5], train_days=35, test_days=5, horizon=horizon)
    for fold in folds:
        expected = BuyAndHold().predict_proba(panel, fold.test)
        assert len(dates[(dates > fold.train[-1]) & (dates < fold.test[0])]) == 5
        for model in (BuyAndHold(), Logistic(), Arima()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model.fit(panel, fold.train, horizon)
                probabilities = model.predict_proba(panel, fold.test)
            if model.name == "logistic":
                assert model.pipeline is not None
            elif model.name == "arima":
                assert model.models
            pd.testing.assert_index_equal(probabilities.index, expected.index)
            assert probabilities.between(0, 1).all()
            assert str(probabilities.dtype) == "float32"
            assert probabilities.name == "p_up"
            output = probabilities.to_frame().join(panel[f"y_{horizon}d"].rename("y_true"))
            output = output.dropna(subset=["y_true"]).reset_index()
            output = conform(output.assign(model=model.name, horizon=horizon, fold=fold.index), "predictions")
            validate(output, "predictions")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                metrics = classification_metrics(output["y_true"].to_numpy(), output["p_up"].to_numpy())
            assert 0 <= metrics["accuracy"] <= 1
            if model.name == "buy_and_hold":
                first_entry = dates.get_loc(fold.test[0])
                last_entry = dates.get_loc(fold.test[-1])
                exit_position = first_entry + ((last_entry - first_entry) // horizon + 1) * horizon
                holding_dates = dates[first_entry : exit_position + 1]
                result = buy_and_hold_benchmark(prices, holding_dates)
                assert result.turnover > 0
                assert result.equity.index[-1] == dates[exit_position]
            else:
                result = backtest(output, labels, horizon)
                assert result.equity.notna().all()
    pd.testing.assert_frame_equal(panel, original)
