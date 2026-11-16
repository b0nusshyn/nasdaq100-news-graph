import pandas as pd
import pytest

from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.core.schema import conform, validate
from src.eval.splitter import walk_forward
from src.models.baselines.buy_and_hold import BuyAndHold


@pytest.fixture
def panel():
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    labels = read_table("labels", FIXTURES_DIR / "labels.parquet")
    return prices.merge(labels, on=["date", "ticker"], validate="one_to_one").set_index(["date", "ticker"])


@pytest.mark.parametrize("horizon", [1, 5])
def test_predictions_on_shared_folds_match_schema(panel, horizon):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    folds = walk_forward(dates, train_days=20, test_days=10, horizon=horizon)
    model = BuyAndHold()
    original = panel.copy(deep=True)
    predictions = []

    for fold in folds:
        assert model.fit(panel, fold.train, horizon) is None
        probabilities = model.predict_proba(panel, fold.test)
        expected_index = panel.loc[
            panel.index.get_level_values("date").isin(fold.test) & panel["in_index"]
        ].index
        pd.testing.assert_index_equal(probabilities.index, expected_index)
        assert probabilities.eq(1.0).all()
        assert str(probabilities.dtype) == "float32"
        assert probabilities.name == "p_up"

        prediction = probabilities.to_frame().join(panel[f"y_{horizon}d"].rename("y_true"))
        prediction = prediction.dropna(subset=["y_true"]).reset_index()
        predictions.append(prediction.assign(model=model.name, horizon=horizon, fold=fold.index))

    assert len(folds) == 3
    output = conform(pd.concat(predictions, ignore_index=True), "predictions")
    validate(output, "predictions")
    assert not output.empty
    assert output["horizon"].eq(horizon).all()
    pd.testing.assert_frame_equal(panel, original)


def test_late_joiner_is_only_predicted_after_joining(panel):
    model = BuyAndHold()
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    probabilities = model.predict_proba(panel, dates[[29, 30]])
    assert (dates[29], "FFF") not in probabilities.index
    assert (dates[30], "FFF") in probabilities.index


def test_predictions_do_not_need_prices_labels_or_training(panel):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    model = BuyAndHold()
    expected = model.predict_proba(panel, dates[-5:])
    minimal_panel = panel[["in_index"]].copy()
    model.fit(minimal_panel, dates[:20], 5)
    pd.testing.assert_series_equal(model.predict_proba(minimal_panel, dates[-5:]), expected)
    assert len(expected) == 30  # Missing trailing labels do not suppress predictions.


@pytest.mark.parametrize("empty_panel", [False, True])
def test_empty_prediction_selection(panel, empty_panel):
    dates = pd.DatetimeIndex([])
    if empty_panel:
        dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique())
        panel = panel.iloc[:0]
    probabilities = BuyAndHold().predict_proba(panel, dates)
    assert probabilities.empty
    assert probabilities.index.names == ["date", "ticker"]
    assert probabilities.name == "p_up"
    assert str(probabilities.dtype) == "float32"


@pytest.mark.parametrize("horizon", [0, 2, True, 1.0, "1", None])
def test_invalid_horizon(panel, horizon):
    with pytest.raises(ValueError, match="horizon must be 1 or 5"):
        BuyAndHold().fit(panel, pd.DatetimeIndex([]), horizon)
