import numpy as np
import pytest

from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.eval.metrics import METRICS, classification_details, classification_metrics


@pytest.mark.parametrize(
    "probabilities, expected",
    [
        ([0.1, 0.2, 0.8, 0.9], {"accuracy": 1, "f1": 1, "mcc": 1, "auc": 1}),
        ([0.9, 0.8, 0.2, 0.1], {"accuracy": 0, "f1": 0, "mcc": -1, "auc": 0}),
        ([0.1, 0.8, 0.4, 0.9], {"accuracy": 0.5, "f1": 0.5, "mcc": 0, "auc": 0.75}),
        ([0.1, 0.5, 0.5, 0.9], {"accuracy": 0.75, "f1": 0.8, "mcc": 1 / np.sqrt(3), "auc": 0.875}),
    ],
)
def test_known_results(probabilities, expected):
    result = classification_metrics(np.array([0, 0, 1, 1]), np.array(probabilities))
    assert result == pytest.approx(expected)
    assert tuple(result) == METRICS
    assert all(isinstance(value, float) for value in result.values())


def test_custom_threshold_does_not_change_auc():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])
    result = classification_metrics(labels, probabilities, threshold=0.9)
    assert result == pytest.approx({"accuracy": 0.75, "f1": 2 / 3, "mcc": 1 / np.sqrt(3), "auc": 0.75})


@pytest.mark.parametrize("threshold", [0, 0.5, 1])
def test_threshold_equality_predicts_up(threshold):
    with pytest.warns(UserWarning, match="MCC"):
        result = classification_metrics(np.array([0, 1]), np.array([threshold, threshold]), threshold=threshold)
    assert result["accuracy"] == 0.5
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["auc"] == 0.5
    assert np.isnan(result["mcc"])


@pytest.mark.parametrize(
    "labels, probabilities, accuracy, f1",
    [
        ([0, 0], [0.1, 0.2], 1, np.nan),
        ([1, 1], [0.8, 0.9], 1, 1),
        ([0, 0], [0.8, 0.9], 0, 0),
        ([1, 1], [0.1, 0.2], 0, 0),
        ([1], [0.8], 1, 1),
    ],
)
def test_single_class_labels_warn_for_undefined_metrics(labels, probabilities, accuracy, f1):
    with pytest.warns(UserWarning) as recorded:
        result = classification_metrics(np.array(labels), np.array(probabilities))
    messages = " ".join(str(warning.message) for warning in recorded)
    assert "MCC is undefined" in messages
    assert "AUC is undefined" in messages
    assert result["accuracy"] == accuracy
    assert np.isnan(result["mcc"])
    assert np.isnan(result["auc"])
    if np.isnan(f1):
        assert np.isnan(result["f1"])
        assert "F1 is undefined" in messages
    else:
        assert result["f1"] == f1


def test_no_predicted_positives_with_mixed_labels():
    with pytest.warns(UserWarning, match="MCC"):
        result = classification_metrics(np.array([0, 1]), np.array([0.1, 0.2]))
    assert result["accuracy"] == 0.5
    assert result["f1"] == 0
    assert np.isnan(result["mcc"])
    assert result["auc"] == 1


def test_synthetic_predictions_against_pairwise_auc():
    predictions = read_table("predictions", FIXTURES_DIR / "predictions" / "random.parquet")
    labels = predictions["y_true"].to_numpy()
    probabilities = predictions["p_up"].to_numpy()
    result = classification_metrics(labels, probabilities)
    assert result["accuracy"] == pytest.approx(np.mean((probabilities >= 0.5) == labels))
    positive_scores = probabilities[labels == 1][:, None]
    negative_scores = probabilities[labels == 0][None, :]
    expected_auc = np.mean((positive_scores > negative_scores) + 0.5 * (positive_scores == negative_scores))
    assert result["auc"] == pytest.approx(expected_auc)


@pytest.mark.parametrize(
    "labels, probabilities",
    [
        ([], []), ([0, 1], [0.2]), ([[0, 1]], [[0.2, 0.8]]), (1, 0.8),
        ([0, 2], [0.2, 0.8]), ([0, 0.5], [0.2, 0.8]), ([0, np.nan], [0.2, 0.8]),
        ([0, np.inf], [0.2, 0.8]), ([0, None], [0.2, 0.8]), (["0", "1"], [0.2, 0.8]),
        ([0, 1], [-0.1, 0.8]), ([0, 1], [0.2, 1.1]), ([0, 1], [np.nan, 0.8]),
        ([0, 1], [0.2, np.inf]), ([0, 1], ["0.2", "0.8"]), ([0, 1], [0.2j, 0.8]),
    ],
)
@pytest.mark.parametrize("metric_function", [classification_metrics, classification_details])
def test_invalid_inputs_raise(labels, probabilities, metric_function):
    with pytest.raises(ValueError):
        metric_function(np.array(labels), np.array(probabilities))


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, "0.5", [0.5], None])
@pytest.mark.parametrize("metric_function", [classification_metrics, classification_details])
def test_invalid_threshold_raises(threshold, metric_function):
    with pytest.raises(ValueError, match="threshold"):
        metric_function(np.array([0, 1]), np.array([0.2, 0.8]), threshold=threshold)


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (0.5, {"tp": 2, "tn": 3, "fp": 1, "fn": 2, "precision": 2 / 3, "recall": 0.5, "specificity": 0.75}),
        (0.8, {"tp": 1, "tn": 3, "fp": 1, "fn": 3, "precision": 0.5, "recall": 0.25, "specificity": 0.75}),
    ],
)
def test_details_known_results_and_metrics_agree(threshold, expected):
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    probabilities = [0.9, 0.5, 0.4, 0.1, 0.8, 0.3, 0.2, 0.1]
    details = classification_details(labels, probabilities, threshold)
    metrics = classification_metrics(labels, probabilities, threshold)
    assert details == pytest.approx(expected)
    assert all(isinstance(details[key], int) for key in ("tp", "tn", "fp", "fn"))
    assert metrics["accuracy"] == (details["tp"] + details["tn"]) / len(labels)
    assert metrics["f1"] == pytest.approx(2 * details["tp"] / (2 * details["tp"] + details["fp"] + details["fn"]))


@pytest.mark.parametrize("threshold", [0, 0.5, 1])
def test_details_threshold_equality_predicts_up(threshold):
    details = classification_details([0, 1], [threshold, threshold], threshold)
    assert details == {"tp": 1, "tn": 0, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1, "specificity": 0}


@pytest.mark.parametrize(
    "labels, probabilities, expected, undefined",
    [
        ([0, 0], [0.1, 0.2], (0, 2, 0, 0, np.nan, np.nan, 1), ["precision", "recall"]),
        ([1, 1], [0.8, 0.9], (2, 0, 0, 0, 1, 1, np.nan), ["specificity"]),
        ([0, 0], [0.8, 0.9], (0, 0, 2, 0, 0, np.nan, 0), ["recall"]),
        ([1, 1], [0.1, 0.2], (0, 0, 0, 2, np.nan, 0, np.nan), ["precision", "specificity"]),
        ([0, 1], [0.1, 0.2], (0, 1, 0, 1, np.nan, 0, 1), ["precision"]),
        ([1], [0.8], (1, 0, 0, 0, 1, 1, np.nan), ["specificity"]),
    ],
)
def test_details_undefined_rates(labels, probabilities, expected, undefined):
    with pytest.warns(UserWarning) as recorded:
        details = classification_details(labels, probabilities)
    assert list(details.values()) == pytest.approx(expected, nan_ok=True)
    assert len(recorded) == len(undefined)
    for name, warning in zip(undefined, recorded, strict=True):
        assert f"{name.capitalize()} is undefined" in str(warning.message)


def test_details_accept_synthetic_predictions():
    predictions = read_table("predictions", FIXTURES_DIR / "predictions" / "random.parquet")
    labels = predictions["y_true"].to_numpy()
    probabilities = predictions["p_up"].to_numpy()
    details = classification_details(labels, probabilities)
    metrics = classification_metrics(labels, probabilities)
    assert sum(details[key] for key in ("tp", "tn", "fp", "fn")) == len(labels)
    assert metrics["accuracy"] == (details["tp"] + details["tn"]) / len(labels)
