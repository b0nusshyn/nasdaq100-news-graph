"""Classification metrics. Owner: Evaluation. Issue #13."""

from __future__ import annotations

import warnings
from numbers import Real

import numpy as np
import pandas as pd

METRICS = ("accuracy", "f1", "mcc", "auc")


def _confusion_counts(y_true: np.ndarray, p_up: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """Validate shared inputs and count outcomes with 1 (up) as positive."""
    y_true = np.asarray(y_true)
    p_up = np.asarray(p_up)
    if y_true.ndim != 1 or p_up.ndim != 1:
        raise ValueError("y_true and p_up must be equally sized 1-D arrays")
    if y_true.shape != p_up.shape:
        raise ValueError("y_true and p_up must be equally sized 1-D arrays")
    if y_true.size == 0:
        raise ValueError("y_true and p_up must not be empty")

    # Check types before comparing values so invalid inputs raise ValueError.
    label_error = "y_true must contain only binary labels 0 and 1, with no missing values"
    if y_true.dtype.kind not in "biuf":
        raise ValueError(label_error)
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(label_error)

    probability_error = "p_up must contain finite probabilities in [0, 1]"
    if p_up.dtype.kind not in "biuf":
        raise ValueError(probability_error)
    valid_probabilities = (p_up >= 0) & (p_up <= 1)
    if not valid_probabilities.all():
        raise ValueError(probability_error)

    threshold_error = "threshold must be a finite number in [0, 1]"
    if not isinstance(threshold, Real):
        raise ValueError(threshold_error)
    if not 0 <= threshold <= 1:
        raise ValueError(threshold_error)

    actual_up = y_true == 1
    predicted_up = p_up >= threshold
    true_positive = int(np.count_nonzero(actual_up & predicted_up))
    true_negative = int(np.count_nonzero(~actual_up & ~predicted_up))
    false_positive = int(np.count_nonzero(~actual_up & predicted_up))
    false_negative = int(np.count_nonzero(actual_up & ~predicted_up))

    return true_positive, true_negative, false_positive, false_negative


def classification_metrics(y_true: np.ndarray, p_up: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Return accuracy, positive-class F1, MCC, and ROC AUC.

    Inputs must be nonempty, equally sized 1-D arrays: binary labels and finite
    probabilities in [0, 1]. Missing labels must be removed by the caller.
    Predict up when p_up >= threshold; AUC uses probabilities, not predictions.
    Undefined metrics return NaN and emit a UserWarning: F1 when neither labels
    nor predictions contain positives, MCC when either is constant, and AUC
    when the labels contain only one class.
    """
    y_true = np.asarray(y_true)
    p_up = np.asarray(p_up)
    true_positive, true_negative, false_positive, false_negative = _confusion_counts(y_true, p_up, threshold)
    actual_up = y_true == 1

    accuracy = (true_positive + true_negative) / y_true.size
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = float("nan")
    if f1_denominator > 0:
        f1 = 2 * true_positive / f1_denominator
    else:
        warnings.warn("F1 is undefined: no actual or predicted positives", UserWarning, stacklevel=2)

    mcc_denominator = np.sqrt(
        (true_positive + false_positive) * (true_positive + false_negative)
        * (true_negative + false_positive) * (true_negative + false_negative)
    )
    mcc = float("nan")
    if mcc_denominator > 0:
        mcc = float((true_positive * true_negative - false_positive * false_negative) / mcc_denominator)
    else:
        warnings.warn("MCC is undefined: labels or predictions contain only one class", UserWarning, stacklevel=2)

    n_positive = true_positive + false_negative
    n_negative = true_negative + false_positive
    auc = float("nan")
    if n_positive > 0 and n_negative > 0:
        # Average ranks give tied positive/negative scores half credit in AUC.
        ranks = pd.Series(p_up).rank(method="average").to_numpy()
        auc = float((ranks[actual_up].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))
    else:
        warnings.warn("AUC is undefined: labels contain only one class", UserWarning, stacklevel=2)

    return {"accuracy": float(accuracy), "f1": f1, "mcc": mcc, "auc": auc}


def classification_details(
    y_true: np.ndarray, p_up: np.ndarray, threshold: float = 0.5,
) -> dict[str, int | float]:
    """Return TP, TN, FP, FN, precision, recall (sensitivity), and specificity.

    Use the same inputs and validation as classification_metrics: 1 means up,
    and p_up >= threshold predicts up. Counts are integers. A rate with a zero
    denominator returns NaN and emits a UserWarning, including single-class data.
    """
    true_positive, true_negative, false_positive, false_negative = _confusion_counts(y_true, p_up, threshold)
    rates = {}
    for name, numerator, denominator in (
        ("precision", true_positive, true_positive + false_positive),
        ("recall", true_positive, true_positive + false_negative),
        ("specificity", true_negative, true_negative + false_positive),
    ):
        if denominator > 0:
            rates[name] = numerator / denominator
        else:
            rates[name] = float("nan")
            warnings.warn(f"{name.capitalize()} is undefined: denominator is zero", UserWarning, stacklevel=2)

    return {
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        **rates,
    }
