"""Classification metrics. Owner: Evaluation. Issue #13."""

from __future__ import annotations

import numpy as np

from src.core.todo import todo

METRICS = ("accuracy", "f1", "mcc", "auc")


def classification_metrics(y_true: np.ndarray, p_up: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Return {accuracy, f1, mcc, auc}."""
    todo(13, "classification metrics")
