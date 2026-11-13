"""Walk-forward splitter with embargo. Owner: Evaluation. Issue #12.

Folds are contiguous in time; between each training window and its test window sits an
embargo of `embargo_days` trading days. It is 5, the longest horizon, for every horizon, so
no training label overlaps the test window and the 1-day and 5-day models share identical
folds. Settings come from configs/walk_forward.yaml, shared by every model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.todo import todo


@dataclass(frozen=True)
class Fold:
    index: int
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


def walk_forward(
    trading_days: pd.DatetimeIndex,
    *,
    train_days: int,
    test_days: int,
    embargo_days: int,
    step_days: int | None = None,
    expanding: bool = False,
) -> list[Fold]:
    """Folds over `trading_days`; the `embargo_days` days before each test window belong to neither split."""
    todo(12, "walk-forward splitter")
