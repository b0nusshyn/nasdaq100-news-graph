"""Walk-forward splitter with embargo. Owner: Evaluation. Issue #12.

Folds are contiguous in time; between each training window and its test window sits an
embargo of `horizon` trading days so no training label overlaps the test window. Settings
come from configs/walk_forward.yaml, shared by every model.
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
    horizon: int,
    step_days: int | None = None,
    expanding: bool = False,
) -> list[Fold]:
    """Folds over `trading_days`; the last `horizon` days before each test window are dropped from train."""
    todo(12, "walk-forward splitter")
