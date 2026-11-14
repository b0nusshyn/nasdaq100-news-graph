"""Walk-forward splitter with embargo. Owner: Evaluation. Issue #12.

Folds are contiguous in time; between each training window and its test window sits an
embargo of `embargo_days` trading days. It is 5, the longest horizon, for every horizon, so
no training label overlaps the test window and the 1-day and 5-day models share identical
folds. Settings come from configs/walk_forward.yaml, shared by every model.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import pandas as pd


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
    embargo_days: int | None = None,
) -> list[Fold]:
    """Return complete chronological folds, numbered from zero.

    `train_days` counts actual training days, followed by a separate embargo.
    The gap defaults to five trading days for both horizons, as configured in
    walk_forward.yaml. Test windows never overlap.
    Dates must be unique, sorted, timezone-naive trading days at midnight.
    Empty or insufficient data returns an empty list; a partial final test is omitted.
    """
    if step_days is None:
        step_days = test_days
    if embargo_days is None:
        embargo_days = 5

    for name, value in (
        ("train_days", train_days), ("test_days", test_days),
        ("horizon", horizon), ("step_days", step_days), ("embargo_days", embargo_days),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if horizon not in (1, 5):
        raise ValueError("horizon must be 1 or 5")
    if embargo_days < horizon:
        raise ValueError("embargo_days must be at least horizon")
    if step_days < test_days:
        raise ValueError("step_days must be at least test_days to avoid overlapping tests")
    if not isinstance(expanding, bool):
        raise ValueError("expanding must be a boolean")

    if not isinstance(trading_days, pd.DatetimeIndex):
        raise TypeError("trading_days must be a pandas DatetimeIndex")
    if trading_days.hasnans or not trading_days.is_unique or not trading_days.is_monotonic_increasing:
        raise ValueError("trading_days must be sorted, unique, and contain no missing dates")
    if trading_days.tz is not None or not trading_days.equals(trading_days.normalize()):
        raise ValueError("trading_days must be timezone-naive dates at midnight")

    folds = []
    first_test_start = train_days + embargo_days
    for test_start in range(first_test_start, 
        len(trading_days) - test_days + 1, step_days):
        # Count embargo days using the supplied trading calendar, not calendar days.
        train_end = test_start - embargo_days
        train_start = 0 if expanding else train_end - train_days
        folds.append(Fold(
            index=len(folds),
            train=trading_days[train_start:train_end],
            test=trading_days[test_start:test_start + test_days],
        ))
    return folds
