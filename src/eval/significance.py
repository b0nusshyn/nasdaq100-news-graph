"""Is the proposed model better than Baseline 6? Owner: Evaluation. Issue #44."""

from __future__ import annotations

import pandas as pd

from src.core.todo import todo


def paired_bootstrap(
    a: pd.DataFrame, b: pd.DataFrame, metric: str = "mcc", n_boot: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Resample dates (block bootstrap) over predictions `a` and `b` on identical (date, ticker) rows.
    Return {delta, ci_low, ci_high, p_value} for metric(a) - metric(b)."""
    todo(44, "paired bootstrap")
