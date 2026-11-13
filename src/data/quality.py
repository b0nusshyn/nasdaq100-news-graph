"""Data-quality report on prices and labels. Owner: Data. Issue #20.

Bias guards that must always hold are tests in tests/test_bias.py (#26), not report lines.
"""

from __future__ import annotations

import pandas as pd

from src.core.todo import todo


def quality_report(prices: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per finding: missing trading days, suspicious jumps (splits), class balance per year."""
    todo(20, "data-quality report")
