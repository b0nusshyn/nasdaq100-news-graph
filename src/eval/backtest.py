"""Long/short backtest from predictions, net of transaction costs. Owner: Evaluation. Issue #18."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.todo import todo


@dataclass
class BacktestResult:
    equity: pd.Series  # cumulative net return, indexed by date
    cumulative_return: float
    sharpe: float  # annualised, 252 trading days
    max_drawdown: float
    turnover: float


def backtest(
    predictions: pd.DataFrame, labels: pd.DataFrame, horizon: int, cost_bps_per_side: float = 5.0
) -> BacktestResult:
    """Each day go long the top quantile of p_up and short the bottom, hold `horizon` days,
    realise `fwd_ret_{horizon}d` from labels, charge `cost_bps_per_side` on every buy and every sell
    (README: 5 bps per side)."""
    todo(18, "backtest")
