"""Long/short backtest from predictions, net of transaction costs. Owner: Evaluation. Issue #18."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series  # cumulative net return, indexed by date
    cumulative_return: float
    sharpe: float  # annualised, 252 trading days
    max_drawdown: float
    turnover: float


def backtest(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: int,
    cost_bps: float | None = None,
    *,
    cost_bps_per_side: float = 5.0,
    quantile: float = 0.2,
) -> BacktestResult:
    """Backtest a long/short strategy using predicted probabilities.
    Buy the top 20% and short the bottom 20% with equal weights.
    Hold positions for 1 or 5 trading days without overlapping periods.
    Calculate cumulative return, Sharpe ratio, and maximum drawdown
    after transaction costs.

    Labels must contain fwd_ret_1d or fwd_ret_5d for the requested horizon;
    ret_* columns are never used because they may contain trailing returns.

    Returns a BacktestResult containing portfolio performance.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon not in (1, 5):
        raise ValueError("horizon must be 1 or 5")

    # Keep the original positional cost argument compatible.
    if cost_bps is not None:
        if cost_bps_per_side != 5.0:
            raise ValueError("supply only one of cost_bps and cost_bps_per_side")
        cost_bps_per_side = cost_bps

    cost_error = "cost_bps_per_side must be finite and nonnegative"
    if not isinstance(cost_bps_per_side, Real):
        raise ValueError(cost_error)
    if not np.isfinite(cost_bps_per_side) or cost_bps_per_side < 0:
        raise ValueError(cost_error)
    if not isinstance(quantile, Real) or not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")

    return_column = f"fwd_ret_{horizon}d"

    for frame, name, required in (
        (predictions, "predictions", {"date", "ticker", "p_up", "horizon", "model"}),
        (labels, "labels", {"date", "ticker", return_column}),
    ):
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} requires columns {sorted(required)}")

        dates = frame["date"]
        if not pd.api.types.is_datetime64_ns_dtype(dates.dtype) or dates.dt.tz is not None:
            raise ValueError(f"{name} dates must be timezone-naive datetime64[ns]")
        if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
            raise ValueError(f"{name} dates must be nonmissing dates at midnight")
        if frame["ticker"].isna().any():
            raise ValueError(f"{name} tickers must not be missing")

    signals = predictions.loc[predictions["horizon"] == horizon].copy()
    if signals["model"].isna().any() or signals["model"].nunique() > 1:
        raise ValueError("backtest requires predictions for one model")

    for frame, name in ((signals, "predictions"), (labels, "labels")):
        if frame.duplicated(["date", "ticker"]).any():
            raise ValueError(f"{name} has duplicate date/ticker rows")

    probabilities = signals["p_up"]
    if not pd.api.types.is_numeric_dtype(probabilities) or not probabilities.between(0, 1).all():
        raise ValueError("p_up must contain finite probabilities in [0, 1]")

    if signals.empty:
        empty_dates = pd.DatetimeIndex([], name="date")
        empty_equity = pd.Series(dtype=float, index=empty_dates, name="equity")
        return BacktestResult(empty_equity, 0.0, float("nan"), 0.0, 0.0)

    calendar = pd.DatetimeIndex(labels["date"].unique()).sort_values()
    if not signals["date"].isin(calendar).all():
        raise ValueError("prediction dates must appear in the labels trading calendar")

    forward_returns = labels.set_index(["date", "ticker"])[return_column]
    signals_by_date = {date: group for date, group in signals.groupby("date")}
    first_entry = calendar.get_loc(signals["date"].min())
    last_entry = calendar.get_loc(signals["date"].max())
    equity_dates = [calendar[first_entry]]
    wealth = [1.0]
    net_returns = []
    turnover = 0.0
    round_trip_cost = 2 * cost_bps_per_side / 10_000

    for entry in range(first_entry, last_entry + 1, horizon):
        exit_position = entry + horizon
        if exit_position >= len(calendar):
            break

        date = calendar[entry]
        day_signals = signals_by_date.get(date)
        period_return = 0.0
        can_trade = (
            day_signals is not None
            and len(day_signals) >= 2
            and day_signals["p_up"].nunique() > 1
        )

        if can_trade:
            ranked = day_signals.sort_values(["p_up", "ticker"])
            side_count = max(1, int(len(ranked) * quantile))
            long_positions = ranked.tail(side_count)
            short_positions = ranked.head(side_count)
            selected = pd.concat([long_positions, short_positions])

            # Read outcomes only after choosing positions.
            selected_keys = pd.MultiIndex.from_frame(selected[["date", "ticker"]])
            returns = forward_returns.reindex(selected_keys).to_numpy(dtype=float)
            if not np.isfinite(returns).all():
                raise ValueError(f"selected trades on {date.date()} require finite {return_column} labels")
            if (returns < -1).any():
                raise ValueError("forward stock returns must be at least -1")

            long_return = returns[:side_count].mean()
            short_return = returns[side_count:].mean()
            gross_return = 0.5 * (long_return - short_return)
            period_return = gross_return - round_trip_cost
            turnover += 2.0

        if period_return <= -1:
            raise ValueError("portfolio lost all capital; subsequent compounding is undefined")
        net_returns.append(period_return)
        equity_dates.append(calendar[exit_position])
        wealth.append(wealth[-1] * (1 + period_return))

    wealth = np.array(wealth)
    equity_index = pd.DatetimeIndex(equity_dates, name="date")
    equity = pd.Series(wealth - 1, index=equity_index, name="equity")
    sharpe = float("nan")
    if len(net_returns) >= 2:
        volatility = np.std(net_returns, ddof=1)
        if volatility > 0:
            mean_return = np.mean(net_returns)
            annualization_factor = np.sqrt(252 / horizon)
            sharpe = float(mean_return / volatility * annualization_factor)

    running_peak = np.maximum.accumulate(wealth)
    drawdowns = 1 - wealth / running_peak
    return BacktestResult(
        equity=equity,
        cumulative_return=float(equity.iloc[-1]),
        sharpe=sharpe,
        max_drawdown=float(drawdowns.max()),
        turnover=turnover,
    )
