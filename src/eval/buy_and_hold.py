"""Long-only, equal-initial-capital benchmark, separate from probability ranking."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from src.eval.backtest import BacktestResult


def buy_and_hold_benchmark(
    prices: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    *,
    cost_bps_per_side: float = 5.0,
) -> BacktestResult:
    """Buy first-day members at adjusted close; hold fixed units with drifting weights.

    Supply every session of one fold, including its final liquidation close.
    Sell on the first observed nonmember close; retain cash and never add entrants.
    Each call starts fresh capital. Halts/acquisitions without an exit quote
    require a reviewed liquidation policy; missing prices or membership raise.
    """
    if not isinstance(trading_days, pd.DatetimeIndex) or len(trading_days) < 2:
        raise ValueError("supply at least two trading sessions")
    if trading_days.hasnans or not trading_days.is_unique or not trading_days.is_monotonic_increasing:
        raise ValueError("trading_days must be sorted, unique and nonmissing")
    if trading_days.tz is not None or not trading_days.equals(trading_days.normalize()):
        raise ValueError("trading_days must be timezone-naive dates at midnight")
    if not isinstance(cost_bps_per_side, Real) or not np.isfinite(cost_bps_per_side):
        raise ValueError("cost_bps_per_side must be finite")
    if not 0 <= cost_bps_per_side < 10_000:
        raise ValueError("cost_bps_per_side must be in [0, 10000)")
    if not {"date", "ticker", "close", "in_index"}.issubset(prices.columns):
        raise ValueError("prices require date, ticker, close and in_index")
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("prices must have unique date/ticker rows")

    panel = prices.set_index(["date", "ticker"])
    if trading_days[0] not in panel.index.get_level_values("date"):
        raise ValueError("missing first-session prices")
    first_day = panel.xs(trading_days[0], level="date")
    if first_day["in_index"].isna().any():
        raise ValueError("missing initial membership")
    basket = first_day.index[first_day["in_index"]]
    if basket.empty:
        raise ValueError("no eligible initial holdings")
    initial_prices = first_day.loc[basket, "close"]
    if not (np.isfinite(initial_prices) & initial_prices.gt(0)).all():
        raise ValueError("initial holdings require finite positive closes")

    cost = cost_bps_per_side / 10_000
    # Capital covers both purchases and their fees, so no borrowing is needed.
    invested = 1 / (1 + cost)
    units = invested / len(basket) / initial_prices
    cash = 0.0
    turnover = invested
    wealth = []
    for date in trading_days:
        if units.empty:
            wealth.append(cash)
            continue
        keys = pd.MultiIndex.from_product([[date], units.index], names=["date", "ticker"])
        holdings = panel.reindex(keys)
        close = holdings["close"].to_numpy(dtype=float)
        invalid_prices = ~np.isfinite(close) | (close <= 0)
        missing_membership = holdings["in_index"].isna().to_numpy()
        invalid_holdings = invalid_prices | missing_membership
        if invalid_holdings.any():
            affected_tickers = units.index[invalid_holdings].tolist()
            raise ValueError(
                f"holdings on {date.date()} require prices and membership for {affected_tickers}; "
                "no zero-return filling or assumed liquidation proceeds"
            )
        values = units.to_numpy() * close
        sell = ~holdings["in_index"].to_numpy(dtype=bool)
        if date == trading_days[-1]:
            sell[:] = True
        proceeds = float(values[sell].sum())
        cash += proceeds * (1 - cost)
        turnover += proceeds
        units = units.iloc[np.flatnonzero(~sell)]
        wealth.append(cash + float(values[~sell].sum()))

    wealth = np.asarray(wealth)
    equity = pd.Series(wealth - 1, index=trading_days.rename("date"), name="equity")
    # Include entry fees in the first holding-period return, not as an extra day.
    previous_wealth = wealth[:-1].copy()
    previous_wealth[0] = 1.0
    returns = wealth[1:] / previous_wealth - 1
    volatility = returns.std(ddof=1) if len(returns) >= 2 else 0.0
    sharpe = float(returns.mean() / volatility * np.sqrt(252)) if volatility > 0 else float("nan")
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return BacktestResult(
        equity=equity,
        cumulative_return=float(equity.iloc[-1]),
        sharpe=sharpe,
        max_drawdown=float((1 - wealth / peaks).max()),
        turnover=turnover,
    )
