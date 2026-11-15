"""Aggregate reviewed adjusted intraday bars into complete regular-session OHLCV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.tickers import canonical_ticker
from src.data.fetch_prices import RAW_COLUMNS, _bounds
from src.data.validation import invalid_ohlc


def regular_session_daily(
    bars: pd.DataFrame, ticker: str, *, bar_minutes: int, start: str, end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use NASDAQ open/close in UTC, including DST and shortened sessions.

    Input OHLC must ALREADY be split/dividend adjusted, with documented identity.
    Never infer adjustment from intraday prices. Reject incomplete session grids;
    do not forward-fill missing minutes or include pre/post-market information.
    Volume is the source's regular-session sum (not necessarily Yahoo's convention).
    """
    import pandas_market_calendars as mcal

    lower, upper = _bounds(start, end)
    ticker = canonical_ticker(ticker)
    if bar_minutes not in (1, 5, 15, 30):
        raise ValueError("Unsupported bar duration")
    required = {"ts", "open", "high", "low", "close", "volume"}
    if required - set(bars):
        raise ValueError(f"Intraday input needs {sorted(required)}")
    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars.ts)
    if bars.ts.isna().any() or bars.ts.dt.tz is None:
        raise ValueError("Intraday timestamps must be non-null and timezone-aware")
    bars["ts"] = bars.ts.dt.tz_convert("UTC")
    local = bars.ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    bars["date"] = local.dt.normalize().astype("datetime64[ns]")
    bars = bars.loc[bars.date.between(lower, upper, inclusive="left")].sort_values("ts")
    if bars.ts.duplicated().any():
        raise ValueError("Duplicate intraday timestamps")
    schedule = mcal.get_calendar("NASDAQ").schedule(start_date=lower, end_date=upper)
    schedule.index = schedule.index.astype("datetime64[ns]")
    bars = bars.join(schedule[["market_open", "market_close"]], on="date")
    bars = bars.loc[bars.ts.ge(bars.market_open) & bars.ts.lt(bars.market_close)]
    numeric = ["open", "high", "low", "close", "volume"]
    bars[numeric] = bars[numeric].astype("float64")
    if not np.isfinite(bars[numeric].to_numpy()).all() or bars[["open", "high", "low", "close"]].le(0).any(axis=None):
        raise ValueError("Invalid intraday OHLCV")
    if bars.volume.lt(0).any() or invalid_ohlc(bars).any():
        raise ValueError("Invalid intraday volume or OHLC range")
    duration = pd.Timedelta(minutes=bar_minutes)
    if ((bars.ts - bars.market_open) % duration).ne(pd.Timedelta(0)).any():
        raise ValueError("Intraday timestamps are off the regular-session bar grid")
    rows, reviews = [], []
    for date, group in bars.groupby("date"):
        opening, closing = group.market_open.iloc[0], group.market_close.iloc[0]
        expected = pd.date_range(opening, closing, freq=duration, inclusive="left")
        complete = pd.DatetimeIndex(group.ts).equals(expected)
        reviews.append({"date": date, "ticker": ticker, "bars": len(group), "expected_bars": len(expected),
                        "complete": complete, "volume": float(group.volume.sum())})
        if not complete:
            continue
        rows.append({"date": date, "ticker": ticker, "open": float(group.open.iloc[0]),
                     "high": float(group.high.max()), "low": float(group.low.min()),
                     "close": float(group.close.iloc[-1]), "volume": float(group.volume.sum())})
    daily = pd.DataFrame(rows, columns=RAW_COLUMNS)
    daily["date"] = pd.to_datetime(daily.date).astype("datetime64[ns]")
    daily["ticker"] = daily.ticker.astype("string")
    daily[numeric] = daily[numeric].astype("float64")
    return daily, pd.DataFrame(reviews, columns=["date", "ticker", "bars", "expected_bars", "complete", "volume"])
