"""Input guards shared by Data stages; the frozen contract remains in src.core."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.schema import conform, validate
from src.core.tickers import canonical_ticker


def invalid_ohlc(frame: pd.DataFrame) -> pd.Series:
    """Ignore only floating-point adjustment noise; never repair provider prices."""
    upper = frame[["open", "low", "close"]].max(axis=1)
    lower = frame[["open", "high", "close"]].min(axis=1)
    return ((frame.high < upper) & ~np.isclose(frame.high, upper, rtol=1e-12, atol=1e-12)) | (
        (frame.low > lower) & ~np.isclose(frame.low, lower, rtol=1e-12, atol=1e-12)
    )


def canonical_table(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    if out.ticker.isna().any():
        raise ValueError(f"{name}: null ticker")
    out["ticker"] = out.ticker.map(canonical_ticker).astype("string")
    if out.ticker.eq("").any():
        raise ValueError(f"{name}: empty ticker")
    out = conform(out, name)
    validate(out, name)
    return out


def checked_prices(prices: pd.DataFrame) -> pd.DataFrame:
    out = canonical_table(prices, "prices")
    fields = ["open", "high", "low", "close", "volume"]
    if not np.isfinite(out[fields].to_numpy()).all():
        raise ValueError("Non-finite OHLCV")
    if (out[["open", "high", "low", "close"]] <= 0).any(axis=None) or out.volume.lt(0).any():
        raise ValueError("Non-positive prices or negative volume")
    if invalid_ohlc(out).any():
        raise ValueError("Inconsistent OHLC range")
    return out


def checked_membership(membership: pd.DataFrame) -> pd.DataFrame:
    out = canonical_table(membership, "membership").sort_values(["ticker", "start"])
    if (out.end.notna() & out.end.lt(out.start)).any():
        raise ValueError("Membership end precedes start")
    for ticker, group in out.groupby("ticker"):
        ends = group.end.iloc[:-1].fillna(pd.Timestamp.max).to_numpy()
        if (ends >= group.start.iloc[1:].to_numpy()).any():
            raise ValueError(f"Overlapping membership spells for {ticker}")
    return out.reset_index(drop=True)
