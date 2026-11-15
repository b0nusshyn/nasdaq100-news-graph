"""Causal technical indicators, computed on the fly (Data, issue #19).

No global standardisation is fitted here. Model scaling must fit training folds only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validation import checked_prices

FEATURES = ("ret_1d", "volatility_20d", "rsi_14", "macd", "macd_signal", "volume_z_20d")


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Return one row per input row, preserving order and warm-up NaNs.

    Volatility: 20-return sample SD (not annualised). RSI: Wilder's initial
    14-period mean followed by smoothing with alpha=1/14. MACD: EMA12-EMA26,
    EMA9 signal, adjust=False. Volume z-score: trailing 20 observations, ddof=1.
    Flat RSI is 50; constant-volume z-score is 0 once the window is available.
    Windows count observed rows; missing sessions must be reviewed via quality_report.
    """
    frame = checked_prices(prices)
    out = frame[["date", "ticker"]].copy()
    for col in FEATURES:
        out[col] = float("nan")
    for _, group in frame.sort_values(["ticker", "date"]).groupby("ticker"):
        close, volume = group.close, group.volume
        returns = close.pct_change(fill_method=None)
        out.loc[group.index, "ret_1d"] = returns
        out.loc[group.index, "volatility_20d"] = returns.rolling(20, min_periods=20).std(ddof=1)
        changes = close.diff()
        gain, loss = changes.clip(lower=0), -changes.clip(upper=0)
        # Seed Wilder smoothing with the arithmetic mean of the first 14 deltas.
        for series in (gain, loss):
            if len(series) > 14:
                series.iloc[14] = series.iloc[1:15].mean()
            series.iloc[:14] = np.nan
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, ignore_na=True).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, ignore_na=True).mean()
        rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
        rsi = rsi.mask(avg_loss.eq(0) & avg_gain.gt(0), 100)
        rsi = rsi.mask(avg_loss.eq(0) & avg_gain.eq(0), 50)
        out.loc[group.index, "rsi_14"] = rsi
        macd = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(
            span=26, adjust=False, min_periods=26
        ).mean()
        out.loc[group.index, "macd"] = macd
        out.loc[group.index, "macd_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        mean = volume.rolling(20, min_periods=20).mean()
        std = volume.rolling(20, min_periods=20).std(ddof=1)
        z = ((volume - mean) / std.replace(0, np.nan)).mask(std.eq(0), 0)
        out.loc[group.index, "volume_z_20d"] = z
    return out
