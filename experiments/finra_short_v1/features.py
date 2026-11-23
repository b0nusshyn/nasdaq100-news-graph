"""Point-in-time short-volume features from cached FINRA daily files, plus a univariate screen.

FINRA posts day t's file after the close, so every feature at decision date t uses files of
days <= t-1 only (shift by one session per ticker).

  short_ratio_1   ShortVolume/TotalVolume of the previous session
  short_ratio_5   mean of the previous 5 sessions' ratios
  short_abn       short_ratio_5 minus the ticker's mean ratio over the previous 60 sessions

Direction hypothesis (Diether, Lee & Werner 2009): more shorting -> lower future returns, so the
screen scores the negated feature. Screening years: 2019-2021 only.

    .venv/bin/python -B -m experiments.finra_short_v1.features
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.tickers import canonical_ticker

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FEATURES = ["short_ratio_1", "short_ratio_5", "short_abn"]


def load_days(days_dir: Path = HERE / "days") -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(p) for p in sorted(days_dir.glob("*.parquet"))], ignore_index=True)
    df["date"] = pd.to_datetime(df.Date, format="%Y%m%d")
    df["ticker"] = df.Symbol.map(canonical_ticker)
    df = df.loc[df.TotalVolume > 0]
    df = df.groupby(["ticker", "date"], as_index=False)[["ShortVolume", "TotalVolume"]].sum()
    df["ratio"] = df.ShortVolume / df.TotalVolume
    return df


def build(prices: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """One row per price key; features use FINRA data up to the previous session only."""
    k = prices[["ticker", "date"]].assign(ticker=lambda x: x.ticker.astype(str))
    k = k.merge(raw[["ticker", "date", "ratio"]], on=["ticker", "date"], how="left")
    k = k.sort_values(["ticker", "date"])
    lag = k.groupby("ticker").ratio.shift(1)  # day t-1 file, available at decision time t
    g = lag.groupby(k.ticker)
    k["short_ratio_1"] = lag
    k["short_ratio_5"] = g.transform(lambda s: s.rolling(5, min_periods=3).mean())
    base = g.transform(lambda s: s.rolling(60, min_periods=40).mean())
    k["short_abn"] = k.short_ratio_5 - base
    return k.drop(columns="ratio")


def within_day_auc(d: pd.DataFrame, col: str, label: str = "y_5d") -> float:
    from sklearn.metrics import roc_auc_score
    a = [roc_auc_score(x[label], x[col]) for _, x in d.dropna(subset=[col]).groupby("date")
         if x[label].nunique() == 2 and len(x) > 20]
    return float(np.mean(a))


def main():
    prices = pd.read_parquet(REPO / "data/prices.parquet", columns=["ticker", "date", "in_index"])
    prices["ticker"] = prices.ticker.astype(str)
    lb = pd.read_parquet(REPO / "data/labels.parquet", columns=["ticker", "date", "y_1d", "y_5d"])
    lb["ticker"] = lb.ticker.astype(str)
    f = build(prices, load_days())
    f.to_parquet(HERE / "features.parquet", index=False)
    d = prices.loc[prices.in_index].merge(f, on=["ticker", "date"]).merge(lb, on=["ticker", "date"])
    rows = []
    for y, g in d.groupby(d.date.dt.year):
        if y < 2019:
            continue
        row = {"year": y, "coverage": round(g.short_abn.notna().mean(), 3)}
        for c in FEATURES:
            neg = g.assign(**{c: -g[c]})
            row[f"{c}_5d"] = round(within_day_auc(neg.dropna(subset=["y_5d"]), c), 4)
            row[f"{c}_1d"] = round(within_day_auc(neg.dropna(subset=["y_1d"]), c, "y_1d"), 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "screen.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
