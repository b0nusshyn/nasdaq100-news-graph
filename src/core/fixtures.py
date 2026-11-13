"""Synthetic, mutually consistent versions of every SCHEMA.md artifact.

Lets each stage be built and tested before its upstream artifact exists:

    python -m src.core.fixtures --out tests/fixtures
    NDX_DATA_DIR=tests/fixtures python -m src.models.train --config configs/gat_lstm.yaml

The data is random; only shapes, keys, dtypes and edge cases are meaningful. Edge cases
included on purpose: a ticker that joins the index mid-sample (`FFF`), ticker-days without
news, and trailing NaN labels where the horizon runs past the last date.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.io import write_table
from src.core.paths import FIXTURES_DIR
from src.core.schema import ALL_FOLDS, EMB_COLS, EMB_DIM, HORIZONS

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
SECTORS = {"AAA": "tech", "BBB": "tech", "CCC": "tech", "DDD": "health", "EEE": "health", "FFF": "health"}
LATE_JOINER = "FFF"


def make_fixtures(n_days: int = 60, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    join_date = dates[n_days // 2]

    membership = pd.DataFrame({
        "ticker": TICKERS,
        "start": [pd.Timestamp("2015-01-02")] * (len(TICKERS) - 1) + [join_date],
        "end": [pd.NaT] * len(TICKERS),
    })

    rows = []
    for t in TICKERS:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        open_ = close * (1 + rng.normal(0, 0.005, n_days))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": t,
            "open": open_,
            "high": np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n_days)),
            "low": np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n_days)),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float),
            "in_index": dates >= join_date if t == LATE_JOINER else True,
        }))
    prices = pd.concat(rows, ignore_index=True)

    labels = prices[["date", "ticker"]].copy()
    by_ticker = prices.groupby("ticker")["close"]
    for h in HORIZONS:
        ret = by_ticker.shift(-h) / prices["close"] - 1
        labels[f"ret_{h}d"] = ret
        labels[f"y_{h}d"] = pd.array(np.where(ret.isna(), pd.NA, (ret > 0).astype(int)), dtype="Int8")

    members = prices.loc[prices["in_index"], ["date", "ticker"]].reset_index(drop=True)
    n_articles = rng.poisson(0.5, len(members)).astype("int32")
    has_news = n_articles > 0
    emb = rng.normal(0, 1, (len(members), EMB_DIM)).astype("float32") * has_news[:, None]
    news = pd.concat([
        members.assign(
            has_news=has_news,
            n_articles=n_articles,
            sentiment=np.where(has_news, rng.uniform(-1, 1, len(members)), 0).astype("float32"),
            confidence=np.where(has_news, rng.uniform(0.5, 1, len(members)), 0).astype("float32"),
        ),
        pd.DataFrame(emb, columns=list(EMB_COLS)),
    ], axis=1)

    article_rows = []
    for i, (date, ticker, n) in enumerate(zip(members["date"], members["ticker"], n_articles, strict=True)):
        for j in range(n):
            aid = f"a{i:05d}_{j}"
            article_rows.append((aid, date, ticker))
            if rng.random() < 0.3:  # some articles co-mention a second firm
                other = rng.choice([t for t in TICKERS if t != ticker])
                article_rows.append((aid, date, other))
    article_tickers = pd.DataFrame(article_rows, columns=["article_id", "date", "ticker"])

    edges = []
    for a, b in itertools.permutations(TICKERS, 2):
        if SECTORS[a] == SECTORS[b]:
            edges.append((ALL_FOLDS, a, b, "sector", 1.0))
    pairs = article_tickers.merge(article_tickers, on="article_id").query("ticker_x != ticker_y")
    for (a, b), n in pairs.groupby(["ticker_x", "ticker_y"]).size().items():
        edges.append((0, a, b, "comention", float(n)))
    rets = prices.pivot(index="date", columns="ticker", values="close").pct_change().corr()
    for a, b in itertools.permutations(TICKERS, 2):
        edges.append((0, a, b, "corr", float(rets.loc[a, b])))
    edges = pd.DataFrame(edges, columns=["fold", "src", "dst", "edge_type", "weight"])

    test = labels.dropna(subset=["y_1d"]).tail(len(TICKERS) * 10)
    predictions = pd.DataFrame({
        "model": "random", "horizon": 1, "fold": 0,
        "date": test["date"].values, "ticker": test["ticker"].values,
        "p_up": rng.uniform(0, 1, len(test)), "y_true": test["y_1d"].astype(int).values,
    })

    return {
        "membership": membership, "prices": prices, "labels": labels,
        "article_tickers": article_tickers, "news_features": news,
        "edges": edges, "predictions": predictions,
    }


def write_fixtures(out_dir: Path = FIXTURES_DIR, **kwargs) -> list[Path]:
    written = []
    for name, df in make_fixtures(**kwargs).items():
        sub = Path("predictions") / "random.parquet" if name == "predictions" else Path(f"{name}.parquet")
        written.append(write_table(df, name, out_dir / sub))
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=FIXTURES_DIR)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for path in write_fixtures(args.out, n_days=args.days, seed=args.seed):
        print(path)


if __name__ == "__main__":
    main()
