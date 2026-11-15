"""One trading day's model input: per-company feature windows plus the company graph. Owner: Modelling. Issue #9.

Only companies in the index on that day become nodes (survivorship), and only edges between
two of them, for this fold or valid in every fold, are kept. Features use data up to the
day's close: a trailing return and that day's news, so nothing from after the prediction
time enters.

Skeleton feature set until Data's indicators (#19) land: trailing 1-day return plus the
scalar news features. The FinBERT embedding and its projection are added in the training
loop (#29).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from src.core.schema import ALL_FOLDS

NEWS_COLS = ["has_news", "sentiment", "confidence"]
FEATURES = ["ret", *NEWS_COLS]  # ret = close[t] / close[t-1] - 1, trailing, not the label


@dataclass(frozen=True)
class Snapshot:
    tickers: list[str]  # node i is tickers[i]
    x: torch.Tensor  # [N, window, len(FEATURES)]
    edge_index: torch.Tensor  # [2, E] (src, dst) node indices
    edge_type: torch.Tensor  # [E] index into edge_types
    edge_weight: torch.Tensor  # [E]


def build_snapshot(
    date,
    prices: pd.DataFrame,
    news_features: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    fold: int,
    window: int,
    edge_types: tuple[str, ...],
) -> Snapshot:
    date = pd.Timestamp(date)
    tickers = sorted(prices.loc[(prices["date"] == date) & prices["in_index"], "ticker"])
    days = sorted(prices.loc[prices["date"] <= date, "date"].unique())[-window:]
    if len(days) < window:
        raise ValueError(f"{date.date()}: only {len(days)} trading days of history, window is {window}")

    p = prices[prices["ticker"].isin(tickers)].sort_values(["ticker", "date"])
    p = p.assign(ret=p.groupby("ticker")["close"].pct_change())
    grid = pd.MultiIndex.from_product([tickers, days], names=["ticker", "date"])
    feats = p.set_index(["ticker", "date"])[["ret"]].reindex(grid)
    # News rows exist only for in-index days; a missing row means no coverage, not neutral news.
    news = news_features.set_index(["ticker", "date"])[NEWS_COLS].reindex(grid)
    news = news.fillna({"has_news": False, "sentiment": 0.0, "confidence": 0.0})
    feats = feats.join(news.astype("float32"))
    if feats.isna().any().any():
        raise ValueError(f"{date.date()}: missing prices inside the {window}-day window")
    x = torch.tensor(feats[FEATURES].to_numpy("float32").reshape(len(tickers), window, len(FEATURES)))

    node = {t: i for i, t in enumerate(tickers)}
    e = edges[
        edges["fold"].isin([fold, ALL_FOLDS])
        & edges["edge_type"].isin(edge_types)
        & edges["src"].isin(node)
        & edges["dst"].isin(node)
    ]
    return Snapshot(
        tickers=tickers,
        x=x,
        edge_index=torch.tensor([e["src"].map(node).tolist(), e["dst"].map(node).tolist()], dtype=torch.long),
        edge_type=torch.tensor(e["edge_type"].map(edge_types.index).tolist(), dtype=torch.long),
        edge_weight=torch.tensor(e["weight"].to_numpy("float32")),
    )
