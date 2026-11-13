"""FinBERT sentiment + embeddings per article, aggregated to (date, ticker). Owner: NLP. Issues #23, #24.

Output: `news_features.parquet` (SCHEMA.md). GPU recommended (Colab).

    python -m src.nlp.encode_news --input data/raw/fnspid_filtered.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.core.todo import todo

MODEL_NAME = "ProsusAI/finbert"


def encode(texts: list[str], batch_size: int = 64, device: str = "cuda") -> pd.DataFrame:
    """Per text: sentiment (p_pos - p_neg), confidence (max prob), emb_0..emb_767 ([CLS])."""
    todo(23, "FinBERT encoding")


def aggregate(article_features: pd.DataFrame, article_tickers: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per in-index (date, ticker); ticker-days without news get has_news=False and zeros."""
    todo(24, "aggregate news features")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.parse_args()
    todo(23, "encode_news")


if __name__ == "__main__":
    main()
