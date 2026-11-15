"""FinBERT sentiment + embeddings per article, aggregated to (date, ticker). Owner: NLP. Issues #23, #24.

Output: `news_features.parquet` (SCHEMA.md). GPU recommended (an 8 GB card is enough).

    python -m src.nlp.encode_news --input data/raw/fnspid_filtered.csv

Task 2.5 (this step) encodes each article once, by headline, and writes the per-article
features to `<data>/raw/article_features.parquet` (raw, outside SCHEMA.md): `article_id`,
`sentiment` (p_pos - p_neg), `confidence` (max class probability) and `emb_0..emb_767`
(final-layer [CLS]). Only articles present in `article_tickers` are encoded. Task 2.6 then
averages them per (date, ticker) into `news_features.parquet`.

Headlines, not bodies: every article has one (23% have no body), FinBERT was trained on short
financial sentences, and one text type keeps the features consistent across articles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.schema import EMB_DIM
from src.core.todo import todo

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 128  # tokens; headlines are far shorter, this only guards outliers
EMB_COLUMNS = [f"emb_{i}" for i in range(EMB_DIM)]


def load_finbert(device: str = "cuda"):
    """Tokenizer and model in eval mode on `device`."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
    return tokenizer, model


def _label_index(model, name: str) -> int:
    for i, label in model.config.id2label.items():
        if str(label).lower() == name:
            return int(i)
    raise ValueError(f"model has no {name!r} label: {model.config.id2label}")


def encode(texts: list[str], batch_size: int = 64, device: str = "cuda", *,
           tokenizer=None, model=None, fp16: bool | None = None, progress: bool = False) -> pd.DataFrame:
    """Per text: sentiment (p_pos - p_neg), confidence (max prob), emb_0..emb_767 ([CLS]).

    Rows follow `texts` order. `tokenizer`/`model` default to FinBERT; pass others for tests.
    `fp16` defaults to True on CUDA (2.2 measured ~3x faster) and False on CPU.
    """
    import torch

    if model is None or tokenizer is None:
        tokenizer, model = load_finbert(device)
    fp16 = (device.startswith("cuda")) if fp16 is None else fp16
    pos, neg = _label_index(model, "positive"), _label_index(model, "negative")

    n = len(texts)
    sentiment = np.empty(n, dtype=np.float32)
    confidence = np.empty(n, dtype=np.float32)
    emb = np.empty((n, model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.float16, enabled=fp16):
        for start in range(0, n, batch_size):
            batch = [t if isinstance(t, str) else "" for t in texts[start:start + batch_size]]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH,
                            return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True)
            probs = out.logits.float().softmax(-1)
            end = start + len(batch)
            sentiment[start:end] = (probs[:, pos] - probs[:, neg]).cpu().numpy()
            confidence[start:end] = probs.max(-1).values.cpu().numpy()
            emb[start:end] = out.hidden_states[-1][:, 0].float().cpu().numpy()
            if progress and (start // batch_size) % 200 == 0:
                print(f"  {end:,} / {n:,}", flush=True)

    if emb.shape[1] != EMB_DIM:
        raise ValueError(f"embedding size {emb.shape[1]} != SCHEMA.md EMB_DIM {EMB_DIM}")
    feats = pd.DataFrame(emb, columns=EMB_COLUMNS)
    feats.insert(0, "confidence", confidence)
    feats.insert(0, "sentiment", sentiment)
    return feats


def headlines(filtered: pd.DataFrame, article_ids: pd.Series | None = None) -> pd.DataFrame:
    """One (article_id, Article_title) row per article, optionally limited to `article_ids`."""
    from src.nlp.match_tickers import article_id

    arts = filtered.dropna(subset=["Url"]).drop_duplicates("Url")
    arts = arts.assign(article_id=arts["Url"].map(article_id))[["article_id", "Article_title"]]
    if article_ids is not None:
        arts = arts[arts["article_id"].isin(set(article_ids))]
    return arts.drop_duplicates("article_id").reset_index(drop=True)


def encode_articles(arts: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """`encode` the headlines of `headlines()` output; returns article_id + features."""
    feats = encode(arts["Article_title"].fillna("").tolist(), **kwargs)
    feats.insert(0, "article_id", arts["article_id"].to_numpy())
    return feats


def aggregate(article_features: pd.DataFrame, article_tickers: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per in-index (date, ticker); ticker-days without news get has_news=False and zeros."""
    todo(24, "aggregate news features")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="encode only the first N articles (smoke test)")
    args = p.parse_args()

    from src.core.io import read_table
    from src.core.paths import raw_dir

    filtered = pd.read_csv(args.input, usecols=["Url", "Article_title"], dtype=str)
    arts = headlines(filtered, read_table("article_tickers")["article_id"])
    if args.limit:
        arts = arts.head(args.limit)
    print(f"encoding {len(arts):,} headlines on {args.device}")
    feats = encode_articles(arts, batch_size=args.batch_size, device=args.device, progress=True)
    out = raw_dir() / "article_features.parquet"
    feats.to_parquet(out, index=False)
    print(f"wrote {len(feats):,} article features -> {out}")
    print("aggregation to news_features.parquet is task 2.6 (issue #24), not implemented yet")


if __name__ == "__main__":
    main()
