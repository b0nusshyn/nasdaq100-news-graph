"""FinBERT sentiment + embeddings per article, aggregated to (date, ticker). Owner: NLP. Issues #23, #24.

Output: `news_features.parquet` (SCHEMA.md). GPU recommended (an 8 GB card is enough).

    python -m src.nlp.encode_news --input data/raw/fnspid_filtered.csv

Task 2.5 (this step) encodes each article once, by headline, and writes the per-article
features to `<data>/raw/article_features.parquet` (raw, outside SCHEMA.md): `article_id`,
`sentiment` (p_pos - p_neg), `confidence` (max class probability) and `emb_0..emb_767`
(final-layer [CLS]). Only articles present in `article_tickers` are encoded; an existing file is
reused unless `--reencode`. Task 2.6 (`aggregate`) then takes a plain mean per in-index
(date, ticker) in the study window and writes `news_features.parquet`; ticker-days without
news get `has_news = False`, `n_articles = 0` and zeros, so "no coverage" stays distinct.

Headlines, not bodies: every article has one (23% have no body), FinBERT was trained on short
financial sentences, and one text type keeps the features consistent across articles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.schema import EMB_DIM

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


def aggregate(article_features: pd.DataFrame, article_tickers: pd.DataFrame, prices: pd.DataFrame,
              start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """One row per in-index (date, ticker); ticker-days without news get has_news=False and zeros.

    Several articles on one ticker-day are combined with a plain mean (SCHEMA.md decision 3).
    `start`/`end` limit the rows to the study window (news was only collected for it); rows
    outside it are left out, which the models read as "no coverage" as well. Article-ticker
    rows that are not in-index days are ignored (survivorship: `in_index` is the filter).
    """
    keys = prices.loc[prices["in_index"], ["date", "ticker"]]
    if start is not None:
        keys = keys[keys["date"] >= pd.Timestamp(start)]
    if end is not None:
        keys = keys[keys["date"] <= pd.Timestamp(end)]
    keys = keys.drop_duplicates().sort_values(["date", "ticker"]).reset_index(drop=True)

    value_cols = ["sentiment", "confidence", *EMB_COLUMNS]
    feats = article_features.set_index("article_id")[value_cols]
    links = article_tickers[["article_id", "date", "ticker"]]
    links = links[links["article_id"].isin(feats.index)]
    missing = article_tickers["article_id"].nunique() - links["article_id"].nunique()
    if missing:
        raise ValueError(f"{missing} article_tickers articles have no features; rerun encoding")

    # Plain mean per (date, ticker): sort by group, sum with reduceat, divide by counts.
    links = links.merge(keys.reset_index().rename(columns={"index": "row"}), on=["date", "ticker"])
    links = links.sort_values("row", kind="stable")
    values = feats.loc[links["article_id"]].to_numpy(dtype=np.float64)
    rows = links["row"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]]) if len(rows) else np.array([], dtype=int)
    group_rows = rows[starts]
    counts = np.diff(np.r_[starts, len(rows)])

    out_values = np.zeros((len(keys), len(value_cols)), dtype=np.float32)
    if len(rows):
        out_values[group_rows] = (np.add.reduceat(values, starts, axis=0) / counts[:, None]).astype(np.float32)
    n_articles = np.zeros(len(keys), dtype=np.int32)
    n_articles[group_rows] = counts

    out = pd.DataFrame(out_values, columns=value_cols)
    out.insert(0, "n_articles", n_articles)
    out.insert(0, "has_news", n_articles > 0)
    out.insert(0, "ticker", keys["ticker"].astype("string").to_numpy())
    out.insert(0, "date", keys["date"].to_numpy())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="encode only the first N articles (smoke test)")
    p.add_argument("--reencode", action="store_true",
                   help="run FinBERT even if raw/article_features.parquet already exists")
    p.add_argument("--start", default="2019-01-01", help="first day of news_features rows")
    p.add_argument("--end", default="2023-12-31", help="last day of news_features rows")
    args = p.parse_args()

    from src.core.io import read_table, write_table
    from src.core.paths import raw_dir

    article_tickers = read_table("article_tickers")
    features_path = raw_dir() / "article_features.parquet"
    if args.reencode or args.limit or not features_path.exists():
        filtered = pd.read_csv(args.input, usecols=["Url", "Article_title"], dtype=str)
        arts = headlines(filtered, article_tickers["article_id"])
        if args.limit:
            arts = arts.head(args.limit)
        print(f"encoding {len(arts):,} headlines on {args.device}")
        feats = encode_articles(arts, batch_size=args.batch_size, device=args.device, progress=True)
        feats.to_parquet(features_path, index=False)
        print(f"wrote {len(feats):,} article features -> {features_path}")
        if args.limit:
            print("--limit is a smoke test: news_features.parquet not written")
            return
    else:
        print(f"using existing {features_path} (pass --reencode to run FinBERT again)")
        feats = pd.read_parquet(features_path)

    news = aggregate(feats, article_tickers, read_table("prices"), args.start, args.end)
    path = write_table(news, "news_features")
    print(f"{len(news):,} in-index ticker-days, {news['has_news'].mean():.1%} with news -> {path}")


if __name__ == "__main__":
    main()
