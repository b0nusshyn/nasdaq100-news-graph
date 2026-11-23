"""Encode a frozen external news candidate using an existing local FinBERT snapshot."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.io import read_table
from src.nlp.encode_news import EMB_COLUMNS, MAX_LENGTH, MODEL_NAME, aggregate, encode_articles
from src.nlp.external_news_candidate import POLICIES
from src.nlp.rebuild_timed_news import END, START, file_hash


def validate_features(features, headlines):
    if not headlines.article_id.is_unique or not features.article_id.equals(headlines.article_id):
        raise ValueError("Encoder output must preserve the unique, ordered headline IDs")
    if list(features.columns) != ["article_id", "sentiment", "confidence", *EMB_COLUMNS]:
        raise ValueError("Unexpected FinBERT output schema")
    if not np.isfinite(features[["sentiment", "confidence", *EMB_COLUMNS]].to_numpy()).all():
        raise ValueError("Nonfinite FinBERT output")
    if not features.sentiment.between(-1, 1).all() or not features.confidence.between(0, 1).all():
        raise ValueError("FinBERT scores outside the expected range")


def aggregate_policies(features, links, prices, coverage):
    output = {}
    for policy in POLICIES:
        aligned = links.loc[links[policy].notna(), ["article_id", "ticker", policy]].rename(columns={policy: "date"})
        if aligned.duplicated(["article_id", "ticker"]).any():
            raise ValueError("Duplicate candidate article links")
        daily = aggregate(features, aligned, prices, START, END)
        joined = daily.merge(coverage, on=["date", "ticker"], how="outer", validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all() or not joined.n_articles.eq(joined[policy + "_count"]).all():
            raise ValueError("Adjusted daily output disagrees with the frozen population or article counts")
        output[policy] = joined.drop(columns="_merge")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--full", action="store_true", help="explicitly enable the full encoding job")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists")
    snap = args.snapshot
    if (not snap.is_dir() or snap.parent.name != "snapshots" or snap.parent.parent.name != "models--ProsusAI--finbert"
            or not re.fullmatch("[0-9a-f]{40}", snap.name)):
        parser.error("An existing local ProsusAI/finbert cache snapshot with full revision is required")
    try:
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        parser.error("Install transformers in an approved encoding environment")
    import torch

    root = args.candidate_dir
    source = root / "candidate_headlines.parquet"
    articles = pd.read_parquet(source, columns=["article_id", "title"])
    if not articles.article_id.is_unique or articles.title.isna().any():
        raise ValueError("Expected unique, non-null frozen candidate headlines")
    if not args.full:
        articles = articles.head(8)
    articles = articles.reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(snap, local_files_only=True)
    model = model.to(args.device).eval()
    start = time.perf_counter()
    features = encode_articles(articles.rename(columns={"title": "Article_title"}), batch_size=args.batch_size,
                               device=args.device, tokenizer=tokenizer, model=model)
    elapsed = time.perf_counter() - start
    validate_features(features, articles)
    manifest = {"model": MODEL_NAME, "revision": snap.name, "snapshot": str(snap), "headline_sha256": file_hash(source),
                "full": args.full, "seconds": elapsed, "article_rows": len(features), "max_tokens": MAX_LENGTH,
                "device": args.device, "batch_size": args.batch_size, "fp16": args.device.startswith("cuda"),
                "embedding": "final-layer CLS", "policy": "exploratory archive time proxy",
                "torch": str(torch.__version__), "transformers": transformers.__version__}
    daily = {}
    if args.full:
        audit_path = root / "link_audit.parquet"
        coverage_path = root / "ticker_day_coverage.parquet"
        links = pd.read_parquet(audit_path, columns=["article_id", "ticker", *POLICIES])
        coverage = pd.read_parquet(coverage_path)
        daily = aggregate_policies(features, links, read_table("prices"), coverage)
        manifest.update(link_audit_sha256=file_hash(audit_path), coverage_sha256=file_hash(coverage_path))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    features.to_parquet(args.output_dir / "article_features.parquet", index=False)
    for policy, frame in daily.items():
        frame.to_parquet(args.output_dir / f"{policy}_news.parquet", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
