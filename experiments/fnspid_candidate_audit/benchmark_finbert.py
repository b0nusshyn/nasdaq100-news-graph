"""Benchmark warmed local FinBERT inference on sampled real candidate headlines."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.io import read_table
from src.nlp.encode_external_candidate import aggregate_policies, validate_features
from src.nlp.encode_news import MAX_LENGTH, encode_articles
from src.nlp.external_news_candidate import POLICIES


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    tok = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
    ).eval()
    root = args.candidate_dir
    all_articles = pd.read_parquet(root / "candidate_headlines.parquet", columns=["article_id", "title"])
    sample = all_articles.sample(n=768, random_state=42).reset_index(drop=True)
    lengths = [len(ids) for ids in tok(sample.title.tolist(), truncation=True, max_length=MAX_LENGTH)["input_ids"]]
    assert min(lengths) >= 2 and max(lengths) <= MAX_LENGTH
    assert tok.cls_token_id is not None and tok.sep_token_id is not None
    batch = tok(sample.title.head(64).tolist(), truncation=True, max_length=MAX_LENGTH, padding=True,
                return_tensors="pt")
    assert (batch.input_ids[:, 0] == tok.cls_token_id).all()
    last = batch.attention_mask.sum(1) - 1
    assert (batch.input_ids[torch.arange(len(last)), last] == tok.sep_token_id).all()

    def encode(frame):
        return encode_articles(frame.rename(columns={"title": "Article_title"}), batch_size=64,
                               device="cpu", tokenizer=tok, model=model)

    encode(sample.head(64))
    repeats = []
    for _ in range(2):
        start = time.perf_counter()
        features = encode(sample)
        repeats.append(time.perf_counter() - start)
        validate_features(features, sample)
    peak_inference = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    smoke = pd.read_parquet(args.smoke_dir / "article_features.parquet")
    again = encode(all_articles.head(8))
    validate_features(smoke, all_articles.head(8))
    np.testing.assert_allclose(smoke.iloc[:, 1:], again.iloc[:, 1:], atol=1e-6, rtol=1e-5)
    max_diff = float(np.abs(smoke.iloc[:, 1:] - again.iloc[:, 1:]).to_numpy().max())
    links = pd.read_parquet(root / "link_audit.parquet", columns=["article_id", "ticker", *POLICIES])
    kept_links = links[links.article_id.isin(features.article_id)]
    assert set(features.article_id) <= set(kept_links.article_id)
    aligned = links[list(POLICIES)].notna().any(axis=1)
    assert set(links.loc[aligned, "article_id"]) == set(all_articles.article_id)
    frozen = pd.read_parquet(root / "ticker_day_coverage.parquet")
    coverage = frozen.copy()
    population = pd.MultiIndex.from_frame(frozen[["date", "ticker"]])
    for pol in POLICIES:
        for rows, frame, key in ((links, frozen, pol + "_count"), (kept_links, coverage, pol + "_count")):
            counts = rows.dropna(subset=[pol]).groupby([pol, "ticker"]).size()
            aligned_counts = counts.reindex(population, fill_value=0).to_numpy()
            if rows is links:
                np.testing.assert_array_equal(frame[key], aligned_counts)
            else:
                frame[key] = aligned_counts
    daily = aggregate_policies(features, kept_links, read_table("prices"), coverage)
    rate = len(sample) / float(np.median(repeats))
    overlay = Path("data/raw/finbert_local/python")
    result = {
        "revision": args.snapshot.name, "snapshot_file_hashes": {
            p.name: sha256(p) for p in sorted(args.snapshot.iterdir()) if p.is_file()},
        "model_config": json.loads((args.snapshot / "config.json").read_text()),
        "tokenizer_file_config": json.loads((args.snapshot / "tokenizer_config.json").read_text()),
        "tokenizer": {"class": type(tok).__name__, "fast": tok.is_fast, "vocab_size": tok.vocab_size,
                      "model_max_length": tok.model_max_length, "encoding_max_tokens": MAX_LENGTH,
                      "padding_side": tok.padding_side, "special_tokens": tok.special_tokens_map},
        "overlay_versions": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions(path=[str(overlay)])
        },
        "effective_versions": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "huggingface-hub", "tokenizers", "safetensors", "numpy", "pandas", "pyarrow")
        },
        "device": "cpu", "fp16": False, "batch": 64, "threads": 4, "warmup_headlines": 64,
        "platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__),
        "sample_seed": 42, "sample_headlines": len(sample),
        "token_quantiles": np.quantile(lengths, [0, .5, .95, 1]).tolist(),
        "seconds_by_trial": repeats, "headlines_per_second": rate,
        "estimated_full_inference_seconds": len(all_articles) / rate,
        "inference_peak_rss_bytes_macos": peak_inference,
        "peak_rss_bytes_macos_with_sample_aggregations": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "disk_free_bytes": shutil.disk_usage(args.output.parent).free,
        "independent_smoke_max_abs_delta": max_diff,
        "sample_aggregation": {pol: {"ticker_days": len(frame), "covered": int(frame.has_news.sum())}
                               for pol, frame in daily.items()},
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
