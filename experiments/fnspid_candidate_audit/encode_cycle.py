"""Resume pinned candidate encoding and build two validated compressed daily tables."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.io import read_table
from src.nlp.encode_external_candidate import validate_features
from src.nlp.encode_news import EMB_COLUMNS, MAX_LENGTH, encode_articles
from src.nlp.external_news_candidate import POLICIES
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "reconstruction_v1"
REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
SNAPSHOT = Path("data/raw/finbert_local/models--ProsusAI--finbert/snapshots") / REVISION
CHUNK = 4096
VALUES = ["sentiment", "confidence", *EMB_COLUMNS]
MIN_FREE = 2 * 1024**3


def guard(path, reserve=0):
    if shutil.disk_usage(path).free < MIN_FREE + reserve:
        raise RuntimeError("Insufficient disk headroom above the 2 GiB safety margin")


def write_table(frame, path):
    guard(path.parent, int(frame.memory_usage(deep=True).sum()) + 16 * 1024**2)
    temp = path.with_suffix(".tmp")
    frame.to_parquet(temp, compression="zstd", index=False)
    guard(path.parent)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "encoded_cycle_v1")
    args = parser.parse_args()
    out = args.output
    out.mkdir(exist_ok=True)
    guard(out)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    headlines_path = CANDIDATE / "candidate_headlines.parquet"
    links_path = CANDIDATE / "link_audit.parquet"
    coverage_path = CANDIDATE / "ticker_day_coverage.parquet"
    articles = pd.read_parquet(headlines_path, columns=["article_id", "title"])
    if len(articles) != 156452 or not articles.article_id.is_unique or articles.title.isna().any():
        raise ValueError("Unexpected candidate encoding population")
    spec = {
        "revision": REVISION, "model": "ProsusAI/finbert", "torch": str(torch.__version__),
        "transformers": transformers.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "batch_size": 64, "threads": 4, "device": "cpu", "dtype": "float32", "max_tokens": MAX_LENGTH,
        "embedding": "final-layer CLS", "chunk": CHUNK, "rows": len(articles),
        "source_hashes": {str(p): file_hash(p) for p in [headlines_path, links_path, coverage_path, Path(__file__),
                                                       *sorted(SNAPSHOT.iterdir())]},
    }
    plan = out / "encoding_plan.json"
    if plan.exists():
        if json.loads(plan.read_text()) != spec:
            raise ValueError("Resume fingerprint mismatch")
    else:
        plan.write_text(json.dumps(spec, indent=2) + "\n")
    if (out / "manifest.json").exists():
        raise ValueError("Encoding already complete")
    chunk_dir = out / "article_features"
    chunk_dir.mkdir(exist_ok=True)
    model = tokenizer = None
    start = time.monotonic()
    chunk_paths = []
    for i in range(0, len(articles), CHUNK):
        subset = articles.iloc[i:i + CHUNK].reset_index(drop=True)
        target = chunk_dir / f"part-{i // CHUNK:04d}.parquet"
        if target.exists():
            features = pd.read_parquet(target)
        else:
            guard(out, 64 * 1024**2)
            if model is None:
                tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
                model = AutoModelForSequenceClassification.from_pretrained(
                    SNAPSHOT, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
            features = encode_articles(subset.rename(columns={"title": "Article_title"}), device="cpu", batch_size=64,
                                       tokenizer=tokenizer, model=model)
            validate_features(features, subset)
            write_table(features, target)
        validate_features(features, subset)
        chunk_paths.append(target)
        print(f"Validated {i + len(subset)}/{len(articles)}; {time.monotonic()-start:.0f}s", flush=True)
    del model, tokenizer, features
    links = pd.read_parquet(links_path, columns=["article_id", "ticker", *POLICIES])
    coverage = pd.read_parquet(coverage_path)
    keys = pd.MultiIndex.from_frame(coverage[["date", "ticker"]])
    prices = read_table("prices", columns=["date", "ticker", "in_index"])
    expected = pd.MultiIndex.from_frame(prices.loc[
        prices.in_index & prices.date.between("2019-01-01", "2023-12-31"), ["date", "ticker"]])
    assert len(keys) == len(expected) == 128519 and keys.is_unique and keys.isin(expected).all()
    outputs = {}
    for policy in POLICIES:
        mapped = links.loc[links[policy].notna(), ["article_id", "ticker", policy]]
        assert not mapped.duplicated(["article_id", "ticker"]).any()
        mapped = mapped.assign(row=keys.get_indexer(pd.MultiIndex.from_frame(mapped[[policy, "ticker"]])))
        mapped = mapped[mapped.row.ge(0)]
        sums = np.zeros((len(keys), len(VALUES)), dtype=np.float64)
        counts = np.zeros(len(keys), dtype=np.int32)
        for part in chunk_paths:
            frame = pd.read_parquet(part).set_index("article_id")
            use = mapped[mapped.article_id.isin(frame.index)]
            np.add.at(sums, use.row.to_numpy(), frame.loc[use.article_id, VALUES].to_numpy())
            np.add.at(counts, use.row.to_numpy(), 1)
        np.testing.assert_array_equal(counts, coverage[policy + "_count"])
        np.divide(sums, counts[:, None], out=sums, where=counts[:, None] > 0)
        if not np.isfinite(sums).all():
            raise ValueError("Nonfinite daily features")
        output = pd.concat([coverage.assign(has_news=counts > 0, n_articles=counts),
                            pd.DataFrame(sums.astype("float32"), columns=VALUES)], axis=1)
        target = out / f"{policy}_news.parquet"
        write_table(output, target)
        outputs[policy] = {"sha256": file_hash(target), "ticker_days": len(output),
                           "covered": int((counts > 0).sum()), "assigned_links": int(counts.sum())}
        del sums, output
    guard(out)
    complete = {**spec, "full": True, "timing": "exploratory proxies; no verified availability",
                "elapsed_seconds_this_invocation": time.monotonic() - start,
                "article_parts": {p.name: file_hash(p) for p in chunk_paths}, "daily": outputs}
    (out / "manifest.json").write_text(json.dumps(complete, indent=2) + "\n")


if __name__ == "__main__":
    main()
