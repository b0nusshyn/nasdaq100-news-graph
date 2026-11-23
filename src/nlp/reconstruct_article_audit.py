"""Recover a partial audit and sample scores from checked project outputs, without encoding."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.core.io import read_table
from src.core.paths import REPO_ROOT, raw_dir
from src.nlp.encode_news import MODEL_NAME
from src.nlp.match_tickers import article_id

KEYS = ["article_id", "ticker", "date"]
SCORE_COLUMNS = ["title", "sentiment", "confidence"]
REVIEW = Path("results/tables/match_spot_check.csv")
TRACE_FILES = tuple(Path("results/interpretability") / name for name in (
    "goog_2023-05-09_h1.csv", "lcid_2023-04-17_h1.csv", "sgen_2023-05-17_h1.csv",
))


def require_link_keys(records: pd.DataFrame, links: pd.DataFrame) -> None:
    keys = pd.MultiIndex.from_frame(records[KEYS])
    if not keys.isin(pd.MultiIndex.from_frame(links[KEYS])).all():
        raise ValueError("Source rows lack an exact article ID, ticker and aligned date mapping")


def recover(links: pd.DataFrame, review: pd.DataFrame, traces: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if links.duplicated(["article_id", "ticker"]).any():
        raise ValueError("Duplicate article–ticker pairs in upstream mapping")
    reviewed = review.rename(columns={"Url": "source_url", "source": "matching_source"}).copy()
    reviewed["article_id"] = reviewed.source_url.map(article_id)
    require_link_keys(reviewed, links)
    reviews = reviewed[KEYS + ["source_url", "title", "matching_source", "verdict"]]
    audit = links.merge(reviews, on=KEYS, how="left", validate="one_to_one")
    audit["matching_source"] = audit.matching_source.fillna("upstream_mapping_method_unknown")
    audit["matching_provenance"] = audit.verdict.notna().map({
        True: str(REVIEW), False: "article_tickers.parquet",
    })
    audit["matching_verified"] = audit.verdict.str.startswith("correct:", na=False)
    audit["matching_review_verdict"] = audit.pop("verdict").fillna("not_reviewed")
    audit["published_at"] = pd.NaT
    audit["feed_arrived_at"] = pd.NaT
    audit["timestamp_source"] = "unknown"
    audit["timestamp_verified"] = False

    scores = pd.concat(traces, ignore_index=True).rename(columns={"src": "ticker"})
    require_link_keys(scores, links)
    consistent = scores[["article_id", *SCORE_COLUMNS]].drop_duplicates()
    if consistent.article_id.duplicated().any() or consistent.isna().any(axis=None):
        raise ValueError("Conflicting or incomplete interpretability sample scores")
    provenance = scores.groupby("article_id").source_artifact.agg(lambda names: ";".join(sorted(set(names))))
    sample = consistent.join(provenance, on="article_id").sort_values("article_id", ignore_index=True)
    sample["documented_source_model"] = MODEL_NAME
    sample["model_revision"] = pd.Series(pd.NA, index=sample.index, dtype="string")
    return audit, sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    out = raw_dir()
    audit_path = out / "article_tickers_audit.csv"
    sample_path = out / "article_features_recovered_sample.parquet"
    if audit_path.exists() or sample_path.exists():
        parser.error("Recovery outputs already exist")
    links = read_table("article_tickers")
    review = pd.read_csv(REPO_ROOT / REVIEW, parse_dates=["date"])
    traces = [pd.read_csv(REPO_ROOT / path, parse_dates=["date"]).assign(source_artifact=str(path))
              for path in TRACE_FILES]
    audit, sample = recover(links, review, traces)
    out.mkdir(parents=True, exist_ok=True)
    with audit_path.open("x", encoding="utf-8", newline="") as stream:
        audit.to_csv(stream, index=False)
    with sample_path.open("xb") as stream:
        sample.to_parquet(stream, index=False)
    print(f"{len(audit):,} article–ticker links -> {audit_path}")
    print(f"{len(sample):,} recovered article sentiment rows -> {sample_path}")


if __name__ == "__main__":
    main()
