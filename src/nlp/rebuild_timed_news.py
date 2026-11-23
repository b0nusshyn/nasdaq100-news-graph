"""Stage an experimental news audit and optionally encode it with a local FinBERT snapshot.

Requires a local FNSPID filtered CSV. Writes only to a new research directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.io import read_table
from src.core.paths import table_path
from src.nlp.encode_news import MODEL_NAME, aggregate, encode_articles
from src.nlp.filter_fnspid import OUT_COLUMNS, REVISION, source_url
from src.nlp.match_tickers import article_id, is_date_only

START, END = "2019-01-01", "2023-12-31"


def utc_values(raw: pd.Series, *, archive: bool = False) -> pd.Series:
    """Parse explicit offsets; in the documented archive only, bare dates/times use UTC."""
    raw = raw.astype("string").str.strip()
    allowed = raw.str.contains(r"(?:Z| UTC|[+-]\d{2}:?\d{2})$", na=False)
    if archive:
        allowed |= raw.str.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?", na=False)
    normalized = raw.str.replace(r" UTC$", "+00:00", regex=True)
    return pd.to_datetime(normalized.where(allowed), utc=True, format="mixed", errors="coerce")


def source_articles(path: Path, wanted: set[str]) -> pd.DataFrame:
    # Read only the required columns, so the long article bodies do not occupy memory.
    missing = set(OUT_COLUMNS) - set(pd.read_csv(path, nrows=0).columns)
    if missing:
        raise ValueError(f"Filtered FNSPID CSV missing columns: {sorted(missing)}")
    parts = []
    for chunk in pd.read_csv(path, usecols=["Url", "Date", "Article_title"], dtype="string", chunksize=50_000):
        chunk = chunk.dropna(subset=["Url"]).copy()
        chunk["article_id"] = chunk.Url.map(article_id)
        parts.append(chunk[chunk.article_id.isin(wanted)])
    if not parts:
        raise ValueError("Filtered source contains no article records")

    return summarize_source_rows(pd.concat(parts, ignore_index=True))


def summarize_source_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Reduce repeated exact URL records to one article while exposing conflicts."""
    rows = rows.copy()
    rows["archive_timestamp"] = utc_values(rows.Date, archive=True)
    grouped = rows.groupby("article_id")
    # Do not choose a convenient timestamp or headline when repeated URL records disagree.
    records = grouped.agg(source_url=("Url", "first"), title=("Article_title", "first"),
                          raw_date_example=("Date", "first"),
                          archive_timestamp=("archive_timestamp", "first"), source_rows=("Date", "size"),
                          valid_timestamp_rows=("archive_timestamp", "count"))
    if grouped.Url.nunique().gt(1).any():
        raise ValueError("Different URLs hash to the same mapped article ID")
    records["timestamp_conflict"] = grouped.archive_timestamp.nunique().gt(1)
    records["headline_conflict"] = grouped.Article_title.nunique().gt(1)
    records.loc[records.timestamp_conflict | records.valid_timestamp_rows.ne(records.source_rows),
                "archive_timestamp"] = pd.NaT
    records.loc[records.headline_conflict | records.title.fillna("").str.strip().eq(""), "title"] = pd.NA
    return records.reset_index()


def cutoff_sessions(
    archive_timestamp: pd.Series, arrivals: pd.Series, schedule: pd.DataFrame, *, publication_proxy: bool,
) -> pd.Series:
    """Return first session whose actual close is strictly after availability.

    Midnight archive values are date-only. Such values cannot establish same-day availability
    even if an arrival was supplied. Publication proxy is an explicit counterfactual policy.
    """
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None)
    closes = pd.DatetimeIndex(schedule["market_close"])
    if not len(closes) or closes.tz is None or not closes.is_monotonic_increasing or closes.has_duplicates:
        raise ValueError("Expected a nonempty increasing timezone-aware exchange close schedule")
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("Expected increasing unique session dates")
    date_only = is_date_only(archive_timestamp)
    missing_or_bad = arrivals.isna() | (arrivals < archive_timestamp)
    if publication_proxy:
        eligible = archive_timestamp.notna() & (arrivals.isna() | ~missing_or_bad)
        available = arrivals.fillna(archive_timestamp)
    else:
        eligible = archive_timestamp.notna() & ~missing_or_bad
        available = arrivals
    positions = closes.searchsorted(pd.DatetimeIndex(available), side="right")
    day_floor = sessions.searchsorted(archive_timestamp.dt.tz_localize(None).dt.normalize(), side="right")
    positions = np.where(date_only, np.maximum(positions, day_floor), positions)
    keep = eligible.to_numpy() & (positions < len(sessions))
    result = pd.Series(pd.NaT, index=archive_timestamp.index, dtype="datetime64[ns]")
    result.loc[keep] = sessions[positions[keep]].to_numpy()
    return result


def prepare(links, articles, arrivals, schedule, *, publication_proxy=False):
    articles = articles.astype({"article_id": links.article_id.dtype})
    audit = links.rename(columns={"date": "original_date"}).merge(
        articles, on="article_id", how="left", validate="many_to_one",
    )
    audit["feed_arrived_at"] = arrivals.reindex(audit.article_id).array
    audit["date_only"] = is_date_only(audit.archive_timestamp)
    audit["published_at"] = audit.archive_timestamp.mask(audit.date_only)
    audit["source_publication_date"] = audit.archive_timestamp.dt.normalize()
    audit["timestamp_source"] = np.where(audit.source_rows.notna(), "FNSPID archive", "source record missing")
    audit["timestamp_verified"] = False  # archive provenance alone is not independent verification
    audit["date"] = cutoff_sessions(
        audit.archive_timestamp, audit.feed_arrived_at, schedule, publication_proxy=publication_proxy,
    )
    audit["news_eligible"] = audit.date.between(START, END) & audit.title.notna()
    audit["availability_basis"] = np.where(
        audit.date.isna(), "unresolved",
        np.where(audit.feed_arrived_at.notna(), "recorded_arrival", "archive_publication_proxy"),
    )
    return audit


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="a new directory")
    parser.add_argument("--arrivals", type=Path, help="CSV with unique article_id and offset-aware feed_arrived_at")
    parser.add_argument("--publication-proxy", action="store_true", help="assume availability at archive publication")
    parser.add_argument("--model-snapshot", type=Path, help="local models--ProsusAI--finbert/snapshots/<commit>")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists")
    if not args.source.is_file():
        parser.error("Provide the local filtered FNSPID CSV; this command never downloads it")
    snapshot = args.model_snapshot
    if snapshot and (not snapshot.is_dir() or snapshot.parent.name != "snapshots"
                     or snapshot.parent.parent.name != "models--ProsusAI--finbert"
                     or not re.fullmatch("[0-9a-f]{40}", snapshot.name)):
        parser.error("Provide an existing ProsusAI/finbert cache snapshot directory with its full revision")

    import pandas_market_calendars as mcal

    links = read_table("article_tickers")
    articles = source_articles(args.source, set(links.article_id))
    arrivals = pd.Series(dtype="datetime64[ns, UTC]")
    if args.arrivals:
        raw = pd.read_csv(args.arrivals, dtype="string").set_index("article_id", verify_integrity=True)
        arrivals = utc_values(raw.feed_arrived_at)
        if arrivals.isna().any():
            parser.error("Arrival rows must contain valid timestamps with explicit offsets")
    schedule = mcal.get_calendar("NASDAQ").schedule(start_date="2018-12-24", end_date="2024-01-10")
    audit = prepare(links, articles, arrivals, schedule, publication_proxy=args.publication_proxy)
    eligible = audit.loc[audit.news_eligible, ["article_id", "date", "ticker"]]
    summary = {
        "archive_expected_source": source_url(), "archive_documented_revision": REVISION,
        "input_sha256": file_hash(args.source), "arrivals_sha256": file_hash(args.arrivals) if args.arrivals else None,
        "mapping_sha256": file_hash(table_path("article_tickers")),
        "policy": "archive_publication_proxy" if args.publication_proxy else "recorded_arrival_required",
        "calendar": "NASDAQ", "calendar_version": mcal.__version__, "cutoff": "strictly before market_close",
        "mapping_rows": len(links), "unique_mapping_articles": links.article_id.nunique(),
        "source_joined_links": int(audit.source_rows.notna().sum()), "source_joined_articles": len(articles),
        "date_only_links": int(audit.date_only.sum()),
        "non_null_archive_timestamp_links": int(audit.archive_timestamp.notna().sum()),
        "timestamp_conflict_articles": int(articles.timestamp_conflict.sum()),
        "headline_conflict_articles": int(articles.headline_conflict.sum()),
        "eligible_links": len(eligible), "eligible_articles": eligible.article_id.nunique(),
        "eligible_changed_dates": int((audit.news_eligible & audit.date.ne(audit.original_date)).sum()),
    }
    prices = read_table("prices")
    historical = read_table("news_features", columns=["date", "ticker", "n_articles"])
    counts = eligible.groupby(["date", "ticker"]).size().rename("reconstructed_articles").reset_index()
    comparison = historical.merge(counts, on=["date", "ticker"], how="left", validate="one_to_one")
    comparison["reconstructed_articles"] = comparison.reconstructed_articles.fillna(0).astype(int)
    summary["official_covered_ticker_days"] = int(historical.n_articles.gt(0).sum())
    summary["reconstructed_covered_ticker_days"] = int(comparison.reconstructed_articles.gt(0).sum())
    summary["changed_ticker_day_counts"] = int(comparison.n_articles.ne(comparison.reconstructed_articles).sum())
    historical_keys = pd.MultiIndex.from_frame(historical[["date", "ticker"]])
    in_comparison = pd.MultiIndex.from_frame(eligible[["date", "ticker"]]).isin(historical_keys)
    summary["eligible_links_outside_official_ticker_day_population"] = int((~in_comparison).sum())

    if snapshot:
        if eligible.empty:
            parser.error("No time-eligible links; supply arrivals or explicitly choose the publication proxy")
        if len(articles) != links.article_id.nunique():
            parser.error("Filtered input is missing mapped article IDs; audit first to review coverage")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True)
        model = model.to(args.device).eval()
        headlines = articles.loc[articles.title.notna(), ["article_id", "title"]]
        features = encode_articles(headlines.rename(columns={"title": "Article_title"}), device=args.device,
                                   tokenizer=tokenizer, model=model, progress=True)
        daily = aggregate(features, eligible, prices, START, END)
        summary["encoding"] = {"model": MODEL_NAME, "revision": snapshot.name, "device": args.device,
                               "article_rows": len(features), "final_layer_cls": True,
                               "fp16": args.device.startswith("cuda")}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit.to_csv(args.output_dir / "article_tickers_audit.csv", index=False)
    comparison.to_csv(args.output_dir / "ticker_day_count_comparison.csv", index=False)
    if snapshot:
        features.to_parquet(args.output_dir / "article_features.parquet", index=False)
        daily.to_parquet(args.output_dir / "news_features.parquet", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
