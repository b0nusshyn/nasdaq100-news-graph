"""Audit two vendor Parquets against the frozen project's exact URL and ticker keys."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import pyarrow.parquet as pq

from src.core.io import read_table
from src.core.tickers import canonical_ticker
from src.nlp.filter_fnspid import filter_chunk, universe
from src.nlp.match_tickers import article_id, assign_trading_day, build_aliases, build_article_tickers
from src.nlp.rebuild_timed_news import cutoff_sessions, utc_values

SHARDS = [f"train-0000{i}-of-00002.parquet" for i in (0, 1)]
COLS = ["Url", "Date", "Stock_symbol", "Article_title"]
START, END = "2019-01-01", "2023-12-31"


def keys(frame, cols):
    return pd.MultiIndex.from_frame(frame[cols])


def counts(mask):
    return int(mask.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    where = next((d for d in [Path("data/raw/fnspid_external_candidate"), Path("data")]
                  if all((d / name).is_file() for name in SHARDS)), None)
    if where is None:
        parser.error("Both local candidate shards are required")
    meta, frames = [], []
    for name in SHARDS:
        path = where / name
        parquet = pq.ParquetFile(path)
        frame = parquet.read(columns=COLS + ["Unnamed: 0", "Publisher", "Author"]).to_pandas()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        nulls = {}
        for i, col in enumerate(parquet.schema.names):
            stats = [parquet.metadata.row_group(j).column(i).statistics for j in range(parquet.num_row_groups)]
            nulls[col] = sum(s.null_count for s in stats) if all(s and s.has_null_count for s in stats) else None
        meta.append({"path": str(path), "bytes": path.stat().st_size, "rows": len(frame), "sha256": digest,
                     "schema": {f.name: str(f.type) for f in parquet.schema_arrow},
                     "footer_null_counts": nulls,
                     "metadata_keys": sorted(k.decode() for k in (parquet.schema_arrow.metadata or {})),
                     "writer": parquet.metadata.created_by})
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    tmp = raw.Url.dropna()
    urls = tmp[tmp.str.strip().ne("")]
    candidate = raw.loc[urls.index].assign(article_id=urls.map(article_id))
    assert candidate.groupby("article_id").Url.nunique().le(1).all()
    candidate["archive_time"] = utc_values(candidate.Date, archive=True)
    candidate["ticker"] = candidate.Stock_symbol.dropna().map(canonical_ticker).reindex(candidate.index)
    mapping = read_table("article_tickers")
    mapped_ids = set(mapping.article_id)
    covered = mapping[mapping.article_id.isin(candidate.article_id)].copy()
    tag_keys = keys(candidate.dropna(subset=["ticker"]).drop_duplicates(["article_id", "ticker"]),
                    ["article_id", "ticker"])
    matched_tag = keys(covered, ["article_id", "ticker"]).isin(tag_keys)
    candidate_tag = keys(candidate, ["article_id", "ticker"]).isin(keys(mapping, ["article_id", "ticker"]))
    groups = candidate.groupby("article_id")
    time_n = groups.archive_time.nunique()
    title_n = candidate.assign(valid_title=candidate.Article_title.where(
        candidate.Article_title.fillna("").str.strip().ne(""))).groupby("article_id").valid_title.nunique()
    unique_time = groups.archive_time.first().where(time_n.eq(1) & groups.archive_time.count().eq(groups.size()))
    unique_title = title_n.eq(1)
    audit_source = pd.read_csv("data/raw/article_tickers_audit.csv", usecols=["article_id", "ticker", "date",
                                                                           "timestamp_verified"], dtype=str)
    assert len(audit_source) == len(mapping)
    assert keys(audit_source.assign(date=pd.to_datetime(audit_source.date)), ["article_id", "ticker", "date"]).isin(
        keys(mapping, ["article_id", "ticker", "date"])).all()

    schedule = mcal.get_calendar("NASDAQ").schedule("2018-12-20", "2024-01-15")
    in_schedule = unique_time[unique_time.notna() & unique_time.between("2018-12-20", "2024-01-14")]
    times = covered.article_id.map(in_schedule)
    known = times.notna()
    aligned = assign_trading_day(times[known], schedule.index).astype("datetime64[ns]")
    proxy = cutoff_sessions(times[known], pd.Series(pd.NaT, index=times[known].index, dtype="datetime64[ns, UTC]"),
                            schedule, publication_proxy=True)
    covered["clean_headline_time"] = covered.article_id.map(unique_title).fillna(False) & known
    covered["legacy_candidate_date"] = aligned
    covered["actual_close_proxy_date"] = proxy
    official_news = read_table("news_features", columns=["date", "ticker", "has_news", "n_articles"])
    news_keys = keys(official_news[official_news.has_news], ["date", "ticker"])
    covered_day = keys(covered, ["date", "ticker"]).unique()
    clean = covered[covered.clean_headline_time & covered.actual_close_proxy_date.between(START, END)]
    proxy_keys = keys(clean.assign(date=clean.actual_close_proxy_date), ["date", "ticker"]).unique()
    etched = candidate.archive_time.dt
    midnight = (etched.hour == 0) & (etched.minute == 0) & (etched.second == 0)
    stamp = utc_values(raw.Date, archive=True)
    utc_hour = stamp.dt.hour.value_counts().sort_index()
    published_year = stamp.dt.year.value_counts().sort_index()
    daily = mapping.assign(recovered=mapping.article_id.isin(candidate.article_id)).groupby(mapping.date.dt.year)
    year = [{"year": int(y), "original_pairs": len(g), "matched_pairs": counts(g.recovered),
             "original_ids": g.article_id.nunique(), "matched_ids": g[g.recovered].article_id.nunique(),
             "matched_dates": g[g.recovered].date.nunique()} for y, g in daily]

    # Compare original filter selection and matcher using candidate data only.
    membership = read_table("membership")
    members = universe(membership, START, END)
    filtered = filter_chunk(raw.assign(Publisher=pd.NA, Article=pd.NA), members, START, END)
    aliases = {t: a for t, a in build_aliases(membership).items() if t in members}
    rematched = build_article_tickers(filtered, aliases, schedule.index, START, END)
    original_keys = keys(mapping, ["article_id", "ticker"])
    rematched_pairs = keys(rematched, ["article_id", "ticker"])
    rematched_exact = keys(rematched, ["article_id", "ticker", "date"]).isin(
        keys(mapping, ["article_id", "ticker", "date"]))
    duplicate_urls = set(urls[urls.duplicated()])

    body_present = 0
    body_variants = {}
    for name in SHARDS:
        for batch in pq.ParquetFile(where / name).iter_batches(batch_size=4096, columns=["Url", "Article"]):
            for url, body in zip(*[col.to_pylist() for col in batch.columns], strict=True):
                if body and body.strip():
                    body_present += 1
                    if url in duplicate_urls:
                        body_variants.setdefault(url, set()).add(hashlib.sha256(body.encode()).digest())
    overlaps = candidate[candidate.article_id.isin(mapped_ids)]
    on_time = unique_time[unique_time.index.isin(mapped_ids)].dropna()
    different_dates = covered.loc[known & covered.legacy_candidate_date.ne(covered.date)]
    outside_rel = proxy[~proxy.between(START, END)]
    result = {
        "shards": meta, "rows": len(raw), "unique_urls": urls.nunique(), "raw_symbols": raw.Stock_symbol.nunique(),
        "canonical_symbols": candidate.ticker.nunique(), "missing_or_blank": {
            c: counts(raw[c].fillna("").str.strip().eq("")) for c in raw},
        "body_present": body_present, "min": str(stamp.min()), "max": str(stamp.max()),
        "utc_suffix": counts(raw.Date.str.endswith(" UTC", na=False)),
        "timestamp_parse_failures": counts(stamp.isna()), "utc_hours": {int(k): int(v) for k, v in utc_hour.items()},
        "utc_years": {int(k): int(v) for k, v in published_year.items()},
        "whole_hour": counts(stamp.dt.minute.eq(0) & stamp.dt.second.eq(0)),
        "midnight_utc": counts(midnight), "distinct_timestamps": stamp.nunique(),
        "largest_timestamp_bucket": int(stamp.value_counts().max()),
        "outside_project_eastern_date": counts(~stamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
                                                .dt.normalize().between(START, END)),
        "duplicate_urls": len(duplicate_urls), "extra_url_rows": len(urls) - urls.nunique(),
        "duplicate_projected_rows": counts(raw.duplicated(COLS)),
        "conflicting_time_ids": counts(time_n.gt(1)), "conflicting_headline_ids": counts(title_n.gt(1)),
        "conflicting_body_urls": sum(len(v) > 1 for v in body_variants.values()),
        "missing_headline_ids": counts(title_n.eq(0)),
        "mapping_ids": len(mapped_ids), "mapping_pairs": len(mapping),
        "matched_ids": covered.article_id.nunique(), "matched_pairs": len(covered),
        "matched_tickers": covered.ticker.nunique(), "matched_session_dates": covered.date.nunique(),
        "matched_min_date": str(covered.date.min()), "matched_max_date": str(covered.date.max()),
        "matched_candidate_rows": len(overlaps), "direct_tag_pairs": counts(matched_tag),
        "overlap_candidate_rows_without_official_tag_pair":
        counts(candidate.article_id.isin(mapped_ids) & ~candidate_tag),
        "consistent_overlapping_intraday_looking_ids": counts(on_time.ne(on_time.dt.normalize())),
        "matched_ids_conflicting_times": counts(time_n[time_n.index.isin(mapped_ids)].gt(1)),
        "matched_ids_conflicting_headlines": counts(title_n[title_n.index.isin(mapped_ids)].gt(1)),
        "matched_ids_no_headline": counts(title_n[title_n.index.isin(mapped_ids)].eq(0)),
        "matched_ids_with_consistent_time": counts(unique_time[unique_time.index.isin(mapped_ids)].notna()),
        "date_comparable_pairs": counts(known),
        "legacy_date_disagreements": counts(aligned.ne(covered.loc[known, "date"])),
        "legacy_date_disagreement_ids": different_dates.article_id.nunique(),
        "legacy_discrepant_pairs_moved_earlier": counts(different_dates.legacy_candidate_date.lt(different_dates.date)),
        "proxy_outside_period_comparable_pairs": len(outside_rel),
        "actual_close_date_disagreements": counts(proxy.ne(covered.loc[known, "date"])),
        "actual_close_vs_legacy_changes": counts(proxy.ne(aligned)),
        "clean_proxy_in_period_ids": clean.article_id.nunique(), "clean_proxy_in_period_pairs": len(clean),
        "official_covered_ticker_days": len(news_keys),
        "official_covered_ticker_days_with_recovered_ids": counts(news_keys.isin(covered_day)),
        "official_covered_ticker_days_with_clean_proxy": counts(news_keys.isin(proxy_keys)),
        "official_news_population_days_with_clean_proxy": counts(
            keys(official_news, ["date", "ticker"]).isin(proxy_keys)),
        "original_filter_candidate_rows": len(filtered), "rematched_pairs": len(rematched),
        "rematched_ids": rematched.article_id.nunique(),
        "rematched_pairs_in_original": counts(rematched_pairs.isin(original_keys)),
        "rematched_exact_triples": counts(rematched_exact), "by_original_aligned_year": year,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, default=int)
        stream.write("\n")


if __name__ == "__main__":
    main()
