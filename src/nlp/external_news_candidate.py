"""Build a partial news candidate using the original mappings and an exploratory timing proxy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.core.io import read_table
from src.nlp.match_tickers import article_id, assign_trading_day, is_date_only
from src.nlp.rebuild_timed_news import END, START, file_hash, summarize_source_rows

SOURCE = "https://huggingface.co/datasets/benstaf/FNSPID-filtered-nasdaq-100"
POLICIES = ("next_session", "extra_session")
DATED_RECAP = (
    r"(?i)^(?:Company News|Stock Market News|Pre.Market Most Active) for "
    r"([A-Za-z]{3,9}\.? \d{1,2},? \d{4})(?:\b|$)"
)


def read_shards(shards, wanted):
    frames = []
    for shard in shards:
        frame = pq.read_table(shard, columns=["Url", "Date", "Article_title"]).to_pandas()
        frame = frame.loc[frame.Url.notna() & frame.Url.str.strip().ne("")].copy()
        frame["article_id"] = frame.Url.map(article_id)
        frames.append(frame.loc[frame.article_id.isin(wanted)].assign(shard=shard.name))
    rows = pd.concat(frames, ignore_index=True)
    if rows.empty:
        raise ValueError("No original article IDs matched")
    articles = summarize_source_rows(rows)
    provenance = rows[["article_id", "shard"]].drop_duplicates().groupby("article_id").shard.agg(",".join)
    # Check every matching headline variant, including those already marked as conflicts.
    dated = pd.to_datetime(rows.Article_title.str.extract(DATED_RECAP, expand=False), format="mixed", errors="coerce")
    addressed = rows.assign(recap_date=dated).groupby("article_id").recap_date.max()
    return articles.join(provenance, on="article_id").join(addressed, on="article_id")


def align_candidate(links, articles, schedule):
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None)
    closes = pd.DatetimeIndex(schedule.market_close)
    archive = articles.archive_timestamp
    date_only = is_date_only(archive)
    archive_day = archive.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    archive_day = archive_day.where(~date_only, archive.dt.tz_localize(None).dt.normalize())
    known = archive.notna() & archive_day.between(sessions[0], sessions[-1])
    legacy = assign_trading_day(archive[known], sessions).astype("datetime64[ns]")
    articles = articles.assign(
        date_only=date_only, archive_day=archive_day, legacy_date=legacy,
        future_dated_recap=articles.recap_date.gt(archive_day),
        timestamp_verified=False,
    )
    audit = links.rename(columns={"date": "original_date"}).merge(articles, on="article_id", how="left",
                                                                  validate="many_to_one")
    disagree = audit.legacy_date.notna() & audit.legacy_date.ne(audit.original_date)
    audit["legacy_date_conflict"] = audit.article_id.isin(audit.loc[disagree, "article_id"])
    audit["archive_time_present"] = audit.archive_timestamp.notna()
    audit["source_present"] = audit.source_rows.notna()
    audit["quarantined"] = audit.source_present & (
        audit.legacy_date_conflict | audit.future_dated_recap.eq(True)
        | audit.title.isna() | audit.legacy_date.isna()
    )
    eligible = audit.source_present & ~audit.quarantined
    base = np.maximum(sessions.searchsorted(audit.archive_day, side="right"),
                      closes.searchsorted(pd.DatetimeIndex(audit.archive_timestamp), side="right"))
    # Never bring an article forward relative to its original mapped session.
    base = np.maximum(base, sessions.searchsorted(audit.original_date))
    for delay, policy in enumerate(POLICIES):
        pos = base + delay
        inside = eligible.to_numpy() & (pos < len(sessions))
        days = pd.Series(pd.NaT, index=audit.index, dtype="datetime64[ns]")
        days.loc[inside] = sessions[pos[inside]].to_numpy()
        keep = inside & days.between(START, END).to_numpy()
        selected = audit.loc[keep]
        if not (
            (selected.archive_timestamp < closes[pos[keep]]).all()
            and (selected.archive_day < days[keep]).all()
            and (selected.original_date <= days[keep]).all()
        ):
            raise ValueError("An article violates its proxy cutoff")
        audit[policy] = days.where(keep)
    return audit


def coverage_table(audit, official):
    keys = ["date", "ticker"]
    daily = official[keys + ["n_articles"]].rename(columns={"n_articles": "official_article_count"})
    for name, subset in (
        ("missing_source_count", audit[~audit.source_present]),
        ("quarantined_source_count", audit[audit.quarantined]),
    ):
        sizes = subset.rename(columns={"original_date": "date"}).groupby(keys).size()
        daily[name] = pd.MultiIndex.from_frame(daily[keys]).map(sizes).fillna(0).astype(int)
    for policy in POLICIES:
        sizes = audit.rename(columns={policy: "date"}).groupby(keys).size()
        daily[policy + "_count"] = pd.MultiIndex.from_frame(daily[keys]).map(sizes).fillna(0).astype(int)
    return daily


def main():
    import pandas_market_calendars as mcal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists")
    if len(set(args.shards)) != 2 or not all(path.is_file() for path in args.shards):
        parser.error("Provide two distinct existing candidate shards")
    links = read_table("article_tickers")
    official = read_table("news_features", columns=["date", "ticker", "n_articles"])
    articles = read_shards(args.shards, set(links.article_id))
    schedule = mcal.get_calendar("NASDAQ").schedule("2018-12-20", "2024-01-15")
    audit = align_candidate(links, articles, schedule)
    assert len(audit) == len(links) and not audit.duplicated(["article_id", "ticker"]).any()
    missing = audit.loc[~audit.source_present, ["article_id"]].drop_duplicates()
    daily = coverage_table(audit, official)
    retained = audit.loc[audit[list(POLICIES)].notna().any(axis=1), "article_id"].unique()
    headlines = articles[articles.article_id.isin(retained)]
    assert headlines.article_id.is_unique and headlines.title.notna().all()
    summary = {
        "source": SOURCE, "policy": "exploratory; no independently verified publication/arrival timestamps",
        "calendar": f"NASDAQ, pandas_market_calendars {mcal.__version__}",
        "shards": {str(p): file_hash(p) for p in args.shards},
        "original_links": len(links), "source_joined_articles": len(articles), "missing_articles": len(missing),
        "headline_rows_for_encoding": len(headlines),
        "quarantined_pairs": int(audit.quarantined.sum()),
        "legacy_conflict_articles": audit.loc[audit.legacy_date_conflict, "article_id"].nunique(),
        "future_dated_recap_articles": audit.loc[audit.future_dated_recap.eq(True), "article_id"].nunique(),
        "official_ticker_days": len(daily),
        "policies": {},
    }
    population = pd.MultiIndex.from_frame(daily[["date", "ticker"]])
    coverage = []
    for policy in POLICIES:
        shifted = audit[audit[policy].notna()]
        inside = pd.MultiIndex.from_frame(shifted[[policy, "ticker"]]).isin(population)
        n = daily[policy + "_count"]
        summary["policies"][policy] = {
            "articles": shifted.article_id.nunique(), "pairs": len(shifted),
            "pairs_on_population": int(inside.sum()), "articles_on_population": shifted[inside].article_id.nunique(),
            "ticker_days": int(n.gt(0).sum()),
            "originally_covered_ticker_days_retained": int((n.gt(0) & daily.official_article_count.gt(0)).sum()),
        }
        view = daily.assign(year=daily.date.dt.year, covered=n.gt(0).astype(int))
        rows = view.groupby(["year", "ticker"]).agg(ticker_days=("covered", "size"), covered=("covered", "sum"),
                                                    missing_source=("missing_source_count", "sum")).reset_index()
        coverage.append(rows.assign(policy=policy))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit.to_parquet(args.output_dir / "link_audit.parquet", index=False)
    headlines.to_parquet(args.output_dir / "candidate_headlines.parquet", index=False)
    missing.to_csv(args.output_dir / "missing_article_ids.csv", index=False)
    daily.to_parquet(args.output_dir / "ticker_day_coverage.parquet", index=False)
    pd.concat(coverage).to_csv(args.output_dir / "coverage_by_year_ticker.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
