"""Rebuild measured coverage, missing-price requests and checksums after data recovery."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version

import pandas as pd

from src.core.io import read_table
from src.core.paths import CONFIGS_DIR, data_dir, raw_dir
from src.data.build_index_history import trading_days
from src.data.fetch_prices import add_in_index
from src.data.validation import checked_membership, checked_prices


def classify_missing_rows(missing: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Keep every missing key; annotate independently documented non-trading days.

    A trading halt never shortens the membership spell or creates a fake price.
    Evidence must establish both the halt and the later index-removal boundary.
    """
    required = {"ticker", "start", "end", "event", "source_url", "membership_source_url"}
    if required - set(events):
        raise ValueError(f"Non-trading events need {sorted(required)}")
    from src.core.tickers import canonical_ticker

    out = missing.copy()
    out["classification"] = "unexplained_missing_quote"
    out["source_url"] = ""
    for event in events.itertuples(index=False):
        start, end = pd.Timestamp(event.start), pd.Timestamp(event.end)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("Invalid non-trading event interval")
        if not all(str(url).startswith("https://") for url in [event.source_url, event.membership_source_url]):
            raise ValueError("Non-trading classifications require halt and membership evidence")
        mask = out.ticker.eq(canonical_ticker(event.ticker)) & out.date.between(start, end)
        if out.loc[mask, "classification"].ne("unexplained_missing_quote").any():
            raise ValueError("Overlapping non-trading classifications")
        out.loc[mask, "classification"] = "documented_nontrading_member_day"
        out.loc[mask, "source_url"] = event.source_url
    return out


def missing_price_rows(prices, membership, start: str, end: str) -> pd.DataFrame:
    """Exact absent in-index keys; bounds are inclusive, calendar is NASDAQ."""
    prices, membership = checked_prices(prices), checked_membership(membership)
    lower, upper = pd.Timestamp(start), pd.Timestamp(end)
    if lower > upper:
        raise ValueError("Audit start must not exceed end")
    days = trading_days(lower, upper)
    expected = []
    for row in membership.itertuples(index=False):
        dates = days[(days >= row.start) & (pd.isna(row.end) | (days <= row.end))]
        expected.append(pd.DataFrame({"date": dates, "ticker": row.ticker}))
    if not expected:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype="string")})
    expected = pd.concat(expected, ignore_index=True)
    observed = prices[["date", "ticker"]]
    joined = expected.merge(observed, on=["date", "ticker"], how="left", indicator=True, validate="one_to_one")
    return joined.loc[joined["_merge"].eq("left_only"), ["date", "ticker"]].sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2019-01-01", help="inclusive experiment boundary")
    parser.add_argument("--end", default="2023-12-31", help="inclusive experiment boundary")
    args = parser.parse_args()
    prices, labels, membership = read_table("prices"), read_table("labels"), read_table("membership")
    if not prices.in_index.equals(add_in_index(prices, membership).in_index):
        raise ValueError("Stored membership flags are stale; rebuild prices before auditing coverage")
    if set(prices[["date", "ticker"]].itertuples(index=False, name=None)) != set(
        labels[["date", "ticker"]].itertuples(index=False, name=None)
    ):
        raise ValueError("Prices and labels have different keys")
    missing = missing_price_rows(prices, membership, args.start, args.end)
    events_path = CONFIGS_DIR / "nontrading_events.csv"
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame(
        columns=["ticker", "start", "end", "event", "source_url", "membership_source_url"]
    )
    missing = classify_missing_rows(missing, events)
    documented = missing.classification.eq("documented_nontrading_member_day")
    unexplained = int((~documented).sum())
    folder = raw_dir()
    folder.mkdir(parents=True, exist_ok=True)
    missing.to_csv(folder / "missing_prices.csv", index=False, date_format="%Y-%m-%d")
    requests = missing.groupby("ticker").agg(
        first_missing=("date", "min"), last_missing=("date", "max"), missing_sessions=("date", "size"),
    ).reset_index()
    requests["unexplained_sessions"] = requests.ticker.map(
        missing.loc[~documented].groupby("ticker").size()
    ).fillna(0).astype(int)
    requests["existing_history"] = requests.ticker.isin(prices.ticker)
    requests["identity_review"] = requests.ticker.map(
        lambda ticker: "corporate-action review: old/new FOX issuers" if ticker in {"FOX", "FOXA"}
        else "use the historical issuer; never substitute acquirer"
    )
    requests.to_csv(folder / "recovery_requests.csv", index=False, date_format="%Y-%m-%d")
    main_prices = prices.loc[prices.in_index & prices.date.between(args.start, args.end)]
    main_labels = main_prices[["date", "ticker"]].merge(labels, on=["date", "ticker"], validate="one_to_one")
    annual = []
    for year, group in main_labels.groupby(main_labels.date.dt.year):
        for horizon in [1, 5]:
            y = group[f"y_{horizon}d"]
            annual.append({"year": int(year), "horizon": horizon, "in_index_rows": len(group),
                           "labelled": int(y.notna().sum()), "up": int(y.eq(1).sum()),
                           "down": int(y.eq(0).sum()), "null": int(y.isna().sum()),
                           "up_fraction": float(y.mean()) if y.notna().any() else None})
    pd.DataFrame(annual).to_csv(folder / "annual_label_balance.csv", index=False)
    entire = requests.loc[~requests.existing_history, "ticker"].tolist()
    summary = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_start": args.start, "experiment_end_inclusive": args.end,
        "observed_first_date": str(prices.date.min().date()),
        "observed_last_date": str(prices.date.max().date()),
        "membership_spells": len(membership), "membership_tickers": int(membership.ticker.nunique()),
        "available_tickers": int(prices.ticker.nunique()), "price_rows": len(prices), "label_rows": len(labels),
        "labelled_1d": int(labels.y_1d.notna().sum()), "labelled_5d": int(labels.y_5d.notna().sum()),
        "observed_in_index_rows": len(main_prices), "missing_in_index_rows": len(missing),
        "documented_nontrading_member_days": int(documented.sum()), "unexplained_missing_quote_rows": unexplained,
        "expected_in_index_rows": len(main_prices) + len(missing),
        "entirely_absent_members": entire, "tickers_with_partial_gaps": requests.loc[
            requests.existing_history, "ticker"
        ].tolist(),
        "versions": {name: version(name) for name in ["pandas", "yfinance", "pandas-market-calendars", "pyarrow"]},
        "reviewed_membership_events_sha256": sha256((CONFIGS_DIR / "membership_events.csv").read_bytes()).hexdigest(),
        "status": "INCOMPLETE: unexplained missing quotes remain" if unexplained else
                  "NO_UNEXPLAINED_MEMBER_QUOTE_GAPS: quote quality and data vintages still require review",
    }
    files = ["membership.parquet", "prices.parquet", "labels.parquet", "raw/prices_raw.csv",
             "raw/failed_tickers.csv", "raw/missing_prices.csv", "raw/recovery_requests.csv",
             "raw/quality_report.csv", "raw/recovery_provenance.json"]
    summary["sha256"] = {name: sha256((data_dir() / name).read_bytes()).hexdigest()
                         for name in files if (data_dir() / name).exists()}
    original = folder / "download_manifest.json"
    baseline = folder / "initial_download_manifest.json"
    if original.exists() and not baseline.exists():
        baseline.write_bytes(original.read_bytes())
    original.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"versions", "sha256"}}, indent=2))


if __name__ == "__main__":
    main()
