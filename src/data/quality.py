"""Data-quality findings on prices, labels and membership (Data, issue #20)."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256

import numpy as np
import pandas as pd

from src.core.io import read_table
from src.core.paths import CONFIGS_DIR, raw_dir, table_path
from src.core.tickers import canonical_ticker
from src.data.audit import classify_missing_rows, missing_price_rows
from src.data.build_index_history import trading_days
from src.data.build_labels import make_labels
from src.data.fetch_prices import add_in_index
from src.data.validation import canonical_table, checked_membership, checked_prices

REPORT_COLUMNS = ["check", "ticker", "year", "date", "value", "detail"]


def review_findings(report: pd.DataFrame, reviewed: pd.DataFrame) -> pd.DataFrame:
    """Join manual dispositions to exact anomaly keys; stale zero counts need review."""
    keys = ["check", "ticker", "date"]
    actual = report.loc[report["check"].isin(["adjustment_review", "zero_volume"])].copy()
    reviewed = reviewed.copy()
    required = {*keys, "flagged_rows", "status", "notes"}
    if required - set(reviewed):
        raise ValueError(f"Quality review needs {sorted(required)}")
    for frame in [actual, reviewed]:
        frame["ticker"] = frame.ticker.map(canonical_ticker)
        frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d").fillna("")
    if reviewed.duplicated(keys).any():
        raise ValueError("Duplicate quality review keys")
    joined = actual.merge(reviewed, on=keys, how="left", validate="one_to_one")
    stale = joined["check"].eq("zero_volume") & joined.value.ne(joined.flagged_rows)
    unknown = joined.status.isna() | joined.notes.isna() | stale
    joined.loc[unknown, "status"] = "unreviewed"
    return joined


def assess_quality(report: pd.DataFrame, missing: pd.DataFrame) -> dict:
    """Fail structural errors and unexplained member gaps; retain advisory flags.

    PASS_WITH_FLAGS means checks ran with disclosed limitations, not that flagged
    quotes were independently certified. Provider failures alone are not proof
    that the combined, recovered dataset still lacks a ticker.
    """
    fatal = {"empty_prices", "non_session_row", "membership_flag_mismatch",
             "label_return_mismatch", "label_direction_mismatch"}
    errors = report.loc[report["check"].isin(fatal)]
    unknown = int(missing.classification.eq("unexplained_missing_quote").sum())
    warnings = report.loc[report["check"].isin(["adjustment_review", "zero_volume", "missing_session"])]
    return {"status": "FAIL" if len(errors) or unknown else "PASS_WITH_FLAGS" if len(warnings) or len(missing)
            else "PASS", "structural_error_findings": len(errors),
            "missing_member_sessions": len(missing), "unexplained_missing_member_sessions": unknown,
            "documented_nontrading_member_sessions": len(missing) - unknown,
            "advisory_findings": len(warnings), "finding_counts": report.groupby("check").size().to_dict(),
            "interpretation": "Flags are retained; no quote, volume or missing session is fabricated."}


def quality_report(
    prices: pd.DataFrame, labels: pd.DataFrame, membership: pd.DataFrame | None = None,
    *, start: str | None = None, end: str | None = None,
) -> pd.DataFrame:
    """Report missing sessions, returns needing adjustment review and yearly balance.

    Optional bounds are inclusive. Membership makes entirely absent tickers and
    missing leading/trailing in-index dates visible. Otherwise only internal gaps
    are detectable. Jump flags are candidates for review, not confirmed splits.
    """
    prices = checked_prices(prices)
    labels = canonical_table(labels, "labels")
    keys = ["date", "ticker"]
    aligned = prices[keys + ["in_index"]].merge(labels, on=keys, how="outer", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Labels must have exactly the same keys as prices")
    findings = []

    def add(check, ticker="", year=None, date=None, value=None, detail=""):
        findings.append(dict(zip(REPORT_COLUMNS, (check, ticker, year, date, value, detail), strict=True)))

    if prices.empty:
        add("empty_prices", value=0)
        return pd.DataFrame(findings, columns=REPORT_COLUMNS)
    lower = pd.Timestamp(start) if start else prices.date.min()
    upper = pd.Timestamp(end) if end else prices.date.max()
    if lower > upper:
        raise ValueError("Quality start must not exceed end")
    sessions = trading_days(lower, upper)
    in_scope = prices.date.between(lower, upper).to_numpy()
    aligned = aligned.loc[aligned.date.between(lower, upper)]
    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date")
        expected = sessions[(sessions >= group.date.min()) & (sessions <= group.date.max())]
        for date in expected.difference(group.date):
            add("missing_session", ticker, date.year, date, 1, "Internal missing OHLCV; not filled")
        scoped = group.loc[group.date.between(lower, upper)]
        for date in pd.DatetimeIndex(scoped.date).difference(trading_days(group.date.min(), group.date.max())):
            add("non_session_row", ticker, date.year, date, 1, "Not a NASDAQ trading session")
        jumps = group.close.pct_change(fill_method=None)
        for idx in group.index[jumps.abs().gt(0.4) & group.date.between(lower, upper)]:
            add("adjustment_review", ticker, int(group.loc[idx, "date"].year), group.loc[idx, "date"],
                float(jumps.loc[idx]), "Absolute adjusted close return > 40%; check corporate actions")
        zeros = scoped.loc[scoped.volume.eq(0)]
        if not zeros.empty:
            add("zero_volume", ticker, value=len(zeros), detail="Review halts or missing provider volume")
    if membership is not None:
        members = checked_membership(membership)
        expected_flags = add_in_index(prices, members)
        wrong = prices.in_index.ne(expected_flags.in_index)
        if wrong.any():
            add("membership_flag_mismatch", value=int(wrong.sum()))
        for ticker, spells in members.groupby("ticker"):
            expected = pd.DatetimeIndex([])
            for row in spells.itertuples(index=False):
                expected = expected.union(sessions[(sessions >= row.start) &
                                                  (pd.isna(row.end) | (sessions <= row.end))])
            observed = prices.loc[prices.ticker.eq(ticker), "date"]
            missing = expected.difference(observed)
            if len(missing):
                add("missing_in_index", ticker, value=len(missing),
                    detail=f"Missing in-index sessions {missing.min().date()} to {missing.max().date()}")
            if observed.empty and len(expected):
                add("missing_ticker", ticker, value=len(expected),
                    detail="Entire member history absent; check failures")
    expected_labels = make_labels(prices)
    actual = labels.set_index(keys).reindex(pd.MultiIndex.from_frame(prices[keys]))
    for horizon in (1, 5):
        observed_ret = actual[f"ret_{horizon}d"].to_numpy(dtype=float)
        expected_ret = expected_labels[f"ret_{horizon}d"].to_numpy(dtype=float)
        incorrect = ~np.isclose(observed_ret, expected_ret, rtol=1e-10, atol=1e-12, equal_nan=True) & in_scope
        if incorrect.any():
            add("label_return_mismatch", value=int(incorrect.sum()), detail=f"{horizon}d calendar horizon")
        observed_y = actual[f"y_{horizon}d"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        expected_y = expected_labels[f"y_{horizon}d"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        wrong_y = ~np.isclose(observed_y, expected_y, equal_nan=True) & in_scope
        if wrong_y.any():
            add("label_direction_mismatch", value=int(wrong_y.sum()),
                detail=f"{horizon}d direction")
        for (ticker, year), group in aligned.loc[aligned.in_index].groupby(
            ["ticker", aligned.loc[aligned.in_index, "date"].dt.year]
        ):
            y = group[f"y_{horizon}d"]
            valid = y.dropna()
            add("class_balance", ticker, int(year), value=float(valid.mean()) if len(valid) else np.nan,
                detail=f"horizon={horizon}; labelled={len(valid)}; null={y.isna().sum()}; "
                       f"up={int(valid.sum())}; down={int(valid.eq(0).sum())}; in-index only")
    return pd.DataFrame(findings, columns=REPORT_COLUMNS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", help="inclusive first session to audit")
    p.add_argument("--end", help="inclusive last session to audit")
    p.add_argument("--strict", action="store_true", help="exit nonzero on structural errors or unexplained member gaps")
    args = p.parse_args()
    prices, labels, membership = read_table("prices"), read_table("labels"), read_table("membership")
    report = quality_report(prices, labels, membership,
                            start=args.start, end=args.end)
    lower, upper = args.start or str(prices.date.min().date()), args.end or str(prices.date.max().date())
    missing = missing_price_rows(prices, membership, lower, upper)
    events_path = CONFIGS_DIR / "nontrading_events.csv"
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame(
        columns=["ticker", "start", "end", "event", "source_url", "membership_source_url"])
    missing = classify_missing_rows(missing, events)
    failures = raw_dir() / "failed_tickers.csv"
    failed = pd.DataFrame(columns=["ticker", "reason"])
    if failures.exists():
        failed = pd.read_csv(failures)
        failed["ticker"] = failed.ticker.map(canonical_ticker)
        extra = [{"check": "download_failed", "ticker": row.ticker, "detail": row.reason}
                 for row in failed.itertuples(index=False)]
        report = pd.concat([report, pd.DataFrame(extra)], ignore_index=True)
    folder = raw_dir()
    folder.mkdir(parents=True, exist_ok=True)
    report.to_csv(folder / "quality_report.csv", index=False)
    missing.to_csv(folder / "quality_missing_sessions.csv", index=False, date_format="%Y-%m-%d")
    counts = prices.groupby("ticker").size()
    failed["combined_price_rows"] = failed.ticker.map(counts).fillna(0).astype(int)
    failed["currently_absent_in_audit"] = failed.ticker.isin(missing.ticker) & failed.combined_price_rows.eq(0)
    failed.to_csv(folder / "quality_download_status.csv", index=False)
    sample = prices.loc[prices.in_index & prices.date.between(lower, upper), ["date", "ticker"]].merge(
        labels, on=["date", "ticker"], validate="one_to_one")
    balances = []
    for year, group in sample.groupby(sample.date.dt.year):
        for h in [1, 5]:
            y = group[f"y_{h}d"]
            balances.append({"year": int(year), "horizon": h, "rows": len(y), "labelled": int(y.notna().sum()),
                             "null": int(y.isna().sum()), "up": int(y.eq(1).sum()), "down": int(y.eq(0).sum()),
                             "up_fraction": float(y.mean()) if y.notna().any() else None})
    pd.DataFrame(balances).to_csv(folder / "quality_class_balance.csv", index=False)
    summary = assess_quality(report, missing)
    review_path = CONFIGS_DIR / "quality_review.csv"
    reviewed = pd.read_csv(review_path) if review_path.exists() else pd.DataFrame(
        columns=["check", "ticker", "date", "flagged_rows", "status", "notes"])
    review = review_findings(report, reviewed)
    review.to_csv(folder / "quality_review_register.csv", index=False)
    summary["unreviewed_advisory_findings"] = int(review.status.eq("unreviewed").sum())
    if summary["unreviewed_advisory_findings"]:
        summary["status"] = "FAIL"
    summary.update(start=lower, end_inclusive=upper, price_rows=len(prices), tickers=int(prices.ticker.nunique()),
                   original_provider_failures=len(failed),
                   sha256={name: sha256(table_path(name).read_bytes()).hexdigest()
                           for name in ["membership", "prices", "labels"]})
    summary["review_config_sha256"] = sha256(review_path.read_bytes()).hexdigest() if review_path.exists() else None
    (folder / "quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report.groupby("check").size().to_string())
    print(f"{summary['status']}: {summary['unexplained_missing_member_sessions']} unexplained member gaps")
    print(folder / "quality_report.csv")
    if args.strict and summary["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
