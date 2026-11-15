"""Data-quality findings on prices, labels and membership (Data, issue #20)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.core.io import read_table
from src.core.paths import raw_dir
from src.data.build_index_history import trading_days
from src.data.build_labels import make_labels
from src.data.fetch_prices import add_in_index
from src.data.validation import canonical_table, checked_membership, checked_prices

REPORT_COLUMNS = ["check", "ticker", "year", "date", "value", "detail"]


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
    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date")
        expected = sessions[(sessions >= group.date.min()) & (sessions <= group.date.max())]
        for date in expected.difference(group.date):
            add("missing_session", ticker, date.year, date, 1, "Internal missing OHLCV; not filled")
        for date in pd.DatetimeIndex(group.date).difference(trading_days(group.date.min(), group.date.max())):
            add("non_session_row", ticker, date.year, date, 1, "Not a NASDAQ trading session")
        jumps = group.close.pct_change(fill_method=None)
        for idx in group.index[jumps.abs().gt(0.4)]:
            add("adjustment_review", ticker, int(group.loc[idx, "date"].year), group.loc[idx, "date"],
                float(jumps.loc[idx]), "Absolute adjusted close return > 40%; check corporate actions")
        zeros = group.loc[group.volume.eq(0)]
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
        incorrect = ~np.isclose(observed_ret, expected_ret, rtol=1e-10, atol=1e-12, equal_nan=True)
        if incorrect.any():
            add("label_return_mismatch", value=int(incorrect.sum()), detail=f"{horizon}d calendar horizon")
        observed_y = actual[f"y_{horizon}d"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        expected_y = expected_labels[f"y_{horizon}d"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        if not np.isclose(observed_y, expected_y, equal_nan=True).all():
            add("label_direction_mismatch", value=int((~np.isclose(observed_y, expected_y, equal_nan=True)).sum()),
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
    args = p.parse_args()
    report = quality_report(read_table("prices"), read_table("labels"), read_table("membership"),
                            start=args.start, end=args.end)
    failures = raw_dir() / "failed_tickers.csv"
    if failures.exists():
        failed = pd.read_csv(failures)
        extra = [{"check": "download_failed", "ticker": row.ticker, "detail": row.reason}
                 for row in failed.itertuples(index=False)]
        report = pd.concat([report, pd.DataFrame(extra)], ignore_index=True)
    folder = raw_dir()
    folder.mkdir(parents=True, exist_ok=True)
    report.to_csv(folder / "quality_report.csv", index=False)
    print(report.groupby("check").size().to_string())
    print(folder / "quality_report.csv")


if __name__ == "__main__":
    main()
