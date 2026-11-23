"""Audit acquired bars without generating post-2023 labels or model metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.core.io import read_table
from src.data.validation import invalid_ohlc
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent
# Stop at the last fully completed NASDAQ session at retrieval.
LAST = "2026-11-19"
COLS = ["open", "high", "low", "close", "volume"]


def active(membership, day):
    return set(membership.loc[membership.start.le(day) & (membership.end.isna() | membership.end.ge(day)), "ticker"])


def main():
    acquire = json.loads((ROOT / "acquisition.json").read_text())
    entries = {Path(r["path"]).stem: r for r in acquire["files"] if r["path"].endswith(".parquet") and "sha256" in r}
    sessions = mcal.get_calendar("NASDAQ").schedule("2023-01-01", LAST).index
    cohort = acquire["cohort"]
    original = read_table("prices")
    membership = read_table("membership")
    external = []
    gaps, checks, overlaps = [], [], []
    for ticker in cohort:
        path = ROOT / "raw" / f"{ticker}.parquet"
        bars = pd.DataFrame(columns=["date", *COLS, "adj_close"])
        if ticker in entries:
            assert file_hash(path) == entries[ticker]["sha256"]
            bars = pd.read_parquet(path)
        bars["date"] = pd.to_datetime(bars.date, format="%Y-%m-%d")
        bars = bars.loc[bars.date.between("2023-01-01", LAST)].copy().sort_values("date")
        if bars.date.duplicated().any():
            raise ValueError(f"Duplicate sessions for {ticker}")
        factor = bars.adj_close / bars.close
        valid = (np.isfinite(bars[[*COLS, "adj_close"]].astype(float)).all(axis=1)
                 & bars.adj_close.gt(0) & bars.volume.ge(0) & factor.gt(0))
        raw_min = bars[["open", "high", "low", "close"]].min(axis=1)
        valid &= raw_min.gt(0)
        if len(bars):
            valid &= ~invalid_ohlc(bars)
        seen = pd.DatetimeIndex(bars.loc[valid & bars.date.isin(sessions), "date"])
        for day in sessions.difference(seen):
            gaps.append({"ticker": ticker, "date": day, "ticker_in_original_membership_on_date":
                         ticker in active(membership, day)})
        for year in range(2023, 2027):
            y = seen[seen.year == year]
            checks.append({"ticker": ticker, "year": year, "expected": int(sum(sessions.year == year)),
                           "usable": len(y), "first": y.min(), "last": y.max(),
                           "source_file": ticker in entries,
                           "bad_bars": int((~valid & bars.date.dt.year.eq(year)).sum()),
                           "non_session": int((~bars.date.isin(sessions) & bars.date.dt.year.eq(year)).sum())})
        if not len(bars):
            continue
        external.append(bars.assign(ticker=ticker))
        # Only calculate price and direction comparisons on the 2023 overlap.
        adj = bars.loc[bars.date.lt("2024-01-01") & valid].set_index("date")
        prev = original[original.ticker.eq(ticker) & original.date.lt("2024-01-01")].set_index("date")
        idx = sessions[sessions.year == 2023]
        adj, prev = adj.reindex(idx), prev.reindex(idx)
        a, b = adj.adj_close, prev.close
        ratio = a / b
        med = ratio.median()
        row = {"ticker": ticker, "overlap": int(ratio.count()), "adj_close_level_ratio_median": med,
               "max_relative_level_deviation_from_ticker_median": (ratio / med - 1).abs().max(),
               "volume_exact_matches": int(adj.volume.eq(prev.volume).sum())}
        for k in ("open", "high", "low", "close"):
            af = adj[k] * adj.adj_close / adj.close
            error = (af / prev[k] / med - 1).abs()
            row[f"normalized_{k}_median_abs_error_bps"] = 10000 * error.median()
        for h in (1, 5):
            ra, rb = a.pct_change(h, fill_method=None), b.pct_change(h, fill_method=None)
            usable = ra.notna() & rb.notna()
            row[f"h{h}_matched_returns"] = int(usable.sum())
            row[f"h{h}_direction_disagreements"] = int((ra.gt(0).ne(rb.gt(0)) & usable).sum())
            row[f"h{h}_median_absolute_return_difference_bps"] = 10000 * (ra - rb).abs().median()
        overlaps.append(row)
    c = pd.DataFrame(checks)
    g = pd.DataFrame(gaps)
    o = pd.DataFrame(overlaps)
    p = pd.concat(external, ignore_index=True)
    for frame, name in [(c, "coverage.csv"), (g, "missing_sessions.csv"), (o, "overlap_2023.csv")]:
        frame.to_csv(ROOT / name, index=False, mode="x")
    with (ROOT / "external_raw_panel.parquet").open("xb") as out:
        p.to_parquet(out, compression="zstd", index=False)
    changes = []
    for date in ("2023-12-29", "2024-12-31", "2025-12-31", LAST):
        members = active(membership, pd.Timestamp(date))
        changes.append({"asof": date, "fixed_cohort_outside_index": sorted(set(cohort)-members),
                        "index_outside_fixed_cohort": sorted(members-set(cohort))})
    status = {"last_completed_session": LAST, "membership": changes, "source_panel_rows": len(p),
              "independent_scoring": False, "training_fits": 0}
    with (ROOT / "audit_summary.json").open("x") as out:
        json.dump(status, out, indent=2)
    print(c.groupby("year")[["expected", "usable", "bad_bars", "non_session"]].sum().to_string())
    print(o.select_dtypes("number").sum().to_string())
    print(c[c.usable.lt(c.expected)&c.source_file].to_string(index=False))


if __name__ == "__main__":
    main()
