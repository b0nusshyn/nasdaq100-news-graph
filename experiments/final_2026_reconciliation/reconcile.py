"""Offline coverage and endpoint audit. Does not compute 2026 returns or labels."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from lxml import html

from src.core.io import read_table
from src.data.build_index_history import trading_days

ROOT = Path(__file__).parent
PRIOR = ROOT.parent / "post2023_validation_v1"
LAST = "2026-11-19"
OHLC = ["open", "high", "low", "close"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_page(path):
    vendor, symbol = path.stem.split("_")
    page = html.parse(path)
    chart = vendor == "chartexchange"
    prices = []
    for tr in page.xpath("//table/tbody/tr"):
        cells = [td.text_content().strip() for td in tr.xpath("td")]
        if len(cells) < 8:
            continue
        try:
            date = pd.to_datetime(cells[0], format="%Y-%m-%d" if chart else "%b %d, %Y")
        except ValueError:
            continue
        o, h, lo, c = map(float, cells[1:5])
        vol = cells[6 if chart else 7]
        adj = None if chart else float(cells[5])
        prices.append((date, symbol, vendor, o, h, lo, c, int(vol.replace(",", "")) if vol != "-" else None, adj))
    cols = ["date", "ticker", "provider", *OHLC, "volume", "adj_close"]
    out = pd.DataFrame(prices, columns=cols)
    assert len(out) and not out.date.duplicated().any(), path
    return out


def endpoint_status(date, target, exit_day, asof, available, common):
    if date > exit_day:
        return "after_last_trade"
    if target > exit_day:
        return "crosses_exit"
    if target > asof:
        return "target_after_snapshot"
    if date not in available or target not in available:
        return "missing_observed_endpoint"
    if date not in common or target not in common:
        return "alternate_source_adjustment_unqualified"
    return "archive_endpoints_present_provisional"


def main():
    paths = sorted((ROOT / "raw").iterdir())
    actions = pd.read_csv(ROOT / "corporate_actions.csv", parse_dates=["last_trade"])
    exits = actions.set_index("ticker").last_trade.to_dict()
    archived_path = PRIOR / "external_raw_panel.parquet"
    archive = pd.read_parquet(archived_path)
    cohort = json.loads((PRIOR / "acquisition.json").read_text())["cohort"]
    assert len(cohort) == len(set(cohort)) == 101
    prior = read_table("prices")
    membership = read_table("membership")
    asof = pd.Timestamp(LAST)
    schedule = trading_days("2023-01-01", asof + pd.Timedelta(days=15))
    sessions = schedule[schedule <= asof]
    anchor = pd.Timestamp("2023-12-29")
    active_at_anchor = membership.start.le(anchor) & (membership.end.isna() | membership.end.ge(anchor))
    assert set(cohort) == set(membership.loc[active_at_anchor, "ticker"])
    alternate = pd.concat([parse_page(p) for p in paths if p.suffix == ".html"], ignore_index=True)
    alternate["post_exit"] = alternate.date.gt(alternate.ticker.map(exits))
    alternate["valid_ohlcv"] = (
        alternate[OHLC].min(axis=1).gt(0)
        & alternate.low.le(alternate[["open", "close"]].min(axis=1))
        & alternate.high.ge(alternate[["open", "close"]].max(axis=1))
        & alternate.volume.gt(0)
    )
    alternate.to_csv(ROOT / "recovered_candidates.csv", index=False, mode="x")
    archive["post_exit"] = archive.date.gt(archive.ticker.map(exits))
    archive.loc[archive.post_exit].to_csv(ROOT / "archive_quarantine.csv", index=False, mode="x")
    comparisons = []
    for ticker, group in alternate[~alternate.post_exit & alternate.valid_ohlcv].groupby("ticker"):
        old = prior[prior.ticker.eq(ticker)].assign(provider="project_adjusted")
        old = old.assign(adj_close=old.close)
        ext = archive[archive.ticker.eq(ticker) & ~archive.post_exit].assign(provider="pinned_archive")
        for source, left in group.groupby("provider"):
            for other, right in {**{k: v for k, v in group.groupby("provider")}, "project_adjusted": old,
                                 "pinned_archive": ext}.items():
                if source == other or (source == "stockanalysis" and other == "chartexchange"):
                    continue
                j = left.merge(right, on="date", suffixes=("_a", "_b"), validate="one_to_one")
                if j.empty:
                    continue
                scale = (j.close_a / j.close_b).median()
                comparisons.append({
                    "ticker": ticker, "source": source, "reference": other, "n": len(j),
                    "first": j.date.min(), "last": j.date.max(), "median_close_ratio": scale,
                    "median_close_abs_error_bps_after_scaling": ((j.close_a / j.close_b / scale - 1).abs()
                                                                 .median() * 10000),
                    "exact_close": int(j.close_a.round(2).eq(j.close_b.round(2)).sum()),
                    "exact_volume": int(j.volume_a.eq(j.volume_b).sum()),
                })
    pd.DataFrame(comparisons).to_csv(ROOT / "source_overlap.csv", index=False, mode="x")

    rows = []
    for ticker in cohort:
        exit_day = exits.get(ticker, pd.Timestamp.max)
        common = set(archive.loc[archive.ticker.eq(ticker) & ~archive.post_exit, "date"])
        recovered = set(alternate.loc[alternate.ticker.eq(ticker) & ~alternate.post_exit
                                      & alternate.valid_ohlcv, "date"])
        dates_with_bars = common | recovered
        intervals = membership[membership.ticker.eq(ticker)]
        for i, date in enumerate(sessions):
            in_index = bool((intervals.start.le(date) & (intervals.end.isna() | intervals.end.ge(date))).any())
            status = ("after_last_trade" if date > exit_day else "archive" if date in common
                      else "alternate_only" if date in recovered else "missing")
            rows.append((date, ticker, in_index, status, *(
                endpoint_status(date, schedule[i + h], exit_day, asof, dates_with_bars, common)
                for h in (1, 5)), schedule[i + 1], schedule[i + 5]))
    frame = pd.DataFrame(rows, columns=["date", "ticker", "historical_index_member", "bar_status",
                                         "status_1d", "status_5d", "target_date_1d", "target_date_5d"])
    frame.to_csv(ROOT / "session_inventory.csv", index=False, mode="x")
    manifest = []
    for path in paths:
        vendor, ticker = path.stem.split("_")
        urls = {
            "chartexchange": f"https://chartexchange.com/symbol/nasdaq-{ticker.lower()}/historical/",
            "stockanalysis": f"https://stockanalysis.com/stocks/{ticker.lower()}/history/?p=5y",
            "yahoo": f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                     "?period1=1672531200&period2=1795219200&interval=1d&events=div%2Csplits",
        }
        manifest.append({"path": str(path.relative_to(ROOT)), "url": urls[vendor], "sha256": digest(path),
                         "bytes": path.stat().st_size, "http_status": 429 if vendor == "yahoo" else 200,
                         "local_file_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime,
                                                                      tz=timezone.utc).isoformat()})
    decision = {"readiness": "FAIL", "score_2026": False, "data_through": LAST, "cohort": len(cohort),
                "reasons": ["Incomplete alternate histories", "Missing single-source candidate coverage",
                            "Unreconciled source adjustments and security-level venue/identity coverage"],
                "inputs_sha256": {str(p.relative_to(ROOT.parent.parent)): digest(p) for p in
                                  [archived_path, PRIOR / "acquisition.json", ROOT / "corporate_actions.csv",
                                   ROOT.parent.parent / "data/membership.parquet",
                                   ROOT.parent.parent / "data/prices.parquet"]},
                "sources": manifest}
    with (ROOT / "readiness.json").open("x") as out:
        json.dump(decision, out, indent=2)


if __name__ == "__main__":
    main()
