"""Reconstruct historical NASDAQ-100 membership from Wikipedia (issue #6)."""

from __future__ import annotations

import argparse
import re
from io import StringIO
from pathlib import Path

import pandas as pd

from src.core.io import write_table
from src.core.paths import CONFIGS_DIR
from src.core.schema import conform, validate
from src.core.tickers import canonical_ticker

WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_Nasdaq-100"
WINDOW_START = pd.Timestamp("2018-01-01")


def trading_days(start, end) -> pd.DatetimeIndex:
    """NASDAQ session dates, including holidays and exceptional closures."""
    import pandas_market_calendars as mcal

    days = mcal.get_calendar("NASDAQ").valid_days(start_date=start, end_date=end)
    return days.tz_localize(None).astype("datetime64[ns]")


def _tables(url: str) -> list[pd.DataFrame]:
    import requests

    response = requests.get(
        url, headers={"User-Agent": "nasdaq100-news-graph/1.0 (academic research)"}, timeout=30
    )
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def _columns(table: pd.DataFrame) -> list[str]:
    return [" ".join(dict.fromkeys(map(str, c))).lower() if isinstance(c, tuple)
            else str(c).lower() for c in table.columns]


def _symbol(value) -> str | None:
    if pd.isna(value) or str(value).strip() in {"", "—", "–", "-"}:
        return None
    value = re.sub(r"\[[^\]]*\]", "", str(value)).strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9.\-]*", value):
        raise ValueError(f"Unrecognised ticker cell: {value!r}")
    return canonical_ticker(value)


def parse_changes(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Select by headers, never by table number; reject ambiguous layouts."""
    matches = []
    for table in tables:
        cols = _columns(table)
        dates = [i for i, c in enumerate(cols) if c == "date"]
        added = [i for i, c in enumerate(cols) if c == "added" or ("added" in c and "ticker" in c)]
        removed = [i for i, c in enumerate(cols) if c == "removed" or ("removed" in c and "ticker" in c)]
        if len(dates) == len(added) == len(removed) == 1:
            out = table.iloc[:, [dates[0], added[0], removed[0]]].copy()
            out.columns = ["date", "added", "removed"]
            out["date"] = pd.to_datetime(
                out["date"].astype(str).str.replace(r"\[[^\]]*\]", "", regex=True)
            )
            for col in ("added", "removed"):
                out[col] = out[col].map(_symbol)
            # Wikipedia sometimes inserts a completely blank separator row.
            # Only discard wholly empty rows; missing dates on actual events still fail.
            out = out.dropna(how="all").reset_index(drop=True)
            matches.append(out)
    if len(matches) != 1:
        raise ValueError(f"Expected one changes table, found {len(matches)}; "
                         f"headers: {[_columns(t) for t in tables]}")
    return matches[0]


def parse_current(tables: list[pd.DataFrame]) -> pd.DataFrame:
    matches = []
    for table in tables:
        cols = _columns(table)
        tickers = [i for i, c in enumerate(cols) if c in {"ticker", "symbol"}]
        if len(tickers) == 1 and any(c in {"company", "security"} for c in cols):
            out = table.copy()
            out.columns = cols
            out = out.rename(columns={cols[tickers[0]]: "ticker"})
            out["ticker"] = out["ticker"].map(_symbol).astype("string")
            matches.append(out)
    if len(matches) != 1:
        raise ValueError(f"Expected one current-components table, found {len(matches)}")
    return matches[0]


def _scrape(parser, fallback: str) -> pd.DataFrame:
    tables = _tables(WIKI_URL)
    try:
        return parser(tables)
    except ValueError:
        # Wikipedia split the original article into separate list/history pages.
        return parser(_tables(fallback))


def scrape_changes() -> pd.DataFrame:
    return _scrape(parse_changes, CHANGES_URL)


def scrape_current() -> pd.DataFrame:
    """Preserve source classifications; ICB must not be relabelled as GICS."""
    return _scrape(parse_current, CURRENT_URL)


def apply_reviewed_events(changes: pd.DataFrame, current: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Supplement omitted events with cited issuer-aware corrections, never a global rename.

    Ticker reuse (old/new Fox) must create separate entities and spells. Only
    corrections touching a symbol present in this source universe are applied.
    A conflicting source event stops the build instead of silently replacing it.
    """
    required = {"date", "added", "removed", "source_url", "identity_url", "reason"}
    if required - set(corrections):
        raise ValueError(f"Reviewed events need {sorted(required)}")
    out = changes.copy()
    out["date"] = pd.to_datetime(out.date)
    universe = set(current.ticker.map(canonical_ticker))
    for field in ["added", "removed"]:
        out[field] = out[field].map(_symbol)
        universe.update(out[field].dropna())
    for row in corrections.itertuples(index=False):
        added, removed, date = _symbol(row.added), _symbol(row.removed), pd.Timestamp(row.date)
        if added not in universe:
            continue
        if not (str(row.source_url).startswith("https://") and str(row.identity_url).startswith("https://")
                and pd.notna(row.reason) and str(row.reason).strip()):
            raise ValueError("Correction needs source, identity evidence and reason")
        on_date = out.loc[out.date.eq(date)]
        if (on_date.added.eq(added) & on_date.removed.eq(removed)).any():
            continue
        if on_date.added.eq(added).any() or on_date.removed.eq(removed).any():
            raise ValueError(f"Reviewed event conflicts with source on {date.date()}: {added}/{removed}")
        out = pd.concat([out, pd.DataFrame({"date": [date], "added": [added], "removed": [removed]})],
                        ignore_index=True)
    return out


def membership_intervals(changes: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Reverse effective-date events; retain separate spells for departures/rejoins."""
    symbols = current["ticker"].map(_symbol)
    if symbols.isna().any() or symbols.duplicated().any() or symbols.empty:
        raise ValueError("Current members must be nonempty, unique canonical tickers")
    events = changes.copy()
    events["date"] = pd.to_datetime(events["date"]).astype("datetime64[ns]")
    if events["date"].isna().any() or (events["date"] != events["date"].dt.normalize()).any():
        raise ValueError("Event dates must be non-null midnight dates")
    for col in ("added", "removed"):
        events[col] = events[col].map(_symbol)
    events = events.loc[events["date"] >= WINDOW_START]
    last = max(events["date"].max(), pd.Timestamp("2023-12-31")) if not events.empty else pd.Timestamp("2023-12-31")
    sessions = trading_days(WINDOW_START - pd.Timedelta(days=10), last + pd.Timedelta(days=10))
    active = dict.fromkeys(symbols, pd.NaT)  # ticker -> inclusive end of spell
    rows = []
    if events[["added", "removed"]].isna().all(axis=1).any():
        raise ValueError("Empty membership event")
    for date, group in events.sort_values("date", ascending=False).groupby("date", sort=False):
        effective = sessions[sessions >= date][0]
        previous = sessions[sessions < effective][-1]
        adds = group["added"].dropna().tolist()
        removes = group["removed"].dropna().tolist()
        if len(set(adds)) != len(adds) or len(set(removes)) != len(removes):
            raise ValueError(f"Duplicate event on {date.date()}")
        # Canonicalised rename events do not break continuous membership.
        renames = set(adds) & set(removes)
        for ticker in renames:
            if ticker not in active:
                raise ValueError(f"Rename for inactive member {ticker} on {date.date()}")
        for ticker in set(adds) - renames:
            if ticker not in active:
                raise ValueError(f"Added {ticker} on {date.date()} but not a member afterwards")
            end = active.pop(ticker)
            if pd.notna(end) and effective > end:
                raise ValueError(f"Invalid spell for {ticker}")
            rows.append((ticker, effective, end))
        for ticker in set(removes) - renames:
            if ticker in active:
                raise ValueError(f"Removed {ticker} on {date.date()} but still a member afterwards")
            active[ticker] = previous
    # Explicit left-censoring sentinel in the requested contract (a market holiday).
    rows.extend((ticker, WINDOW_START, end) for ticker, end in active.items()
                if pd.isna(end) or end >= WINDOW_START)
    out = conform(pd.DataFrame(rows, columns=["ticker", "start", "end"]), "membership")
    validate(out, "membership")
    out = out.sort_values(["ticker", "start"]).reset_index(drop=True)
    for ticker, group in out.groupby("ticker"):
        ends = group["end"].iloc[:-1].fillna(pd.Timestamp.max)
        if (ends.to_numpy() >= group["start"].iloc[1:].to_numpy()).any():
            raise ValueError(f"Overlapping spells for {ticker}")
    return out


def validate_member_counts(membership: pd.DataFrame) -> pd.Series:
    days = trading_days("2019-01-01", "2023-12-31")
    counts = pd.Series(0, index=days, name="members")
    for row in membership.itertuples(index=False):
        counts += ((days >= row.start) & (pd.isna(row.end) | (days <= row.end))).astype(int)
    bad = counts.loc[~counts.between(100, 103)]
    if not bad.empty:
        raise ValueError(f"Daily membership outside 100–103: {bad.head().to_dict()}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--changes-csv", type=Path, help="previously downloaded Wikipedia changes (offline)")
    parser.add_argument("--current-csv", type=Path, help="matching current-components snapshot (offline)")
    parser.add_argument("--reviewed-events", type=Path, default=CONFIGS_DIR / "membership_events.csv",
                        help="cited omitted events, including the distinct old/new Fox issuers")
    args = parser.parse_args()
    if bool(args.changes_csv) != bool(args.current_csv):
        parser.error("--changes-csv and --current-csv must be supplied together")
    if args.changes_csv:
        changes = pd.read_csv(args.changes_csv, parse_dates=["date"]).dropna(how="all")
        current = pd.read_csv(args.current_csv)
    else:
        changes, current = scrape_changes(), scrape_current()
    if args.reviewed_events.exists():
        changes = apply_reviewed_events(changes, current, pd.read_csv(args.reviewed_events))
    else:
        raise FileNotFoundError(f"Reviewed membership events not found: {args.reviewed_events}")
    membership = membership_intervals(changes, current)
    counts = validate_member_counts(membership)
    path = write_table(membership, "membership")
    print(f"{path}: {len(membership)} spells, {membership['ticker'].nunique()} tickers; "
          f"2019–2023 daily members {counts.min()}–{counts.max()}")


if __name__ == "__main__":
    main()
