"""Download adjusted daily OHLCV and write membership-aware prices (issues #7, #14).

    python -m src.data.fetch_prices --start 2018-01-01 --end 2026-11-30
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.core.io import read_table, write_table
from src.core.paths import raw_dir
from src.core.tickers import canonical_ticker
from src.data.build_index_history import trading_days
from src.data.validation import checked_membership, checked_prices, invalid_ohlc

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
RAW_COLUMNS = ["date", "ticker", *PRICE_COLUMNS]
DEFAULT_END = "2026-11-30"


def _bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    for date in (start_date, end_date):
        if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
            raise ValueError("start/end must be timezone-naive midnight dates")
    if start_date >= end_date:
        raise ValueError("start must be before exclusive end")
    return start_date, end_date


def normalize_prices(frame: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    """Handle either yfinance column ordering and preserve Eastern session dates."""
    ticker = canonical_ticker(ticker)
    start_date, end_date = _bounds(start, end)
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        field_levels = [i for i in range(frame.columns.nlevels)
                        if set(PRICE_COLUMNS) <= {str(v).lower() for v in frame.columns.get_level_values(i)}]
        if len(field_levels) != 1:
            raise ValueError("Cannot identify OHLCV level in yfinance columns")
        field_level = field_levels[0]
        for i in range(frame.columns.nlevels):
            if i != field_level:
                symbols = {canonical_ticker(str(v)) for v in frame.columns.get_level_values(i)}
                if symbols != {ticker}:
                    raise ValueError(f"Unexpected ticker columns: {symbols}")
        frame.columns = frame.columns.get_level_values(field_level)
    frame.columns = [str(c).lower() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate price columns")
    out = frame.loc[:, PRICE_COLUMNS].astype("float64").copy()
    dates = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if dates.tz is not None:
        dates = dates.tz_convert("America/New_York").tz_localize(None)
    dates = dates.normalize().astype("datetime64[ns]")
    out.insert(0, "ticker", pd.array([ticker] * len(out), dtype="string"))
    out.insert(0, "date", dates)
    out = out.loc[(out.date >= start_date) & (out.date < end_date)].reset_index(drop=True)
    if out.empty:
        raise ValueError("Empty OHLCV in requested range")
    if out.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate price dates")
    if not out.date.isin(trading_days(start_date, end_date)).all():
        raise ValueError("Price dates outside NASDAQ sessions")
    if not np.isfinite(out[PRICE_COLUMNS].to_numpy()).all():
        raise ValueError("Null or non-finite OHLCV; ticker needs data-quality review")
    if (out[["open", "high", "low", "close"]] <= 0).any(axis=None) or (out.volume < 0).any():
        raise ValueError("Non-positive price or negative volume")
    if invalid_ohlc(out).any():
        raise ValueError("Inconsistent OHLC range")
    return out.sort_values("date").reset_index(drop=True)


def download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch separately; failure details are in result.attrs['failed_tickers']."""
    _bounds(start, end)
    import yfinance as yf

    frames, failures = [], []
    symbols = sorted({canonical_ticker(t) for t in tickers})
    for position, ticker in enumerate(symbols, start=1):
        try:
            frame = yf.download(
                ticker, start=start, end=end, auto_adjust=True,
                progress=False, threads=False, timeout=30,
            )
            if frame is None or frame.empty:
                raise ValueError("Empty download (possibly delisted or renamed)")
            frames.append(normalize_prices(frame, ticker, start, end))
            print(f"[{position}/{len(symbols)}] {ticker}: {len(frames[-1])} rows", flush=True)
        except Exception as exc:  # isolate provider and data failures per ticker
            failures.append({"ticker": ticker, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {ticker}: {exc}")
    if frames:
        result = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    else:
        result = pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype="string"),
            **{col: pd.Series(dtype="float64") for col in PRICE_COLUMNS},
        })
    result.attrs["failed_tickers"] = failures
    return result


def add_in_index(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Inclusive spell boundaries; keep every warm-up and out-of-index row."""
    members = checked_membership(membership)
    out = checked_prices(prices.assign(in_index=False))
    unknown = set(out.ticker) - set(members.ticker)
    if unknown:
        raise ValueError(f"Prices contain tickers absent from membership: {sorted(unknown)}")
    for row in members.itertuples(index=False):
        mask = out.ticker.eq(row.ticker) & out.date.ge(row.start)
        if pd.notna(row.end):
            mask &= out.date.le(row.end)
        out.loc[mask, "in_index"] = True
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=DEFAULT_END, help="exclusive end (README default; CHECKLIST says 2023-12-31)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--finalize-only", action="store_true", help="use raw/prices_raw.csv without network")
    mode.add_argument("--retry-failed", action="store_true",
                      help="retry failed_tickers.csv, retaining successful raw rows")
    args = p.parse_args()
    _bounds(args.start, args.end)
    membership = read_table("membership")
    tickers = sorted({canonical_ticker(t) for t in membership.ticker})
    if not tickers:
        raise ValueError("Membership is empty")
    if args.finalize_only:
        raw = pd.read_csv(raw_dir() / "prices_raw.csv", parse_dates=["date"])
        if raw.empty:
            raise ValueError("Raw prices are empty")
        path = write_table(add_in_index(raw, membership), "prices")
        print(f"{path}: {len(raw)} rows (no network)")
        return
    previous = None
    if args.retry_failed:
        previous = pd.read_csv(raw_dir() / "prices_raw.csv", parse_dates=["date"])
        failed = pd.read_csv(raw_dir() / "failed_tickers.csv")
        tickers = sorted({canonical_ticker(t) for t in failed.ticker})
        if not tickers:
            print("No failed tickers to retry")
            return
        if not previous.empty and not previous.date.between(
            pd.Timestamp(args.start), pd.Timestamp(args.end), inclusive="left"
        ).all():
            raise ValueError("Existing raw prices extend outside retry bounds; use the original download range")
    prices = download(tickers, args.start, args.end)
    failures = pd.DataFrame(prices.attrs["failed_tickers"], columns=["ticker", "reason"])
    downloaded_count, downloaded_rows = prices.ticker.nunique(), len(prices)
    if previous is not None and not previous.empty:
        previous["ticker"] = previous.ticker.map(canonical_ticker)
        # Retried tickers are replaced as a whole; never merge stale and new histories.
        previous = previous.loc[~previous.ticker.isin(prices.ticker.unique())]
        prices = pd.concat([previous, prices], ignore_index=True)
    folder = raw_dir()
    folder.mkdir(parents=True, exist_ok=True)
    # Raw intermediates have no frozen table contract; CSV avoids inventing a Parquet schema.
    prices.to_csv(folder / "prices_raw.csv", index=False, date_format="%Y-%m-%d")
    failures.to_csv(folder / "failed_tickers.csv", index=False)
    print(f"Downloaded {downloaded_count}/{len(tickers)} tickers, {downloaded_rows} rows this run; "
          f"failed {len(failures)}. End {args.end} is exclusive.")
    print(f"Raw prices: {folder / 'prices_raw.csv'}; failures: {folder / 'failed_tickers.csv'}")
    if prices.empty:
        raise RuntimeError("All tickers failed; see raw/failed_tickers.csv")
    final = add_in_index(prices, membership)
    path = write_table(final, "prices")
    print(f"{path}: {len(final)} rows, {int(final.in_index.sum())} in-index rows")


if __name__ == "__main__":
    main()
