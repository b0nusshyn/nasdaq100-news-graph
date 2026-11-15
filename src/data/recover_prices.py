"""Recover missing ticker histories from reviewed archived Yahoo CSVs (Data).

No network: imports a checksum-pinned manifest after identity review, then rebuilds
prices and labels. Existing ticker histories are never spliced across vintages.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.io import read_table, write_table
from src.core.paths import raw_dir, table_path
from src.core.tickers import canonical_ticker
from src.data.build_labels import make_labels
from src.data.fetch_prices import RAW_COLUMNS, _bounds, add_in_index, normalize_prices
from src.data.validation import checked_prices


def archived_prices(frame: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    """Archived Yahoo OHLC is split-adjusted; adj close adds dividend adjustment.

    Multiply each OHLC by adj_close / close, as Yahoo auto_adjust does; retain
    archived volume. Validate after date filtering; never fill gaps or prices.
    """
    _bounds(start, end)
    frame = frame.copy()
    frame.columns = [str(c).strip().lower().replace('_', ' ') for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate archive columns")
    required = ["date", "open", "high", "low", "close", "adj close", "volume"]
    if set(required) - set(frame):
        raise ValueError(f"Archive needs {required}")
    frame["date"] = pd.to_datetime(frame.date).astype("datetime64[ns]")
    if frame.date.isna().any() or frame.date.ne(frame.date.dt.normalize()).any():
        raise ValueError("Archive dates must be non-null midnight dates")
    frame = frame.loc[frame.date.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="left")]
    numeric = frame[["open", "high", "low", "close", "adj close", "volume"]].astype("float64")
    if not np.isfinite(numeric.to_numpy()).all() or numeric[["close", "adj close"]].le(0).any(axis=None):
        raise ValueError("Invalid archive price/adjustment factor")
    factor = numeric["adj close"] / numeric.close
    for field in ["open", "high", "low", "close"]:
        numeric[field] *= factor
    numeric["close"] = numeric["adj close"]
    numeric.index = pd.DatetimeIndex(frame.date)
    return normalize_prices(numeric, canonical_ticker(ticker), start, end)


def load_reviewed_sources(manifest: Path, start: str, end: str) -> tuple[pd.DataFrame, list[dict]]:
    """Manifest must pin file hash, source URL and reviewed issuer-identity evidence.

    Filename equality alone is insufficient: ticker symbols can be reused by
    unrelated companies, as the archive's GEN history demonstrates.
    """
    from urllib.parse import urlparse

    entries = pd.read_csv(manifest, keep_default_na=False)
    columns = {"ticker", "file", "sha256", "source_url", "identity_url", "identity_review"}
    if columns - set(entries) or entries.empty:
        raise ValueError(f"Nonempty manifest needs {sorted(columns)}")
    frames, provenance = [], []
    seen = set()
    for entry in entries.to_dict("records"):
        ticker = canonical_ticker(entry["ticker"])
        if ticker in seen:
            raise ValueError(f"Multiple archive vintages for {ticker}")
        seen.add(ticker)
        if entry["identity_review"] != "reviewed":
            raise ValueError(f"Identity review required for {ticker}")
        for field in ["source_url", "identity_url"]:
            parsed = urlparse(entry[field])
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError(f"Missing HTTPS {field} for {ticker}")
        source = (manifest.parent / entry["file"]).resolve()
        # A manifest may only consume files within its own directory tree.
        if not source.is_relative_to(manifest.parent.resolve()):
            raise ValueError("Archive file escapes manifest folder")
        digest = sha256(source.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {ticker}")
        converted = archived_prices(pd.read_csv(source), ticker, start, end)
        frames.append(converted)
        provenance.append({**entry, "ticker": ticker, "rows": len(converted),
                           "first_date": str(converted.date.min().date()),
                           "last_date": str(converted.date.max().date())})
    return pd.concat(frames, ignore_index=True), provenance


def merge_missing_histories(
    existing: pd.DataFrame, supplemental: pd.DataFrame, membership: pd.DataFrame,
) -> pd.DataFrame:
    """Accept only entirely missing ticker series; preserve existing values exactly."""
    # Membership can gain reviewed issuer splits since the previous price build.
    # Recompute flags while preserving all existing OHLCV values and keys.
    existing = add_in_index(existing, membership)
    supplemental = add_in_index(supplemental, membership)
    overlap = set(existing.ticker) & set(supplemental.ticker)
    if overlap:
        raise ValueError(f"Refusing to splice archive/current adjustment vintages for {sorted(overlap)}")
    return checked_prices(pd.concat([existing, supplemental], ignore_index=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-01-01", help="exclusive end")
    p.add_argument("--apply", action="store_true", help="write reviewed recovery; default validates and previews")
    args = p.parse_args()
    supplemental, provenance = load_reviewed_sources(args.manifest, args.start, args.end)
    combined = merge_missing_histories(read_table("prices"), supplemental, read_table("membership"))
    labels = make_labels(combined)
    print(f"Recovery: {supplemental.ticker.nunique()} tickers, {len(supplemental)} archived rows; "
          f"combined {len(combined)} rows. Coverage can remain incomplete.")
    if not args.apply:
        print("Preview only; use --apply after reviewing the manifest")
        return
    folder = raw_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = folder / "snapshots" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    paths = [table_path("prices"), table_path("labels"), folder / "prices_raw.csv",
             folder / "download_manifest.json", folder / "quality_report.csv", folder / "recovery_provenance.json"]
    for path in paths:
        if path.exists():
            shutil.copy2(path, backup / path.name)
    write_table(combined, "prices")
    write_table(labels, "labels")
    combined[RAW_COLUMNS].to_csv(folder / "prices_raw.csv", index=False, date_format="%Y-%m-%d")
    # Preserve Yahoo's failure list: archived coverage may be only partial.
    (folder / "recovery_provenance.json").write_text(json.dumps({
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "backup": str(backup),
        "previous_provenance_backup": str(backup / "recovery_provenance.json")
        if (backup / "recovery_provenance.json").exists() else None,
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "requested_start": args.start, "requested_end_exclusive": args.end,
        "sources": provenance,
    }, indent=2) + "\n")
    print(f"Wrote prices and labels; previous versions saved in {backup}")


if __name__ == "__main__":
    main()
