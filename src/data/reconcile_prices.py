"""Reconcile reviewed daily recovery with archived quotes and one adjustment/volume basis per series."""

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
from src.data.fetch_prices import RAW_COLUMNS, _bounds, add_in_index
from src.data.stage_research_prices import EXPECTED_SHA256, SOURCE_COLUMNS, SOURCE_IDS, SOURCE_URL


def reconcile_quote_history(
    daily: pd.DataFrame, archived_quote: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    """Use the daily source's adjustment factor throughout the recovered series.

    Where checksum-pinned Yahoo quote OHLC exists, keep those quote observations
    and apply the source factor; elsewhere use published adjusted OHLC. Volume
    always uses the daily source's SIP convention, avoiding a volume-basis splice.
    No adjustment is estimated from future returns or fitted to a model.
    """
    required = {*SOURCE_COLUMNS, "open", "high", "low", "close", "adj_factor"}
    if required - set(daily):
        raise ValueError(f"Daily recovery needs {sorted(required)}")
    if daily.empty or daily.ticker.nunique() != 1 or daily.date.duplicated().any():
        raise ValueError("Recovery must contain one nonempty unique ticker series")
    out = daily[["date", "ticker", "aopen", "ahigh", "alow", "aclose", "volume"]].rename(
        columns={"aopen": "open", "ahigh": "high", "alow": "low", "aclose": "close"}
    ).reset_index(drop=True)
    factor = daily.adj_factor.to_numpy(dtype=float)
    if not np.isfinite(factor).all() or (factor <= 0).any():
        raise ValueError("Invalid published adjustment factors")
    for field in ["open", "high", "low", "close"]:
        if not np.allclose(daily[field].to_numpy() * factor, daily[f'a{field}'].to_numpy(),
                           rtol=1e-8, atol=1e-8):
            raise ValueError("Published adjusted OHLC disagrees with adjustment factor")
    summary = {"rows": len(out), "archived_quote_rows_used": 0, "max_raw_close_relative_difference": None}
    if archived_quote is not None:
        reference = archived_quote.copy()
        reference.columns = [str(c).strip().lower() for c in reference.columns]
        reference["date"] = pd.to_datetime(reference.date).astype("datetime64[ns]")
        if reference.date.duplicated().any():
            raise ValueError("Duplicate archived quote dates")
        overlap = daily.reset_index(drop=True).reset_index(names="position").merge(
            reference[["date", "open", "high", "low", "close"]], on="date", suffixes=("_daily", "_archive"),
        )
        if len(overlap) < 20:
            raise ValueError("Insufficient raw-quote overlap for identity/unit comparison")
        delta = (overlap.close_archive / overlap.close_daily - 1).abs()
        if delta.median() > 0.001 or delta.quantile(0.99) > 0.001 or delta.max() > 0.01:
            raise ValueError("Raw quote units or issuer identity disagree; manual review required")
        positions = overlap.position.to_numpy()
        for field in ["open", "high", "low", "close"]:
            out.loc[positions, field] = overlap[f'{field}_archive'].to_numpy() * overlap.adj_factor.to_numpy()
        summary.update(archived_quote_rows_used=len(overlap), max_raw_close_relative_difference=float(delta.max()))
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="reviewed ticker/source-id/issuer evidence CSV")
    parser.add_argument("--quote-manifest", type=Path, required=True,
                        help="checksum-pinned archived Yahoo quote manifest")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01", help="exclusive end")
    parser.add_argument("--apply", action="store_true", help="default: validate and write a review summary only")
    args = parser.parse_args()
    start, end = _bounds(args.start, args.end)
    digest = sha256()
    with args.input.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError("Research source checksum mismatch")
    reviewed = pd.read_csv(args.manifest, keep_default_na=False)
    required = {"ticker", "source_id", "identity_url", "identity_review"}
    if required - set(reviewed) or reviewed.empty or reviewed.ticker.duplicated().any():
        raise ValueError("A nonempty unique reviewed issuer manifest is required")
    references = pd.read_csv(args.quote_manifest, keep_default_na=False)
    reference_map = {canonical_ticker(row.ticker): row for row in references.itertuples(index=False)}
    source_ids = reviewed.source_id.tolist()
    daily = pd.read_parquet(args.input, columns=list(dict.fromkeys([
        *SOURCE_COLUMNS, "open", "high", "low", "close", "adj_factor",
    ])), filters=[("ticker", "in", source_ids), ("date", ">=", start), ("date", "<", end)])
    frames, summaries = [], []
    for row in reviewed.itertuples(index=False):
        ticker = canonical_ticker(row.ticker)
        if row.identity_review != "reviewed" or not row.identity_url.startswith("https://"):
            raise ValueError(f"Identity review is missing for {ticker}")
        if canonical_ticker(SOURCE_IDS.get(row.source_id, row.source_id)) != ticker:
            raise ValueError(f"Unexpected source identity for {ticker}")
        group = daily.loc[daily.ticker.eq(row.source_id)].copy()
        group["date"] = group.date.astype("datetime64[ns]")
        group["ticker"] = ticker
        reference = None
        if ticker in reference_map:
            ref = reference_map[ticker]
            path = (args.quote_manifest.parent / ref.file).resolve()
            if not path.is_relative_to(args.quote_manifest.parent.resolve()):
                raise ValueError("Quote reference escapes manifest directory")
            if sha256(path.read_bytes()).hexdigest() != ref.sha256 or ref.identity_review != "reviewed":
                raise ValueError(f"Quote checksum/identity mismatch for {ticker}")
            reference = pd.read_csv(path)
        recovered, summary = reconcile_quote_history(group, reference)
        frames.append(recovered)
        summaries.append({"ticker": ticker, "source_id": row.source_id, "identity_url": row.identity_url, **summary})
    supplemental = pd.concat(frames, ignore_index=True)
    existing, membership = read_table("prices"), read_table("membership")
    selected = {canonical_ticker(t) for t in reviewed.ticker}
    old_keys = existing.loc[existing.ticker.isin(selected), ["date", "ticker"]]
    retained = old_keys.merge(supplemental[["date", "ticker"]], on=["date", "ticker"], how="left", indicator=True)
    if not retained["_merge"].eq("both").all():
        raise ValueError("Recovery would discard existing price keys; refused")
    combined = add_in_index(pd.concat([existing.loc[~existing.ticker.isin(selected)], supplemental], ignore_index=True),
                            membership)
    labels = make_labels(combined)
    folder = raw_dir() / "research_stage"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(folder / "reconciliation_review.csv", index=False)
    print(f"Reviewed {len(selected)} series; {len(combined)} combined rows; added {len(combined)-len(existing)} keys")
    if not args.apply:
        print("Preview only; inspect research_stage/reconciliation_review.csv before --apply")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = raw_dir() / "snapshots" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    for path in [table_path("prices"), table_path("labels"), raw_dir() / "prices_raw.csv",
                 raw_dir() / "download_manifest.json", raw_dir() / "quality_report.csv",
                 raw_dir() / "recovery_provenance.json"]:
        if path.exists():
            shutil.copy2(path, backup / path.name)
    write_table(combined, "prices")
    write_table(labels, "labels")
    combined[RAW_COLUMNS].to_csv(raw_dir() / "prices_raw.csv", index=False, date_format="%Y-%m-%d")
    (raw_dir() / "recovery_provenance.json").write_text(json.dumps({
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "source_url": SOURCE_URL,
        "source_sha256": digest.hexdigest(), "backup": str(backup),
        "previous_provenance_backup": str(backup / "recovery_provenance.json"),
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "quote_manifest_sha256": sha256(args.quote_manifest.read_bytes()).hexdigest(), "sources": summaries,
        "price_basis": "Published adjustment factors; pinned Yahoo quote OHLC on overlaps, daily source elsewhere",
        "volume_basis": "Daily source SIP convention throughout each recovered series",
        "limitations": "Provider adjustment rounding and source-only quote discrepancies still require QA",
    }, indent=2) + "\n")
    print(f"Wrote prices/labels with consistent per-series basis; backup: {backup}")


if __name__ == "__main__":
    main()
