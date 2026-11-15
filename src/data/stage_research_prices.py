"""Stage checksum-pinned research daily bars for independent review; never overwrite artifacts."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.io import read_table
from src.core.paths import raw_dir
from src.core.tickers import canonical_ticker
from src.data.fetch_prices import _bounds, add_in_index

REVISION = "79458172fb5190d72ab539e658e428c6810772f7"
EXPECTED_SHA256 = "c1232a3d6764d1bcf737bdfe8419c5e40c0e4c9d31ca85a8d9c87507b6b54beb"
SOURCE_URL = ("https://huggingface.co/datasets/Liangzhi-Li/quant-papers-vs-baselines-data/resolve/"
              f"{REVISION}/r3_data/us/daily_merged.parquet")
SOURCE_IDS = {"FOXA~201903": "TFCFA", "FOX~201903": "TFCF",
              "CA~201811": "CA", "ESRX~201812": "ESRX"}
SOURCE_COLUMNS = ["date", "ticker", "aopen", "ahigh", "alow", "aclose", "volume", "source"]


def prepare_daily_candidates(frame: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Use documented adjusted fields and explicit retired-issuer identities."""
    if set(SOURCE_COLUMNS) - set(frame):
        raise ValueError(f"Research daily input needs {SOURCE_COLUMNS}")
    out = frame[SOURCE_COLUMNS].copy()
    out["source_id"] = out.ticker.astype("string")
    out["ticker"] = out.ticker.map(lambda value: canonical_ticker(SOURCE_IDS.get(value, value)))
    out = out.rename(columns={"aopen": "open", "ahigh": "high", "alow": "low", "aclose": "close"})
    provenance = out[["source_id", "source"]].reset_index(drop=True)
    checked = add_in_index(out, membership)
    return pd.concat([checked, provenance], axis=1)


def compare_candidates(existing: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Price/return discrepancies remain review findings, never automatic corrections."""
    findings = []
    for ticker, group in candidates.groupby("ticker"):
        common = existing.loc[existing.ticker.eq(ticker), ["date", "close", "volume"]].merge(
            group[["date", "close", "volume"]], on="date", suffixes=("_old", "_candidate"),
        ).sort_values("date")
        ratio_drift = return_error = np.nan
        if len(common) > 1:
            ratio = common.close_old / common.close_candidate
            ratio_drift = float((ratio / ratio.median() - 1).abs().max())
            return_error = float((common.close_old.pct_change(fill_method=None) -
                                  common.close_candidate.pct_change(fill_method=None)).abs().max())
        findings.append({"ticker": ticker, "candidate_rows": len(group), "overlap_rows": len(common),
                         "first_date": group.date.min(), "last_date": group.date.max(),
                         "normalised_close_ratio_max_drift": ratio_drift,
                         "max_overlap_return_difference": return_error,
                         "source": ",".join(sorted(group.source.unique())),
                         "decision": "review_required_not_applied"})
    return pd.DataFrame(findings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True,
                        help="downloaded daily_merged.parquet, never the final artifact")
    parser.add_argument("--tickers", nargs="+", help="default: current recovery_requests.csv symbols")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01", help="exclusive final date")
    args = parser.parse_args()
    start, end = _bounds(args.start, args.end)
    digest = sha256()
    with args.input.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError("Research source checksum mismatch; use the pinned documented revision")
    tickers = {canonical_ticker(t) for t in (args.tickers or pd.read_csv(raw_dir() / "recovery_requests.csv").ticker)}
    source_ids = sorted(tickers | {old for old, new in SOURCE_IDS.items() if new in tickers})
    # This Parquet is a raw vendor input, not a frozen project artifact.
    frame = pd.read_parquet(args.input, columns=SOURCE_COLUMNS,
                            filters=[("ticker", "in", source_ids), ("date", ">=", start), ("date", "<", end)])
    if frame.empty:
        raise ValueError("No candidate rows for the requested symbols/window")
    candidates = prepare_daily_candidates(frame, read_table("membership"))
    comparison = compare_candidates(read_table("prices"), candidates)
    folder = raw_dir() / "research_stage"
    folder.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(folder / "candidate_prices.csv", index=False, date_format="%Y-%m-%d")
    comparison.to_csv(folder / "comparison.csv", index=False)
    (folder / "provenance.json").write_text(json.dumps({
        "source_url": SOURCE_URL, "sha256": digest.hexdigest(), "revision": REVISION,
        "source_adjustment": "aopen/ahigh/alow/aclose are source-documented split/dividend-adjusted fields",
        "known_issues": ["Alpaca SIP volume convention differs from Yahoo",
                         "Reported daily close discrepancies around March 2020 circuit-breaker sessions",
                         "Corporate-action calibration and issuer mappings require independent review"],
        "status": "STAGED_ONLY: prices.parquet and labels.parquet were not modified",
    }, indent=2) + "\n")
    print(f"Staged {len(candidates)} candidate rows for {candidates.ticker.nunique()} tickers in {folder}")
    print("Review source identity, adjustment and overlap differences before applying any changes.")


if __name__ == "__main__":
    main()
