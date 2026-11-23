"""Build the isolated SEC_LAGGED_PROXY audit from reviewed offline inputs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from src.core.tickers import RENAMES, SHARE_CLASSES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PERIODIC = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
METRICS = ["revenue_growth_yoy", "operating_margin", "net_margin_change_yoy"]


def eligible_session(filed, accepted, timezone, sessions, lag):
    if lag not in (2, 5):
        raise ValueError("Use two complete sessions or the five-session sensitivity")
    boundary = pd.Timestamp(filed)
    if pd.notna(accepted):
        stamp = pd.Timestamp(accepted)
        if stamp.tzinfo is None:
            if pd.isna(timezone):
                raise ValueError("Acceptance timezone required")
            stamp = stamp.tz_localize(timezone)
        boundary = max(boundary, pd.Timestamp(stamp.tz_convert("America/New_York").date()))
    index = sessions.searchsorted(boundary, side="right") + lag
    return sessions[index] if index < len(sessions) else pd.NaT


def financial_features(facts, filing):
    rows = facts.loc[(facts.accession == filing.accession) & (facts.cik == filing.cik)]
    rows = rows.loc[rows.unit.eq("USD_millions") & rows.scope.eq("consolidated_us_gaap") &
                    rows.source.str.startswith("https://www.sec.gov/Archives/", na=False)]
    duration = (rows.end - rows.start).dt.days + 1
    rows = rows.loc[duration.between(75, 105)]
    quarter = rows.loc[rows.end.eq(filing.report_period)]
    reference = filing.report_period - pd.DateOffset(years=1)
    previous = rows.loc[(rows.end - reference).dt.days.abs().le(7)]
    if len(quarter) != 1 or quarter.revenue.iloc[0] <= 0:
        return dict.fromkeys(METRICS)
    now = quarter.iloc[0]
    result = dict.fromkeys(METRICS)
    result["operating_margin"] = now.operating_income / now.revenue
    if len(previous) == 1 and previous.revenue.iloc[0] > 0:
        past = previous.iloc[0]
        result["revenue_growth_yoy"] = now.revenue / past.revenue - 1
        result["net_margin_change_yoy"] = now.net_income / now.revenue - past.net_income / past.revenue
    return result


def assemble(keys, identities, filings, facts, sessions, lag):
    if keys.duplicated(["ticker", "date"]).any() or not keys.date.isin(sessions).all():
        raise ValueError("Expected unique market-session keys")
    if not keys.date.dt.year.between(2019, 2023).all():
        raise ValueError("Only 2019–2023 keys are permitted")
    if filings.accession.duplicated().any():
        raise ValueError("Duplicate accessions")
    events = []
    for filing in filings.itertuples(index=False):
        date = eligible_session(filing.filed_date, filing.accepted_local, filing.timezone, sessions, lag)
        supported = (filing.form in PERIODIC and pd.notna(date)
                     and filing.report_period <= filing.filed_date
                     and filing.source.startswith("https://www.sec.gov/Archives/")
                     and identities.cik.eq(filing.cik).any())
        events.append({**filing._asdict(), "eligible_date": date, "supported": supported})
        if supported:
            events[-1].update(financial_features(facts, filing))
    ledger = pd.DataFrame(events)
    positions = pd.Series(range(len(sessions)), index=sessions)
    output = keys.copy()
    output["policy"] = "SEC_LAGGED_PROXY"
    output["lag_complete_sessions"] = lag
    output["coverage_reason"] = "issuer_unresolved_for_date"
    output["cik"] = pd.NA
    output["identity_basis"] = pd.NA
    metadata = ["accession", "form", "filed_date", "accepted_local", "timezone",
                "eligible_date", "source", "report_period"]
    for col in metadata:
        output[col] = pd.NaT if col in ("filed_date", "eligible_date", "report_period") else pd.NA
    for col in [*METRICS, "filing_session_age", "filed_10k", "filed_10q"]:
        output[col] = float("nan")
    for identity in identities.itertuples(index=False):
        mask = output.ticker.eq(identity.ticker) & output.date.between(identity.start, identity.end)
        if output.loc[mask, "cik"].notna().any():
            raise ValueError("Overlapping historical issuer mapping")
        output.loc[mask, ["cik", "identity_basis"]] = [identity.cik, identity.identity_basis]
        output.loc[mask, "coverage_reason"] = "no_recent_verified_periodic_accession"
        rows = ledger.loc[ledger.cik.eq(identity.cik) & ledger.supported].sort_values(
            ["eligible_date", "report_period", "filed_date", "accepted_local", "accession"])
        for event in rows.itertuples(index=False):
            age = output.date.map(positions) - positions[event.eligible_date]
            active = mask & age.between(0, 90)
            output.loc[active, "coverage_reason"] = "sparse_proxy_snapshot"
            output.loc[active, "filing_session_age"] = age[active]
            output.loc[active, "filed_10k"] = float(event.form.startswith("10-K"))
            output.loc[active, "filed_10q"] = float(event.form.startswith("10-Q"))
            for col in metadata + METRICS:
                output.loc[active, col] = getattr(event, col, None)
    output["has_financial_facts"] = output[METRICS].notna().any(axis=1)
    output["filing_history_complete"] = False
    output["strict_public_verified"] = False
    return output, ledger


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lag", type=int, choices=(2, 5), default=2)
    args = parser.parse_args()
    paths = [HERE / f for f in ("identities.csv", "filings.csv", "facts.csv")]
    price, label = ROOT / "data/prices.parquet", ROOT / "data/labels.parquet"
    windows = [ROOT / f"experiments/gat_edge_weight_v1/runs/w{i}_h5_s42_original.parquet"
               for i in (0, 1)]
    sectors_path = ROOT / "src/graph/gics.csv"
    identities, filings, facts = (pd.read_csv(p, dtype={"cik": str}) for p in paths)
    for frame, cols in [(identities, ("start", "end")), (filings, ("filed_date", "report_period")),
                        (facts, ("start", "end"))]:
        for col in cols:
            frame[col] = pd.to_datetime(frame[col], errors="raise")
    prices = pd.read_parquet(price, columns=["ticker", "date", "in_index"])
    labels = pd.read_parquet(label, columns=["ticker", "date", "y_5d"])
    keys = prices.loc[prices.in_index & prices.date.dt.year.between(2019, 2023),
                      ["ticker", "date"]].merge(labels, validate="one_to_one")
    keys = keys.loc[keys.y_5d.notna(), ["ticker", "date"]].reset_index(drop=True)
    sessions = mcal.get_calendar("NASDAQ").schedule("2019-01-01", "2023-12-31").index
    output, ledger = assemble(keys, identities, filings, facts, sessions, args.lag)
    sectors = pd.read_csv(sectors_path).set_index("ticker").gics_sector
    records = []
    for path in windows:
        archived = pd.read_parquet(path, columns=["ticker", "date", "y_true"])
        comparison = archived.merge(labels, on=["ticker", "date"], validate="one_to_one")
        if len(comparison) != len(archived) or not comparison.y_true.eq(comparison.y_5d).all():
            raise ValueError("Assessment labels differ from archived run")
        joined = archived.merge(output, on=["ticker", "date"], validate="one_to_one")
        if len(joined) != len(archived):
            raise ValueError("Assessment population changed")
        joined["sector"] = joined.ticker.map(sectors).fillna("unclassified")
        joined["period"] = int(joined.date.dt.year.iloc[0])
        records.append(joined.drop(columns="y_true"))
    audit = pd.concat(records, ignore_index=True)
    audit["has_filing"] = audit.accession.notna()
    audit["has_issuer"] = audit.cik.notna()
    coverage = (audit.groupby(["period", "sector"], observed=True).agg(
        rows=("date", "size"), issuers_mapped=("has_issuer", "sum"),
        filing_rows=("has_filing", "sum"), financial_rows=("has_financial_facts", "sum"))
        .reset_index())
    if shutil.disk_usage(HERE).free < 2 * 1024**3 + 32 * 1024**2:
        raise RuntimeError("Insufficient space to preserve the 2 GiB safety margin")
    args.output.mkdir(parents=True, exist_ok=False)
    output.to_parquet(args.output / "features.parquet", index=False)
    ledger.to_csv(args.output / "accessions.csv", index=False)
    audit.to_parquet(args.output / "assessment_coverage.parquet", index=False)
    coverage.to_csv(args.output / "sector_coverage.csv", index=False)
    inventory = keys.groupby("ticker").date.agg(["min", "max", "size"]).reset_index()
    inventory = inventory.merge(identities, how="left", on="ticker")
    corp_path = ROOT / "experiments/final_2026_reconciliation/corporate_actions.csv"
    corp = pd.read_csv(corp_path, dtype={"cik": str}).set_index("ticker")
    inventory["corporate_ledger_cik_lead"] = inventory.ticker.map(corp.cik)
    inventory["historical_aliases"] = inventory.ticker.map(
        lambda ticker: "|".join(old for old, new in RENAMES.items() if new == ticker))
    inventory["share_class_group"] = inventory.ticker.map(
        lambda ticker: next(("|".join(group) for group in SHARE_CLASSES if ticker in group), ""))
    inventory["mapping_status"] = inventory.cik.notna().map(
        {True: "reviewed_partial_interval", False: "unresolved"})
    inventory.to_csv(args.output / "issuer_inventory.csv", index=False)
    hashes = {str(p.relative_to(ROOT)): _sha256(p)
              for p in [*paths, price, label, *windows, sectors_path, corp_path,
                        ROOT / "src/core/tickers.py", Path(__file__).resolve()]}
    (args.output / "manifest.json").write_text(json.dumps({
        "policy": "SEC_LAGGED_PROXY", "complete_sessions_lag": args.lag,
        "sparse_sample": True, "training_ready": False, "sha256": hashes}, indent=2) + "\n")
    print(audit.groupby("period")[["has_issuer", "has_filing", "has_financial_facts"]].sum())


if __name__ == "__main__":
    main()
