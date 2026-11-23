"""Rebuild assessment and full coverage using local, reviewed SEC evidence."""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from experiments.sec_acquisition_v1.submissions import read_candidates
from experiments.sec_lagged_proxy.build import HERE as BASE
from experiments.sec_lagged_proxy.build import ROOT, _sha256, assemble

HERE = Path(__file__).resolve().parent



def add_cached_filings(identities, filings, cache):
    """Use SEC submissions metadata for filing events, never for financial facts."""
    rows = []
    sources = set()
    for cik, group in identities.groupby("cik"):
        if not (cache / f"CIK{cik}.json").exists():
            continue
        for record in read_candidates(cache, cik):
            sources.add(cache / record["source"].rsplit("/", 1)[-1])
            if record["form"] not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
                continue
            filed = pd.Timestamp(record["filed"])
            interval = group.loc[group.start.le(filed) & group.end.ge(filed)]
            stamp = pd.Timestamp(record["accepted_raw"])
            period = pd.Timestamp(record["period"])
            if interval.empty or pd.isna(stamp) or stamp.tzinfo is None or pd.isna(period):
                continue
            local = stamp.tz_convert("America/New_York").tz_localize(None).isoformat()
            accession = record["accession"]
            old = filings.loc[filings.accession.eq(accession)]
            if not old.empty:
                previous = old.iloc[0]
                if (previous.cik != cik or previous.form != record["form"]
                        or previous.filed_date != filed or previous.report_period != period
                        or pd.Timestamp(previous.accepted_local) != pd.Timestamp(local)):
                    raise ValueError(f"Reviewed filing conflicts with SEC cache: {accession}")
                continue
            rows.append(dict(ticker=interval.ticker.iloc[0], cik=cik, accession=accession,
                             form=record["form"], report_period=period, filed_date=filed,
                             accepted_local=local, timezone="America/New_York",
                             public_by=None, source=(
                                 f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                                 f"{accession.replace('-', '')}/{accession}-index.htm"),
                             metadata_source=record["source"],
                             metadata_sha256=record["sha256"]))
    return pd.concat([filings, pd.DataFrame(rows)], ignore_index=True), sorted(sources)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--identity-extensions", type=Path)
    parser.add_argument("--reviewed-facts", type=Path)
    parser.add_argument("--new-identities", type=Path)
    args = parser.parse_args()
    source_paths = [Path(__file__), BASE / "build.py"]
    frames = []
    for old, new, dates in [
        ("identities", "identity_additions", ["start", "end"]),
        ("filings", "filing_additions", ["filed_date", "report_period"]),
        ("facts", "fact_additions", ["start", "end"]),
    ]:
        paths = [BASE / f"{old}.csv", HERE / f"{new}.csv"]
        source_paths.extend(paths)
        frames.append(pd.concat([pd.read_csv(p, dtype={"cik": str}, parse_dates=dates)
                                 for p in paths], ignore_index=True))
    identities, filings, facts = frames
    if args.identity_extensions:
        updates = pd.read_csv(args.identity_extensions, dtype={"cik": str},
                              parse_dates=["start", "end"])
        check = updates.merge(identities, on=["ticker", "cik"], validate="one_to_one")
        if len(check) != len(updates) or (check.start_x > check.start_y).any() or (
                check.end_x != check.end_y).any():
            raise ValueError("Extensions must extend a reviewed interval for the same issuer")
        identities = pd.concat([identities.loc[~identities.ticker.isin(updates.ticker)],
                                updates], ignore_index=True)
        source_paths.append(args.identity_extensions)
    if args.new_identities:
        added = pd.read_csv(args.new_identities, dtype={"cik": str}, parse_dates=["start", "end"])
        inventory = pd.read_csv(BASE / "pilot_2/issuer_inventory.csv")
        if (added.ticker.duplicated().any() or added.ticker.isin(identities.ticker).any()
                or not added.ticker.isin(inventory.ticker).all()
                or not added.cik.str.fullmatch(r"\d{10}").fillna(False).all()
                or not added.identity_basis.str.startswith("https://www.sec.gov/Archives/").all()
                or not added.start.le(added.end).all()):
            raise ValueError("Expected distinct reviewed historical issuer intervals")
        identities = pd.concat([identities, added], ignore_index=True)
        source_paths.append(args.new_identities)
    if args.cache:
        filings, cache_paths = add_cached_filings(identities, filings, args.cache)
        source_paths += cache_paths + [ROOT / "experiments/sec_acquisition_v1/submissions.py"]
    if args.reviewed_facts:
        additional = pd.read_csv(args.reviewed_facts, dtype={"cik": str},
                                 parse_dates=["start", "end"])
        joined = additional.merge(filings[["cik", "accession"]], validate="many_to_one")
        if len(joined) != len(additional):
            raise ValueError("Additional facts must reference known issuer accessions")
        for row in additional.itertuples(index=False):
            prefix = (f"https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/"
                      f"{row.accession.replace('-', '')}/")
            if not row.source.startswith(prefix):
                raise ValueError(f"Fact source does not match accession: {row.accession}")
        facts = pd.concat([facts, additional], ignore_index=True)
        if facts.duplicated(["cik", "accession", "start", "end", "unit", "scope"]).any():
            raise ValueError("Duplicate fact context")
        source_paths.append(args.reviewed_facts)
    keys_path = BASE / "pilot_2/features.parquet"
    keys = pd.read_parquet(keys_path, columns=["ticker", "date"])
    sessions = mcal.get_calendar("NASDAQ").schedule("2019-01-01", "2023-12-31").index
    features, ledger = assemble(keys, identities, filings, facts, sessions, 2)

    sections = []
    label_path = ROOT / "data/labels.parquet"
    sector_path = ROOT / "src/graph/gics.csv"
    inventory_path = BASE / "pilot_2/issuer_inventory.csv"
    source_paths.extend([keys_path, label_path, sector_path, inventory_path])
    labels = pd.read_parquet(label_path, columns=["ticker", "date", "y_5d"])
    sectors = pd.read_csv(sector_path).set_index("ticker").gics_sector
    for i in range(2):
        path = ROOT / f"experiments/gat_edge_weight_v1/runs/w{i}_h5_s42_original.parquet"
        source_paths.append(path)
        archived = pd.read_parquet(path, columns=["ticker", "date", "y_true"])
        target = archived.merge(labels, how="left", validate="one_to_one")
        if not target.y_true.eq(target.y_5d).fillna(False).all():
            raise ValueError("Assessment labels differ")
        joined = archived[["ticker", "date"]].merge(features, validate="one_to_one")
        pd.testing.assert_frame_equal(joined[["ticker", "date"]], archived[["ticker", "date"]])
        sections.append(joined)
    assessment = pd.concat(sections, ignore_index=True)
    assessment["year"] = assessment.date.dt.year
    assessment["sector"] = assessment.ticker.map(sectors).fillna("unclassified")
    assessment["has_issuer"] = assessment.cik.notna()
    assessment["has_filing"] = assessment.accession.notna()
    assessment["form"] = assessment["form"].fillna("missing")
    assessment["complete_provenance"] = assessment[["cik", "accession", "source", "filed_date",
                                                    "accepted_local", "eligible_date"]].notna().all(axis=1)
    covered = assessment.has_filing
    if (assessment.loc[covered, "eligible_date"] > assessment.loc[covered, "date"]).any():
        raise ValueError("Future filing joined")
    inventory = pd.read_csv(inventory_path, dtype={"cik": str})
    fields = ["ticker", "min", "max", "size", "corporate_ledger_cik_lead", "historical_aliases",
              "share_class_group"]
    inventory = inventory[fields].merge(identities, how="left", validate="one_to_one")
    inventory["identity_status"] = inventory.cik.notna().map(
        {True: "reviewed_partial_interval", False: "unresolved"})
    if shutil.disk_usage(HERE).free < 2 * 1024**3 + 32 * 1024**2:
        raise RuntimeError("Insufficient space above 2 GiB reserve")
    args.output.mkdir(parents=True)
    features.to_parquet(args.output / "features.parquet", index=False)
    assessment.to_parquet(args.output / "assessment.parquet", index=False)
    ledger.to_csv(args.output / "accessions.csv", index=False)
    inventory.to_csv(args.output / "identity_inventory.csv", index=False)
    for dimension in [[], ["sector"], ["ticker"], ["form"], ["coverage_reason"]]:
        summaries = assessment.groupby(["year", *dimension]).agg(
            rows=("date", "size"), tickers=("ticker", "nunique"), issuer_rows=("has_issuer", "sum"),
            event_rows=("has_filing", "sum"), fundamental_rows=("has_financial_facts", "sum"),
            provenance_rows=("complete_provenance", "sum"))
        summaries.to_csv(args.output / f"coverage_{dimension[0] if dimension else 'year'}.csv")
    manifest = {"policy": "SEC_LAGGED_PROXY", "complete_sessions_lag": 2,
                "history_complete": False, "training_ready": False,
                "source_sha256": {str(p.resolve().relative_to(ROOT)): _sha256(p) for p in source_paths}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
