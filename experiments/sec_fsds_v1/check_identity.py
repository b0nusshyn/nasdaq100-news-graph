"""Offline: compare FSDS prefix-based ticker->CIK mapping with the hand-reviewed intervals.

    .venv/bin/python -B -m experiments.sec_fsds_v1.check_identity experiments/sec_fsds_v1/store_v1
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REVIEWED = ["experiments/sec_lagged_proxy/identities.csv",
            "experiments/sec_acquisition_v1/batch_01/identities.csv",
            "experiments/sec_acquisition_v1/batch_02/new_identities.csv"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("store", type=Path)
    args = p.parse_args()
    store = pd.read_parquet(args.store / "store.parquet", columns=["ticker", "date", "cik"])
    store["ticker"] = store.ticker.astype(str)
    rev = pd.concat([pd.read_csv(ROOT / f, dtype={"cik": str}) for f in REVIEWED], ignore_index=True)
    rev[["start", "end"]] = rev[["start", "end"]].apply(pd.to_datetime)
    rows = []
    for r in rev.itertuples(index=False):
        s = store.loc[store.ticker.eq(r.ticker) & store.date.between(r.start, r.end)]
        rows.append({"ticker": r.ticker, "reviewed_cik": r.cik, "rows": len(s),
                     "agree": int(s.cik.eq(r.cik).sum()), "unmapped": int(s.cik.isna().sum()),
                     "disagree": int((s.cik.notna() & s.cik.ne(r.cik)).sum())})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(f"total agree {out.agree.sum()} unmapped {out.unmapped.sum()} disagree {out.disagree.sum()}")
    if out.disagree.sum():
        raise SystemExit("Mapping disagrees with reviewed identities")


if __name__ == "__main__":
    main()
