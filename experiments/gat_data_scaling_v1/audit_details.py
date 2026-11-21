"""Recompute exceptional membership keys, use of history, and yearly label summaries."""
import json

import pandas as pd

from experiments.gat_data_scaling_v1.study import OLD, ROOT, data, expand, parts, records
from src.core.io import read_table
from src.data.build_index_history import trading_days


def main():
    prices, labels, panel = data()
    members = prices.loc[prices.in_index & prices.date.ge("2019-01-01"), ["date", "ticker"]]
    seen = set(members.itertuples(index=False, name=None))
    schedule = trading_days("2019-01-01", "2023-12-29")
    assert set(schedule) == set(prices.loc[prices.date.ge("2019-01-01"), "date"])
    missing = set()
    for r in read_table("membership").itertuples(index=False):
        dates = schedule[(schedule >= r.start) & ((schedule <= r.end) if pd.notna(r.end) else True)]
        missing.update((d.date().isoformat(), r.ticker) for d in dates if (d, r.ticker) not in seen)
    fit_rows = {}
    prior = records(OLD / "ledger.jsonl")
    for w in (0, 1):
        r = prior[f"w{w}_h5_s42_original"]
        _, fit, val = parts(panel, r)
        fold, full, _, _, _ = expand(panel, r, 1.)
        fit_rows[str(w)] = {"prior_fit_days": len(fit), "prior_fit_rows": len(panel.rows(fit, 5)),
                            "full_fit_rows": len(panel.rows(full, 5)),
                            "unused_fit_rows": len(panel.rows(full.difference(fit), 5)),
                            "validation_rows": len(panel.rows(val, 5)),
                            "assessment_rows": len(panel.rows(fold.test, 5))}
    z = members.merge(labels, on=["date", "ticker"], validate="one_to_one")
    z = z.dropna(subset=["y_5d"]).assign(year=lambda f: f.date.dt.year)
    annual = z.groupby("year").agg(rows=("y_5d", "size"), positive_rate=("y_5d", "mean"),
                                  mean_absolute_return=("fwd_ret_5d", lambda s: s.abs().mean()),
                                  sessions=("date", "nunique"))
    root = ROOT / "runs"
    assert [list(x) for x in sorted(missing)] == json.loads((root / "missing_member_keys.json").read_text())
    assert fit_rows == json.loads((root / "utilization.json").read_text())
    archived = pd.read_csv(root / "annual_population.csv", index_col="year")
    pd.testing.assert_frame_equal(annual, archived, check_dtype=False, check_index_type=False)
    print("Historical utilization and annual label audit reproduced.")


if __name__ == "__main__":
    main()
