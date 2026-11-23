"""Verify matched populations and produce development and regime tables."""
import json

import numpy as np
import pandas as pd

from experiments.gat_edge_weight_v1.summarize import block_interval
from experiments.offline_rank_v1.study import ROOT, compute


def main():
    out = ROOT / "runs"
    runs = [json.loads(s) for s in (out / "ledger.jsonl").read_text().splitlines()]
    r = {(x["year"], x["horizon"], x["arm"]): x for x in runs}
    assert len(runs) == len(r) == 16
    print("| Run | Fit AUC | Earlier AUC | Assessment AUC | Accuracy | Majority | MCC | Balanced | F1 | Up | "
          "AUC contrast [95% block interval] |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    keys = {}
    for (year, h, arm), x in r.items():
        data = pd.read_parquet(out / f'{x["id"]}.parquet')
        assert data.date.dt.year.max() <= 2025
        assert len(data) == x["rows"] and not data.duplicated(["date", "ticker"]).any()
        observed = data[["date", "ticker", "y"]]
        pd.testing.assert_frame_equal(observed, keys.setdefault((year, h), observed))
        m = x["assessment"]
        replay = compute(data, data.score.to_numpy(), x["cut"])
        assert replay["positive_calls"] == m["positive_calls"]
        np.testing.assert_allclose([replay["auc"], replay["accuracy"]], [m["auc"], m["accuracy"]])
        base = {"pairwise": "boosted", "pairwise_residual": "pairwise", "boosted": "logistic"}.get(arm)
        diff = block_interval(m["daily"], r[year, h, base]["assessment"]["daily"]) if base else "—"
        cols = [x["fit_auc"], x["calibration_auc"], *(m[k] for k in
                ("auc", "accuracy", "majority", "mcc", "balanced", "f1")), m["positive_calls"]/x["rows"]]
        values = " | ".join("—" if v is None else f"{v:.4f}" for v in cols)
        print(f'| {x["id"]} | {values} | {diff} |')
    print("\n## Trailing regimes\n\n| Run | Regime | Dates | AUC | Accuracy | Majority |"
          "\n|---|---|---:|---:|---:|---:|")
    for x in runs:
        for k, v in x["regimes"].items():
            print(f'| {x["id"]} | {k} | {v["dates"]} | {v["auc"]:.4f} | {v["accuracy"]:.4f} | {v["majority"]:.4f} |')


if __name__ == "__main__":
    main()
