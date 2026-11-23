"""Replay both ridge components and compare saved baseline runs without fitting."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.cohort_outperformance_v1.study import dataset
from experiments.gat_edge_weight_v1.summarize import block_interval
from experiments.market_direction_v1.study import features
from experiments.submission_readiness_v1.decompose import BASE, ROOT, scores


def main():
    out = ROOT / "runs"
    cfg = json.loads((out / "manifest.json").read_text())
    records = [json.loads(x) for x in (out / "ledger.jsonl").read_text().splitlines()]
    baselines = {r["id"]: r for r in map(json.loads, (BASE / "ledger.jsonl").read_text().splitlines())}
    assert len(records) == 4
    x, _ = features(dataset()[0])
    print("| Window | Arm | Accuracy | Majority | MCC | AUC | Δ AUC vs logistic [95%] |")
    print("|---|---|---:|---:|---:|---:|---|")
    with threadpool_limits(limits=2):
        for r in records:
            run = r["id"]
            f = pd.read_parquet(out / f"{run}.parquet")
            panel = f[["date", "ticker"]].merge(x, on=["date", "ticker"], validate="one_to_one")
            models = joblib.load(out / f"{run}.joblib")
            raw = scores(panel, models["stock"], models["market"], cfg["stock"], cfg["market"])
            for key, s in zip(("stock", "combined"), raw, strict=True):
                np.testing.assert_allclose(f[key], s, atol=1e-12)
                np.testing.assert_array_equal(f[f"{key}_up"], s >= r["results"][key]["cutoff"])
            for k in ("stock", "combined", "logistic", "boosted", "pairwise"):
                m = r["results"][k]["metrics"] if k in r["results"] else baselines[f"{run}_{k}"]["assessment"]
                interval = block_interval(m["daily"], baselines[f"{run}_logistic"]["assessment"]["daily"])
                print(f'| {run} | {k} | {m["accuracy"]:.4f} | {m["majority"]:.4f} | {m["mcc"]:.4f} | '
                      f'{m["auc"]:.4f} | {interval} |')


if __name__ == "__main__":
    main()
