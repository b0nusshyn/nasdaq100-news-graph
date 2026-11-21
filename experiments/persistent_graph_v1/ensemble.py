"""Combine two previously fitted original-attention models with a predetermined rule."""
import json

import pandas as pd

from experiments.boosted_decision_v1.replay import metrics
from experiments.gat_edge_weight_v1.summarize import block_interval
from experiments.market_direction_v1.study import finite
from experiments.persistent_graph_v1.study import PRIOR, ROOT
from experiments.persistent_graph_v1.verify import ledger
from src.models.ladder_report import auc, within_day_auc
from src.nlp.rebuild_timed_news import file_hash


def main():
    out = ROOT / "ensemble"
    out.mkdir()
    source = ledger(PRIOR / "ledger.jsonl")
    records = []
    for w in (0, 1):
        for h in (1, 5):
            names = [f"w{w}_h{h}_s{s}_original" for s in (42, 43)]
            keys = ["date", "ticker", "y_true"]
            paths = [PRIOR / f"{name}.parquet" for name in names]
            a, b = map(pd.read_parquet, paths)
            pd.testing.assert_frame_equal(a[keys], b[keys])
            p = a[keys].copy()
            t, u = [source[k]["threshold"] for k in names]
            p["p_up"] = ((a.p_up - t) + (b.p_up - u)) / 2
            calls = p.p_up.ge(0)
            p["up"] = calls
            m = metrics(p.y_true, calls)
            m["within_day_auc"] = within_day_auc(p)
            daily = {str(d.date()): auc(g.y_true, g.p_up) for d, g in p.groupby("date")}
            ci = block_interval(daily, source[names[0]]["metrics"]["daily_auc"])
            name = f"w{w}_h{h}_ensemble"
            p.rename(columns={"p_up": "margin"}).to_parquet(out / f"{name}.parquet", index=False)
            checkpoints = [x.with_suffix(".pt") for x in paths]
            records.append(dict(id=name, metrics=m, paired_auc_vs_seed42=ci, sources=names,
                                thresholds=[t, u], weights=[.5, .5], ranges=source[names[0]]["dates"],
                                hashes={str(x): file_hash(x) for x in [*paths, *checkpoints]}))
            print(name, m["within_day_auc"], m["accuracy"], ci)
    (out / "results.json").write_text(json.dumps(finite(records), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
