"""Evaluate training-only edge persistence with an edge-count-matched control."""
import json
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.gat_lstm_tuning.feature_comparison import clean
from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold
from src.graph.build_edges import corr_edges
from src.models.ladder_report import auc
from src.models.panel import build_panel
from src.models.train import GraphSources, fold_report, train_fold, training_partitions
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent
PRIOR = ROOT.parent / "gat_edge_weight_v1/runs"
KEYS = ["src", "dst"]


def topology(full, first, second):
    keep = first.loc[first.weight.gt(0), KEYS].merge(
        second.loc[second.weight.gt(0), KEYS], on=KEYS, validate="one_to_one")
    persistent = full.merge(keep, on=KEYS, validate="one_to_one")
    if persistent.empty or len(persistent) % 2:
        raise ValueError("persistence requires nonempty symmetric edges")
    pairs = full[full.src.lt(full.dst)].sort_values(["weight", "src", "dst"], ascending=[False, True, True])
    pairs = pairs.head(len(persistent)//2)[KEYS]
    bidirectional = pd.concat([pairs, pairs.rename(columns={"src": "dst", "dst": "src"})], ignore_index=True)
    sparse = full.merge(bidirectional, on=KEYS, validate="one_to_one")
    assert len(sparse) == len(persistent)
    return persistent, sparse


def graph_set(prices, fit_fold):
    full = corr_edges(prices, [fit_fold])
    a, b = np.array_split(fit_fold.train, 2)
    half1, half2 = [corr_edges(prices, [Fold(fit_fold.index, d, fit_fold.test)], min_overlap=63) for d in (a, b)]
    persistent, sparse = topology(full, half1, half2)
    return {"original": full, "persistent": persistent, "sparse": sparse}, (half1, half2)


def overlap(left, right):
    a, b = (set(zip(p.src, p.dst, strict=True)) for p in (left, right))
    return {"intersection": len(a & b), "jaccard": len(a & b)/len(a | b), "retained_fraction": len(a & b)/len(a)}


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    assert prices.date.max().year == 2023
    panel = build_panel(prices, labels, empty, embeddings=False)
    refs = {r["id"]: r for r in map(json.loads, (PRIOR / "ledger.jsonl").read_text().splitlines())}
    out = ROOT / "runs"
    out.mkdir()
    paths = [Path(__file__), ROOT / "protocol.md", PRIOR / "ledger.jsonl", PRIOR / "manifest.json",
             *[Path("data") / f"{n}.parquet" for n in ("prices", "labels", "membership")],
             Path("src/models/train.py"), Path("src/models/gat_lstm.py"), Path("src/graph/build_edges.py")]
    (out / "manifest.json").write_text(json.dumps({str(p): file_hash(p) for p in paths}, indent=2))
    prepared = {}
    for w in (0, 1):
        ref = refs[f"w{w}_h5_s42_original"]
        ranges = ref["dates"]
        assess = panel.days[(panel.days >= ranges["assessment"][0]) & (panel.days <= ranges["assessment"][1])]
        start = panel.days.get_loc(assess[0])
        fold = Fold(w, panel.days[start-312:start-5], assess)
        fit, val = training_partitions(fold, ref["config"])
        for d, k in ((fit, "fit"), (val, "validation"), (assess, "assessment")):
            assert ranges[k] == [str(d[0].date()), str(d[-1].date()), len(d)]
        fit_fold = Fold(w, fit, assess)
        graphs, halves = graph_set(prices, fit_fold)
        pd.testing.assert_frame_equal(graphs["original"], pd.read_parquet(PRIOR / f"graph_{w}.parquet"))
        for k, g in graphs.items():
            g.to_parquet(out / f"graph_{w}_{k}.parquet", index=False)
        prepared[w] = (fold, fit, graphs, halves)

    plan = [(0, 5, 42, "persistent")]
    plan += [(w, h, seed, arm) for seed in (42, 43) for w in (0, 1) for h in (1, 5)
             for arm in (("persistent", "sparse") if seed == 42 else ("persistent",))
             if (w, h, seed, arm) != plan[0]]
    with (out / "ledger.jsonl").open("x") as ledger:
        for w, h, seed, arm in plan:
            if shutil.disk_usage(ROOT).free < 2.1 * 1024**3:
                raise OSError("disk safety margin")
            fold, fit, graph, _ = prepared[w]
            ref = refs[f"w{w}_h{h}_s{seed}_original"]
            cfg = deepcopy(ref["config"])
            cfg["model"] = f"persistent_v1_{arm}"
            name = f"w{w}_h{h}_s{seed}_{arm}"

            def select(p, folds, *, fit_dates=fit, selected=graph[arm], index=w):
                assert p is prices and len(folds) == 1 and folds[0].index == index and folds[0].train.equals(fit_dates)
                return selected

            print("start", name, flush=True)
            with patch("src.graph.build_edges.corr_edges", side_effect=select):
                preds, info, state = train_fold(panel, GraphSources(prices=prices), fold, h, cfg,
                                                return_model=True, log=lambda *_: None)
            controls = [ref["id"]] + ([f"w{w}_h{h}_s42_price"] if seed == 42 else [])
            for key in controls:
                original = pd.read_parquet(PRIOR / f"{key}.parquet")
                cols = ["date", "ticker", "y_true"]
                pd.testing.assert_frame_equal(preds[cols], original[cols])
            m = fold_report(preds, info)
            m["majority"] = max(m["base_rate"], 1-m["base_rate"])
            m["balanced"] = ((preds.p_up.ge(info["threshold"]) & preds.y_true.eq(1)).sum()/preds.y_true.sum()
                             + (preds.p_up.lt(info["threshold"]) & preds.y_true.eq(0)).sum()/preds.y_true.eq(0).sum())/2
            m["daily_auc"] = {str(d.date()): auc(g.y_true, g.p_up) for d, g in preds.groupby("date")}
            preds.to_parquet(out / f"{name}.parquet", index=False, compression="zstd")
            torch.save(state.model.state_dict(), out / f"{name}.pt")
            row = dict(id=name, seed=seed, arm=arm, config=cfg, ranges=ref["dates"], controls=controls,
                       rows=len(preds), metrics=m, **info)
            print(json.dumps(clean(row), allow_nan=False), file=ledger, flush=True)
            print("done", name, m["within_day_auc"], flush=True)
    diagnostics = {}
    for w, (fold, _, graphs, halves) in prepared.items():
        later = corr_edges(prices, [Fold(w, fold.test, fold.test)], min_overlap=63)
        diagnostics[w] = {"fit_halves": overlap(*halves), "assessment_only_edges": len(later),
                          "systems": {k: {"edges": len(g), "nodes": int(g.src.nunique()),
                                          "assessment_overlap": overlap(g, later)} for k, g in graphs.items()}}
    (out / "graph_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
