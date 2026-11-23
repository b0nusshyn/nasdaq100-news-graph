"""Verify source/artifact hashes and replay scaling checkpoints on identical keys."""
import json
from pathlib import Path

import pandas as pd
import torch

from experiments.gat_data_scaling_v1.study import OLD, ROOT, data, expand, records, score
from experiments.gat_edge_weight_v1.summarize import block_interval
from experiments.historical_demo_v1.replay import compare
from src.models.train import Fitted, GraphSources, build_model, fold_inputs
from src.nlp.rebuild_timed_news import file_hash


def main(*, decay=False):
    torch.set_num_threads(2)
    prices, _, panel = data()
    prior = records(OLD / "ledger.jsonl")
    out = ROOT / ("recency_runs" if decay else "runs")
    full = records(ROOT / "runs/ledger.jsonl") if decay else {}
    all_records = records(out / "ledger.jsonl")
    for p, h in json.loads((out / "manifest.json").read_text()).items():
        assert file_hash(Path(p)) == h
    print("| Run | Fit from–to | Sessions | Labelled rows | Edges | Optimizer steps | Fit AUC | Val AUC | "
          f'Assessment AUC | Fit–assessment gap | Accuracy | Majority | MCC | Balanced | '
          f'Δ AUC vs {"full unweighted" if decay else "252"} [95%] |')
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for k, r in all_records.items():
        if "skipped" in r:
            continue
        for p, h in r["files"].items():
            assert file_hash(Path(p)) == h
        w, s = r["fold"], r["seed"]
        base = prior[f"w{w}_h5_s{s}_original"]
        fold, _, _, _, _ = expand(panel, base, r["fraction"])
        cfg = r["config"]
        types, feat, graph = fold_inputs(panel, GraphSources(prices=prices), fold, cfg)
        assert len(graph["src"]) == r["edges_per_day"]
        model = build_model(len(feat.names), types, cfg)
        model.load_state_dict(torch.load(out / f"{k}.pt", map_location="cpu", weights_only=True))
        state = Fitted(model, feat, graph, panel.day_index(fold.test), 5, cfg["train"]["batch_days"])
        fresh, m = score(panel, state, fold.test, 5, r)
        archived = pd.read_parquet(out / f"{k}.parquet")
        assert compare(fresh, archived) <= 1e-7
        start, end, n = r["ranges"]["fit"]
        train_auc, val_auc = [r[key]["within_day_auc"] for key in ("train", "validation_metrics")]
        a = m["within_day_auc"]
        ref = full[f"w{w}_p100_s{s}"]["assessment"] if decay else base["metrics"]
        ci = block_interval(m["daily_auc"], ref["daily_auc"])
        print(f'| {k} | {start}–{end} | {n} | {r["train_rows"]} | {r["edges_per_day"]} | {r["steps"]} | '
              f'{train_auc:.4f} | {val_auc:.4f} | {a:.4f} | {train_auc-a:+.4f} | {m["accuracy"]:.4f} | '
              f'{m["majority"]:.4f} | {m["mcc"]:.4f} | {m["balanced_accuracy"]:.4f} | {ci} |')
    print("\n## Archived 252-session controls\n")
    print("| Run | Accuracy | Majority | MCC | AUC |\n|---|---:|---:|---:|---:|")
    for w in (0, 1):
        for s in (42, 43):
            p = prior[f"w{w}_h5_s{s}_original"]
            m = p["metrics"]
            print(f'| {p["id"]} | {m["accuracy"]:.4f} | {m["majority"]:.4f} | '
                  f'{m["mcc"]:.4f} | {m["within_day_auc"]:.4f} |')
    if not decay:
        print("\n## Validation-only window selection diagnostic\n")
        print("| Window | Seed | Selected fraction | Earlier AUC | Assessment AUC |")
        print("|---|---:|---:|---:|---:|")
        for w in (0, 1):
            for s in (42, 43):
                eligible = [r for r in all_records.values() if r.get("fold") == w and r.get("seed") == s]
                r = max(eligible, key=lambda x: (x["best_val_within_day_auc"], -x["fraction"]))
                print(f'| {2022+w} | {s} | {r["fraction"]:.0%} | {r["best_val_within_day_auc"]:.4f} | '
                      f'{r["assessment"]["within_day_auc"]:.4f} |')


if __name__ == "__main__":
    main()
