"""Replay fitted checkpoint outputs and compare matched assessment metrics."""
import json

import pandas as pd
import torch

from experiments.boosted_decision_v1.replay import metrics
from experiments.gat_edge_weight_v1.summarize import block_interval
from experiments.historical_demo_v1.replay import compare, infer
from experiments.persistent_graph_v1.study import PRIOR, ROOT
from src.eval.run_baselines import load_inputs
from src.models.panel import build_panel, fit_features
from src.models.train import build_model, fold_edges
from src.nlp.rebuild_timed_news import file_hash


def ledger(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def main():
    torch.set_num_threads(2)
    out = ROOT / "runs"
    rows = ledger(out / "ledger.jsonl")
    assert len(rows) == 12
    prior = ledger(PRIOR / "ledger.jsonl")
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    panel = build_panel(prices, labels, empty, embeddings=False)
    print("| Run | Train probe AUC | Validation AUC | AUC | Accuracy | Majority | MCC | Balanced | F1 | % Up |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    used = {k for r in rows.values() for k in r["controls"]}
    for name, r in (prior | rows).items():
        if name not in rows and name not in used:
            continue
        is_new = name in rows
        here = out if is_new else PRIOR
        path = here / f"{name}.parquet"
        saved = pd.read_parquet(path)
        m = r["metrics"]
        config = r["config"]
        chosen = next(x for x in r["history"] if x["step"] == r["best_step"])
        if is_new:
            start, end, _ = r["ranges"]["fit"]
            f = fit_features(panel, panel.days[(panel.days >= start) & (panel.days <= end)],
                             window=config["window_days"], price=tuple(config["features"]["price"]),
                             news=(), embeddings=False)
            edges = pd.read_parquet(out / f'graph_{r["fold"]}_{r["arm"]}.parquet')
            g = fold_edges(edges, panel, r["fold"], ("corr",))
            model = build_model(len(f.names), ("corr",), config)
            model.load_state_dict(torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True))
            fresh = infer(model, panel, f, g, panel.day_index(saved.date.unique()), r["horizon"],
                          config["train"]["batch_days"])
            assert compare(fresh, saved) <= 1e-7
        v = metrics(saved.y_true, saved.p_up.ge(r["threshold"]))
        values = (chosen["train_probe_within_day_auc"], r["best_val_within_day_auc"], m["within_day_auc"],
                  m["accuracy"], m["majority"], m["mcc"], v["balanced"], m["f1"], m["share_predicted_up"])
        print(f'| {name} | {" | ".join(f"{x:.4f}" if x is not None else "—" for x in values)} |')
    print("\n| Candidate | Comparator | Paired AUC difference [95% date-block interval] |\n|---|---|---|")
    for name, r in rows.items():
        comparator = [*r["controls"]]
        if r["seed"] == 42 and r["arm"] == "persistent":
            comparator.append(name.replace("persistent", "sparse"))
        for k in comparator:
            diff = block_interval(r["metrics"]["daily_auc"], (prior | rows)[k]["metrics"]["daily_auc"])
            print(f"| {name} | {k} | {diff} |")
    captures = {}
    for name in rows:
        p = out / f"{name}.parquet"
        captures[name] = {"predictions": str(p), "prediction_sha256": file_hash(p),
                          "checkpoint": str(p.with_suffix(".pt")),
                          "checkpoint_sha256": file_hash(p.with_suffix(".pt"))}
    (ROOT / "artifact_hashes.json").write_text(json.dumps(captures, indent=2))


if __name__ == "__main__":
    main()
