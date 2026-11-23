"""Verify hashes and replay saved states, then compare matched assessments."""
import json
from pathlib import Path

import pandas as pd
import torch

from experiments.boosted_decision_v1.replay import metrics
from experiments.gat_edge_weight_v1.summarize import block_interval
from experiments.gat_regularization_v1.study import PREVIOUS, ROOT
from experiments.historical_demo_v1.replay import compare, infer
from src.eval.run_baselines import load_inputs
from src.models.panel import build_panel, fit_features
from src.models.train import build_model, fold_edges
from src.nlp.rebuild_timed_news import file_hash


def read(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def verify(root, expected):
    torch.set_num_threads(2)
    out = root / "runs"
    current = read(out / "ledger.jsonl")
    assert len(current) == expected
    prior = read(PREVIOUS / "ledger.jsonl")
    for p, h in json.loads((out / "manifest.json").read_text()).items():
        assert file_hash(Path(p)) == h
    prices, labels, news = load_inputs(include_news=False, legacy_labels=True)
    panel = build_panel(prices, labels, news, embeddings=False)
    print("| Run | Fit accuracy / AUC | Val accuracy / AUC | Assessment accuracy | Majority | MCC | AUC | "
          "Balanced | F1 | Share Up | Paired AUC vs original [95%] |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, r in current.items():
        for p, h in r["files"].items():
            assert file_hash(Path(p)) == h
        cfg = r["config"]
        edge = pd.read_parquet(PREVIOUS / f'graph_{r["fold"]}.parquet')
        g = fold_edges(edge, panel, r["fold"], ("corr",))
        start, end, _ = r["ranges"]["fit"]
        dates = panel.days[(panel.days >= start) & (panel.days <= end)]
        f = fit_features(panel, dates, window=cfg["window_days"], price=tuple(cfg["features"]["price"]),
                         news=(), embeddings=False)
        model = build_model(len(f.names), ("corr",), cfg)
        model.load_state_dict(torch.load(out / f"{name}.pt", map_location="cpu", weights_only=True))
        saved = pd.read_parquet(out / f"{name}.parquet")
        p = infer(model, panel, f, g, panel.day_index(saved.date.unique()), r["horizon"], cfg["train"]["batch_days"])
        assert compare(p, saved) <= 1e-7
        m = r["assessment"]
        b = metrics(saved.y_true, saved.p_up.ge(r["threshold"]))["balanced"]
        fit, val = (r["fit_eval"][k] for k in ("fit", "validation"))
        contrast = block_interval(m["daily_auc"], prior[r["source_control"]]["metrics"]["daily_auc"])
        print(f'| {name} | {fit["accuracy"]:.4f} / {fit["within_day_auc"]:.4f} | '
              f'{val["accuracy"]:.4f} / {val["within_day_auc"]:.4f} | {m["accuracy"]:.4f} | {m["majority"]:.4f} | '
              f'{m["mcc"]:.4f} | {m["within_day_auc"]:.4f} | {b:.4f} | {m["f1"]:.4f} | '
              f'{m["share_predicted_up"]:.4f} | {contrast} |')
    print("\n| Matched control | Accuracy | Majority | MCC | AUC |\n|---|---:|---:|---:|---:|")
    for name, r in prior.items():
        if r["arm"] in ("price", "original"):
            m = r["metrics"]
            print(f'| {name} | {m["accuracy"]:.4f} | {m["majority"]:.4f} | '
                  f'{m["mcc"]:.4f} | {m["within_day_auc"]:.4f} |')


if __name__ == "__main__":
    verify(ROOT, 16)
