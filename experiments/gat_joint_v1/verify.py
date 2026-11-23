"""Replay outer predictions and verify inner-only lambda choices."""
import json
from pathlib import Path

import pandas as pd
import torch

from experiments.gat_edge_weight_v1.summarize import block_interval
from experiments.gat_joint_v1.study import OLD, ROOT, SMALL, data, parts, records, score
from experiments.market_direction_v1.study import finite
from src.models.train import Fitted, GraphSources, build_model, fold_inputs
from src.nlp.rebuild_timed_news import file_hash


def cell(r):
    return r["fold"], r["horizon"], r["seed"]


def main():
    torch.set_num_threads(2)
    prices, _, panel = data()
    prior = records(OLD / "ledger.jsonl")
    small = records(SMALL / "ledger.jsonl")
    candidates = records(ROOT / "runs/candidates.jsonl")
    choices = records(ROOT / "runs/selection.jsonl")
    assert len(candidates) == 16 and len(choices) == 8
    for r in candidates.values():
        for p, h in r["files"].items():
            assert file_hash(Path(p)) == h
        assert r["lambda_rank"] in (.10, .30)
    for r in choices.values():
        a = candidates[r["id"]]
        same = [b for b in candidates.values() if cell(a) == cell(b)]
        assert len(same) == 2 and max(same, key=lambda b: (b["best_val_within_day_auc"], -b["lambda_rank"])) == a

    print("| Run | Fit accuracy / AUC | Val accuracy / AUC | Accuracy | Majority | MCC | AUC | Balanced | "
          "Δ AUC vs 64-unit [95%] | Δ AUC vs 32-unit [95%] |")
    print("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    all_runs = []
    for root in (ROOT, ROOT.parent / "gat_margin_weight_v1"):
        out = root / "runs"
        alternatives = {"trainer.py": out / "executed_trainer.py.snapshot",
                        "study.py": out / "executed_study.py.snapshot"}
        for p, h in json.loads((out / "manifest.json").read_text()).items():
            f = Path(p)
            if file_hash(f) != h:
                assert file_hash(alternatives[f.name]) == h
        fits = records(out / "selected.jsonl")
        assert len(fits) == 8
        for r in fits.values():
            if root == ROOT:
                assert r["id"] in choices
            for p, h in r["files"].items():
                assert file_hash(Path(p)) == h
            w, h, s = cell(r)
            ref, low = prior[f"w{w}_h{h}_s{s}_original"], small[f"w{w}_h{h}_s{s}_small"]
            fold, _, _ = parts(panel, ref)
            cfg = r["config"]
            types, features, graph = fold_inputs(panel, GraphSources(prices=prices), fold, cfg)
            model = build_model(len(features.names), types, cfg)
            ck = out / f'{r["id"]}.pt'
            model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
            state = Fitted(model, features, graph, panel.day_index(fold.test), h, cfg["train"]["batch_days"])
            p, m = score(panel, state, fold.test, h, r)
            assert file_hash(Path(r["prediction"])) == r["prediction_sha256"]
            saved = pd.read_parquet(r["prediction"])
            pd.testing.assert_frame_equal(p[["date", "ticker", "y_true"]], saved[["date", "ticker", "y_true"]])
            assert p.p_up.sub(saved.p_up).abs().max() <= 1e-7
            a = block_interval(m["daily_auc"], ref["metrics"]["daily_auc"])
            b = block_interval(m["daily_auc"], low["assessment"]["daily_auc"])
            f, v = r["fit"], r["validation_metrics"]
            print(f'| {r["id"]} | {f["accuracy"]:.4f} / {f["within_day_auc"]:.4f} | '
                  f'{v["accuracy"]:.4f} / {v["within_day_auc"]:.4f} | {m["accuracy"]:.4f} | '
                  f'{m["majority"]:.4f} | {m["mcc"]:.4f} | {m["within_day_auc"]:.4f} | '
                  f'{m["balanced_accuracy"]:.4f} | {a} | {b} |')
            all_runs.append(dict(id=r["id"], checkpoint=str(ck), files=r["files"], prediction=r["prediction"],
                                 prediction_sha256=r["prediction_sha256"], ranges=r["ranges"], horizon=h, seed=s,
                                 config=cfg, metrics=m, lambda_rank=r.get("lambda_rank")))
    prior_registry = ROOT.parent / "gat_regularization_v1/best_model_registry.json"
    leads = json.loads(prior_registry.read_text())["leaders"]
    criteria = {"accuracy": lambda m: m["accuracy"], "within_day_auc": lambda m: m["within_day_auc"],
                "mcc": lambda m: m["mcc"] if m["mcc"] is not None else -1,
                "accuracy_lift": lambda m: m["accuracy"] - m["majority"]}
    for h, levels in leads.items():
        for key, fn in criteria.items():
            levels[key] = max([levels[key], *[r for r in all_runs if r["horizon"] == int(h)]],
                              key=lambda r: fn(r["metrics"]))
    registry = dict(prior=str(prior_registry), leaders=leads, prospective_promotion=None,
                    fallback=str(ROOT.parent/"historical_demo_v1/development_replay.json"), new_outer=all_runs)
    (ROOT / "best_model_registry.json").write_text(json.dumps(finite(registry), indent=2, allow_nan=False))
    print("\n## Earlier validation candidates\n")
    print("| Fit | Step | Fit accuracy | Fit AUC | Validation accuracy | Validation AUC | Selected |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for k, r in candidates.items():
        f, v = r["fit"], r["validation_metrics"]
        print(f'| {k} | {r["best_step"]} | {f["accuracy"]:.4f} | {f["within_day_auc"]:.4f} | '
              f'{v["accuracy"]:.4f} | {v["within_day_auc"]:.4f} | {"yes" if k in choices else "no"} |')
    print("\n## BCE assessment controls\n")
    print("| Fit | Accuracy | Majority | MCC | AUC |\n|---|---:|---:|---:|---:|")
    for r in list(prior.values()) + list(small.values()):
        if r["arm"] not in ("original", "small"):
            continue
        m = r["metrics"] if "metrics" in r else r["assessment"]
        print(f'| {r["id"]} | {m["accuracy"]:.4f} | {m["majority"]:.4f} | {m["mcc"]:.4f} | '
              f'{m["within_day_auc"]:.4f} |')


if __name__ == "__main__":
    main()
