"""Recompute full-population metrics for saved original-direction predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.boosted_decision_v1.replay import metrics
from experiments.market_direction_v1.study import finite
from src.models.ladder_report import within_day_auc
from src.models.train import build_model
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent
EX = ROOT.parent
OFF = Path("data/predictions")
CH = Path("data/attention/checkpoints")


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def entry(p, y, call, score, path, model, h, period, seed, config, checkpoint=None):
    q = p[["date", "ticker"]].copy()
    q["y_true"], q["p_up"] = y.to_numpy(), score.to_numpy()
    assert not q.duplicated(["date", "ticker"]).any() and np.isfinite(q.p_up).all()
    assert set(q.y_true.unique()).issubset({0, 1}) and q.date.dt.year.max() < 2026
    return dict(model=model, horizon=int(h), period=period, seed=seed, rows=len(q),
                dates=[str(q.date.min().date()), str(q.date.max().date()), q.date.nunique()],
                metrics=metrics(q.y_true, pd.Series(call.to_numpy(dtype=bool), index=q.index)) |
                {"within_day_auc": within_day_auc(q)},
                prediction=str(path), prediction_sha256=file_hash(path), configuration=config,
                checkpoint=str(checkpoint) if checkpoint and checkpoint.is_file() else None,
                checkpoint_sha256=file_hash(checkpoint) if checkpoint and checkpoint.is_file() else None)


def official():
    rec = read(OFF / "gat_lstm_run.json")
    saved = read(CH / "checkpoints.json")
    assert rec["config"] == saved["config"]
    q = pd.read_parquet(OFF / "gat_lstm.parquet")
    archived = pd.read_parquet("data/attention/gat_lstm_contribution.parquet")
    keys = ["horizon", "fold", "date", "ticker"]
    p = q.merge(archived[keys + ["p_up", "y_true"]], on=keys, validate="one_to_one", suffixes=("", "_att"))
    assert len(p) == len(q) == len(archived)
    np.testing.assert_array_equal(p.y_true, p.y_true_att)
    diff = float(abs(p.p_up - p.p_up_att).max())
    assert diff < 1e-5
    folds = []
    for meta in saved["folds"]:
        h, fold = meta["horizon"], meta["fold"]
        r = next(x for x in rec["folds"] if (x["horizon"], x["fold"]) == (h, fold))
        assert meta["best_step"] == r["best_step"] and meta["threshold"] == r["threshold"]
        ckpt = CH / meta["checkpoint"]
        model = build_model(len(meta["features"]), tuple(saved["edge_types"]), rec["config"])
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        f = q[q.horizon.eq(h) & q.fold.eq(fold)]
        row = entry(f, f.y_true, f.p_up.ge(r["threshold"]), f.p_up, OFF / "gat_lstm.parquet", "gat_lstm_T1",
                    h, f"official_fold_{fold}", 42, rec["config"], ckpt)
        for a, b in (("accuracy", "accuracy"), ("within_day_auc", "within_day_auc"), ("mcc", "mcc")):
            np.testing.assert_allclose(row["metrics"][a], r["test"][b], atol=1e-7, rtol=0)
        folds.append(row | {"threshold": r["threshold"], "checkpoint_step": meta["best_step"]})
    averages = {str(h): {k: float(np.mean([f["metrics"][k] for f in folds if f["horizon"] == h]))
                         for k in ("accuracy", "majority", "mcc", "within_day_auc")}
                for h in (1, 5)}
    return dict(full_fold_means=averages, folds=folds, archived_attention_max_abs_difference=diff,
                checkpoint_manifest=str(CH / "checkpoints.json"), official_run=str(OFF / "gat_lstm_run.json"))


def development():
    rows = []
    for name in ("offline_rank_v1", "submission_readiness_v1", "gat_edge_weight_v1"):
        folder = EX / name / "runs"
        for r in lines(folder / "ledger.jsonl"):
            path = folder / f'{r["id"]}.parquet'
            f = pd.read_parquet(path)
            ref = {"study": name, "record": r["id"], "ledger": str(folder / "ledger.jsonl")}
            if name == "gat_edge_weight_v1":
                rows.append(entry(f, f.y_true, f.p_up.ge(r["threshold"]), f.p_up, path, r["arm"], r["horizon"],
                                  r["fold"], r["seed"], ref | {"config": r["config"]}, path.with_suffix(".pt")))
            elif name == "offline_rank_v1":
                rows.append(entry(f, f.y, f.up, f.score, path, r["arm"], r["horizon"], r["year"], 42,
                                  ref | {"manifest": str(folder / "resume_manifest.json")},
                                  path.with_suffix(".joblib")))
            else:
                for k in ("stock", "combined"):
                    rows.append(entry(f, f.y, f[f"{k}_up"], f[k], path, f"ridge_{k}", r["horizon"], r["year"], None,
                                      ref | {"manifest": str(folder / "manifest.json")}, path.with_suffix(".joblib")))
    p = EX / "post2023_validation_v1"
    for r in lines(p / "development_runs.jsonl")[1:]:
        path = p / "development_predictions" / f'{r["year"]}_{r["horizon"]}_{r["arm"]}.parquet'
        f = pd.read_parquet(path)
        rows.append(entry(f, f.y_true, f.pred_up, f.p_up, path, r["arm"], r["horizon"], r["year"], 42,
                          {"study": p.name, "ledger": str(p / "development_runs.jsonl")}))
    m = EX / "market_direction_v1"
    for n, d in (("runs.jsonl", "predictions"), ("followup_runs.jsonl", "followup_predictions")):
        for r in lines(m / n)[1:]:
            path = m / d / f'w{r["window"]}_h{r["horizon"]}_{r["model"]}.parquet'
            f = pd.read_parquet(path)
            rows.append(entry(f, f.y_true, f.pred_up, f.raw_score, path, r["model"], r["horizon"], r["window"], 42,
                              {"study": m.name, "ledger": str(m / n)}))
    news = EX / "fnspid_candidate_audit"
    for n in sorted(news.glob("cycle_*.jsonl")):
        header, *fits = lines(n)
        for r in fits:
            path = Path(r["predictions"])
            f = pd.read_parquet(path)
            rows.append(entry(f, f.y_true, f.p_up.ge(r["threshold"]), f.p_up, path, r["arm"], r["horizon"], r["fold"],
                              header.get("seed", header.get("config", {}).get("seed")),
                              {"study": news.name, "ledger": str(n)}))
    return rows


def main():
    rows = development()
    published = official()
    # Individual assessed-window maxima are retrospective and never pooled across targets.
    opt = {}
    for h in (1, 5):
        eligible = [r for r in rows if r["horizon"] == h]
        opt[str(h)] = {k: max(eligible, key=lambda r: r["metrics"][k]) for k in ("accuracy", "within_day_auc")}
        opt[str(h)]["accuracy_lift"] = max(eligible, key=lambda r: r["metrics"]["accuracy"]-r["metrics"]["majority"])
    stable = {}
    for h in (1, 5):
        eligible = [r for r in rows if r["horizon"] == h and isinstance(r["period"], int) and r["period"] >= 2024]
        groups = {}
        for r in eligible:
            k = (r["configuration"]["study"], r["model"])
            groups.setdefault(k, []).append(r)
        complete = [g for g in groups.values() if {r["period"] for r in g} == {2024, 2025}]
        best = max(complete, key=lambda g: min(r["metrics"]["within_day_auc"] for r in g))
        stable[str(h)] = {"criterion": "maximize worst-window raw/saved score within-day AUC on 2024–2025",
                          "qualifies_as_validated": False, "runs": best}
    original_graph = [r for r in rows if r["configuration"]["study"] == "gat_edge_weight_v1"
                      and r["model"] == "original"]
    data = dict(official_gat=published, development_maxima=opt, cross_period_candidates=stable,
                rows=rows, experimental_gat_best_auc=max(
                    original_graph,
                    key=lambda r: r["metrics"]["within_day_auc"]))
    (ROOT / "best_model_registry.json").write_text(json.dumps(finite(data), indent=2, allow_nan=False)+"\n")
    print(len(rows), published["full_fold_means"])


if __name__ == "__main__":
    main()
