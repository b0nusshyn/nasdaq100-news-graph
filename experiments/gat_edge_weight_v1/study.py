"""Matched historical development study of signed graph messages and daily centering."""
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from experiments.gat_lstm_tuning.feature_comparison import clean
from experiments.gat_lstm_tuning.run import digest
from src.core.config import load_config
from src.core.paths import table_path
from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold
from src.graph.build_edges import corr_edges
from src.models.ladder_report import auc
from src.models.panel import build_panel
from src.models.train import GraphSources, fold_report, train_fold, training_partitions

ROOT = Path(__file__).parent
ARMS = ("price", "original", "signed", "centered")
RANGES = (("2022-07-18", "2022-12-30"), ("2023-07-24", "2023-12-20"))


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    assert prices.date.max().year == 2023 and labels.date.max().year == 2023
    panel = build_panel(prices, labels, empty, embeddings=False)
    cfg_path = Path("configs/gat_lstm.yaml")
    base = load_config(cfg_path)
    base["embargo_days"] = 5
    base["features"].update(news=[], embeddings=False)
    base["train"].update(epochs=12, validate_every_steps=None, validation_fraction=50 / 307,
                         preprocessing_scope="weight_fit", record_history=True)
    sources = GraphSources(prices=prices)
    output = ROOT / "runs"
    output.mkdir()
    paths = [table_path(name) for name in ("prices", "labels", "membership")]
    paths += [cfg_path, Path(__file__), *sorted(Path("src/models").glob("*.py")), Path("src/graph/build_edges.py")]
    manifest = {"config": base, "arms": ARMS, "ranges": RANGES, "torch": str(torch.__version__),
                "threads": 2, "max_fits": 24, "source_hashes": {str(p): digest(p) for p in paths}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    folds = []
    for w, (first, last) in enumerate(RANGES):
        days = panel.days[(panel.days >= first) & (panel.days <= last)]
        start = panel.days.get_loc(days[0])
        fold = Fold(w, panel.days[start - 312:start - 5], days)
        fit, val = training_partitions(fold, base)
        assert (len(fit), len(val)) == (252, 50)
        assert panel.days.get_loc(val[0]) - panel.days.get_loc(fit[-1]) == 6
        assert panel.days.get_loc(days[0]) - panel.days.get_loc(val[-1]) == 6
        assert panel.days.get_loc(days[-1]) + 5 < len(panel.days)
        g = corr_edges(prices, [Fold(w, fit, days)])
        g.to_parquet(output / f"graph_{w}.parquet", index=False)
        folds.append((fold, fit, val))

    with (output / "ledger.jsonl").open("x") as ledger:
        def fit_one(seed, arm, horizon, fold, fit, val):
            if shutil.disk_usage(ROOT).free < 2.1 * 1024**3:
                raise OSError("Disk space below experimental margin")
            cfg = deepcopy(base)
            cfg.update(seed=seed, model=f"edge_v1_{arm}")
            cfg["gat"].update(edge_types=[] if arm == "price" else ["corr"],
                              signed_messages=arm in ("signed", "centered"), center_graph_inputs=arm == "centered")
            name = f"w{fold.index}_h{horizon}_s{seed}_{arm}"
            print(f"start {name}", flush=True)
            preds, info, state = train_fold(panel, sources, fold, horizon, cfg, return_model=True, log=lambda *_: None)
            rows = panel.rows(fold.test, horizon)
            np.testing.assert_array_equal(preds[["date", "ticker"]].to_numpy(), rows[["date", "ticker"]].to_numpy())
            np.testing.assert_array_equal(preds.y_true, rows.y)
            metrics = fold_report(preds, info)
            metrics["majority"] = max(metrics["base_rate"], 1 - metrics["base_rate"])
            metrics["share_predicted_down"] = 1 - metrics["share_predicted_up"]
            metrics["daily_auc"] = {str(d.date()): auc(g.y_true, g.p_up) for d, g in preds.groupby("date")}
            preds.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
            torch.save(state.model.state_dict(), output / f"{name}.pt")
            ranges = {k: [str(d[0].date()), str(d[-1].date()), len(d)]
                      for k, d in (("fit", fit), ("validation", val), ("assessment", fold.test))}
            record = dict(id=name, arm=arm, seed=seed, config=cfg, rows=len(preds), dates=ranges,
                          metrics=metrics, **info)
            print(json.dumps(clean(record), allow_nan=False), file=ledger, flush=True)
            print(f"done {name}: AUC {metrics['within_day_auc']:.4f}, {info['seconds']:.1f}s", flush=True)

        for fold, fit, val in folds:
            for h in (1, 5):
                for arm in ARMS:
                    fit_one(42, arm, h, fold, fit, val)
        # Fixed replication isolates the edge-weight intervention from centering.
        for fold, fit, val in folds:
            for h in (1, 5):
                for arm in ("original", "signed"):
                    fit_one(43, arm, h, fold, fit, val)


if __name__ == "__main__":
    main()
