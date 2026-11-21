"""Pre-registered seed ensemble of the scale-free-MACD GAT-LSTM, alone and blended with factors.

Seeds 42-51 (all ten kept; none selected on assessment results). Ensemble p_up = mean of the
ten seeds' p_up; threshold = mean of the ten validation-selected thresholds (no assessment
information). The factor blend is exactly experiments/factor_blend_v1 (50/50 within-day z).
Individual-seed results are reported to show the spread that seed *picking* would exploit.
"""
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.factor_blend_v1.study import factors, metrics, zday
from experiments.macd_scale_v1.study import normalise_macd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RANGES = (("2022-07-18", "2022-12-30"), ("2023-07-24", "2023-12-20"))
SEEDS = list(range(42, 52))


def main():
    import torch

    from experiments.gat_lstm_tuning.feature_comparison import clean
    from src.core.config import load_config
    from src.eval.run_baselines import load_inputs
    from src.eval.splitter import Fold
    from src.models.panel import build_panel
    from src.models.train import GraphSources, fold_report, train_fold, training_partitions

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = ROOT / "runs"
    out.mkdir()
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    panel = normalise_macd(build_panel(prices, labels, empty, embeddings=False), prices)
    fac = factors(prices)
    base = load_config(REPO / "configs/gat_lstm.yaml")
    base["embargo_days"] = 5
    base["features"].update(news=[], embeddings=False)
    base["train"].update(epochs=12, validate_every_steps=None, validation_fraction=50 / 307,
                         preprocessing_scope="weight_fit", record_history=True)
    sources = GraphSources(prices=prices)
    ledger = (out / "ledger.jsonl").open("x")

    def log(rec):
        print(json.dumps(clean(rec), allow_nan=False), file=ledger, flush=True)
        m = rec["metrics"]
        print(f"{rec['id']:24s} wdAUC {m['within_day_auc']:.4f} acc {m['accuracy']:.4f} "
              f"maj {m['majority']:.4f} mcc {m['mcc']:.4f}", flush=True)

    for w, (first, last) in enumerate(RANGES):
        days = panel.days[(panel.days >= first) & (panel.days <= last)]
        start = panel.days.get_loc(days[0])
        fold = Fold(w, panel.days[start - 312:start - 5], days)
        fit, val = training_partitions(fold, base)
        assert (len(fit), len(val)) == (252, 50)
        preds, thresholds = [], []
        for s in SEEDS:
            if shutil.disk_usage(out).free < 2.1 * 1024**3:
                raise OSError("Disk below 2 GiB margin")
            cfg = deepcopy(base)
            cfg.update(seed=s, model="edge_v1_macd_norm")
            cfg["gat"].update(edge_types=["corr"], signed_messages=False, center_graph_inputs=False)
            p, info, state = train_fold(panel, sources, fold, 5, cfg, return_model=True, log=lambda *_: None)
            p["ticker"] = p.ticker.astype(str)
            thr = fold_report(p, info)["threshold"]
            name = f"w{w}_h5_s{s}_macd_norm"
            p.to_parquet(out / f"{name}.parquet", index=False)
            torch.save(state.model.state_dict(), out / f"{name}.pt")
            log({"id": name, "arm": "single", "seed": s, "threshold": thr, "metrics": metrics(p, thr)})
            preds.append(p)
            thresholds.append(thr)
        ens = preds[0][["date", "ticker", "y_true"]].copy()
        for p in preds[1:]:
            assert p[["date", "ticker"]].equals(preds[0][["date", "ticker"]])
        ens["p_up"] = np.mean([p.p_up.to_numpy() for p in preds], axis=0)
        thr = float(np.mean(thresholds))
        ens.to_parquet(out / f"w{w}_h5_ensemble10.parquet", index=False)
        log({"id": f"w{w}_ensemble10", "arm": "ensemble", "threshold": thr, "metrics": metrics(ens, thr)})
        g = ens.merge(fac, on=["ticker", "date"], how="left", validate="one_to_one")
        zf = (zday(g, "lowvol") + zday(g, "lowmax")) / 2
        zf = (zf - zf.groupby(g.date).transform("mean")) / zf.groupby(g.date).transform("std")
        zb = 0.5 * zday(g, "p_up").fillna(0) + 0.5 * zf.fillna(0)
        day = g.groupby("date").p_up
        b = g.assign(p_up=day.transform("mean") + day.transform("std") * zb)
        b[["date", "ticker", "p_up", "y_true"]].to_parquet(out / f"w{w}_h5_ensemble10_blend.parquet", index=False)
        log({"id": f"w{w}_ensemble10_blend", "arm": "ensemble_blend", "threshold": thr,
             "metrics": metrics(b, thr)})
    ledger.close()


if __name__ == "__main__":
    main()
