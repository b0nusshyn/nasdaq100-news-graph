"""Use all eligible historic rows with fixed training-only age decay."""
import json
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from experiments.gat_data_scaling_v1.study import OLD, ROOT, data, emit, expand, records, score
from experiments.gat_joint_v1.trainer import train_fold
from src.models.train import GraphSources
from src.nlp.rebuild_timed_news import file_hash


def age_weights(panel, days, half_life=252):
    w = np.full(len(panel.days), np.nan, dtype=np.float32)
    idx = panel.day_index(days)
    w[idx] = np.exp2((idx - idx[-1])/half_life)
    return w


def objective(logits, batch, weight):
    m = batch["target"]
    w = torch.as_tensor(weight[batch["node_day"][m].numpy()], dtype=logits.dtype, device=logits.device)
    if not bool(torch.isfinite(w).all()):
        raise ValueError("age weighting outside the fit dates")
    z = F.binary_cross_entropy_with_logits(logits[m], batch["y"][m], reduction="none")
    return (z*w).sum() / w.sum().clamp_min(torch.finfo(w.dtype).tiny)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, _, panel = data()
    baseline = records(ROOT / "runs/ledger.jsonl")
    assert len(baseline) == 16
    prior = records(OLD / "ledger.jsonl")
    out = ROOT / "recency_runs"
    out.mkdir()
    inputs = [Path(__file__), ROOT / "study.py", ROOT / "recency_protocol.md", ROOT / "runs/manifest.json",
              ROOT / "runs/ledger.jsonl", ROOT.parent / "gat_joint_v1/trainer.py"]
    (out / "manifest.json").write_text(json.dumps({str(p): file_hash(p) for p in inputs}, indent=2))
    with (out/"ledger.jsonl").open("x") as ledger:
        for w in (0, 1):
            for seed in (42, 43):
                if shutil.disk_usage(ROOT).free < 2.1*1024**3:
                    raise OSError("disk margin")
                ref = prior[f"w{w}_h5_s{seed}_original"]
                fold, fit, val, cfg, _ = expand(panel, ref, 1.)
                cfg["model"] = "history_recency_decay"
                name = f"w{w}_recency_s{seed}"
                base = baseline[f"w{w}_p100_s{seed}"]
                print("start", name, flush=True)
                _, info, state = train_fold(panel, GraphSources(prices=prices), fold, 5, cfg,
                                            objective=partial(objective, weight=age_weights(panel, fit)),
                                            predict_test=False, return_model=True, log=lambda *_: None)
                p, m = score(panel, state, fold.test, 5, info)
                cols = ["date", "ticker", "y_true"]
                orig = pd.read_parquet(ROOT / "runs" / f'{base["id"]}.parquet')
                pd.testing.assert_frame_equal(orig[cols], p[cols])
                cp, pf = out / f"{name}.pt", out / f"{name}.parquet"
                torch.save(state.model.state_dict(), cp)
                p.to_parquet(pf, index=False, compression="zstd")
                fit_m, val_m = [score(panel, state, d, 5, info)[1] for d in (fit, val)]
                emit(ledger, dict(id=name, config=cfg, seed=seed, fraction=1., half_life=252,
                                  ranges=base["ranges"], train_rows=base["train_rows"], assessment=m,
                                  train=fit_m, validation_metrics=val_m,
                                  files={str(x): file_hash(x) for x in (cp, pf)}, **info))
                print("done", name, m["within_day_auc"], flush=True)


if __name__ == "__main__":
    main()
