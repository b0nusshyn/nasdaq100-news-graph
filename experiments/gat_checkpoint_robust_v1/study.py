"""Retrospectively test an earlier-data-only checkpoint criterion."""
import json
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.gat_lstm_tuning.feature_comparison import clean
from experiments.gat_regularization_v1.study import PREVIOUS
from experiments.gat_regularization_v1.verify import read
from experiments.historical_demo_v1.replay import evaluate, infer
from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold
from src.models.ladder_report import auc, within_day_auc
from src.models.panel import build_panel
from src.models.train import GraphSources, train_fold, training_partitions
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent


def worst_half(frame):
    days = np.sort(frame.date.unique())
    left, right = np.array_split(days, 2)
    if not len(left) or not len(right):
        raise ValueError("checkpoint selection needs two chronological sets of dates")
    scores = [within_day_auc(frame[frame.date.isin(d)]) for d in (left, right)]
    return float(np.min(scores))


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    assert prices.date.max().year == 2023
    panel = build_panel(prices, labels, empty, embeddings=False)
    priors = read(PREVIOUS / "ledger.jsonl")
    out = ROOT / "runs"
    out.mkdir()
    inputs = [Path(__file__), ROOT / "protocol.md", Path("src/models/train.py"), Path("src/models/gat_lstm.py"),
              Path("src/graph/build_edges.py"), PREVIOUS / "ledger.jsonl", PREVIOUS / "manifest.json",
              *[Path("data") / f"{n}.parquet" for n in ("prices", "labels", "membership")]]
    (out / "manifest.json").write_text(json.dumps({str(p): file_hash(p) for p in inputs}, indent=2))
    with (out / "ledger.jsonl").open("x") as stream:
        for w in (0, 1):
            for seed in (42, 43):
                if shutil.disk_usage(ROOT).free < 2.1 * 1024**3:
                    raise OSError("disk safety margin")
                ref = priors[f"w{w}_h5_s{seed}_original"]
                config = deepcopy(ref["config"])
                config["model"] = "checkpoint_robust"
                dates = ref["dates"]
                start, end, _ = dates["assessment"]
                assess = panel.days[(panel.days >= start) & (panel.days <= end)]
                pos = panel.days.get_loc(assess[0])
                fold = Fold(w, panel.days[pos-312:pos-5], assess)
                fit, val = training_partitions(fold, config)
                assert all([str(d[0].date()), str(d[-1].date()), len(d)] == dates[k]
                           for k, d in (("fit", fit), ("validation", val), ("assessment", assess)))
                name = f"w{w}_h5_s{seed}_worst_half"
                print("start", name, flush=True)
                with patch("src.models.ladder_report.within_day_auc", side_effect=worst_half):
                    p, info, state = train_fold(panel, GraphSources(prices=prices), fold, 5, config,
                                                return_model=True, log=lambda *_: None)
                original = pd.read_parquet(PREVIOUS / f'{ref["id"]}.parquet')
                pd.testing.assert_frame_equal(original[["date", "ticker", "y_true"]], p[["date", "ticker", "y_true"]])
                m = evaluate(p, info)
                m["daily_auc"] = {str(d.date()): auc(g.y_true, g.p_up) for d, g in p.groupby("date")}
                evals = {}
                for kind, d in (("fit", fit), ("validation", val)):
                    f = infer(state.model, panel, state.features, state.graph, panel.day_index(d), 5, state.batch_days)
                    evals[kind] = evaluate(f, info)
                pf, ck = out / f"{name}.parquet", out / f"{name}.pt"
                p.to_parquet(pf, index=False, compression="zstd")
                torch.save(state.model.state_dict(), ck)
                row = dict(id=name, seed=seed, config=config, assessment=m, fit_eval=evals, ranges=dates,
                           source_control=ref["id"], files={str(x): file_hash(x) for x in (pf, ck)}, **info)
                print(json.dumps(clean(row), allow_nan=False), file=stream, flush=True)
                print("done", name, m["within_day_auc"], flush=True)


if __name__ == "__main__":
    main()
