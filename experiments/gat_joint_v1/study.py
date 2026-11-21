"""Select ranking weight on earlier validation and score the selected historical models."""
import json
import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path

import pandas as pd
import torch

from experiments.gat_joint_v1.loss import joint
from experiments.gat_joint_v1.trainer import train_fold
from experiments.gat_lstm_tuning.feature_comparison import clean
from experiments.historical_demo_v1.replay import evaluate, infer
from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold
from src.models.ladder_report import auc
from src.models.panel import build_panel
from src.models.train import GraphSources, training_partitions
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent
OLD = ROOT.parent / "gat_edge_weight_v1/runs"
SMALL = ROOT.parent / "gat_regularization_v1/runs"


def records(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def data():
    prices, labels, news = load_inputs(include_news=False, legacy_labels=True)
    assert prices.date.max().year == labels.date.max().year == 2023
    return prices, labels, build_panel(prices, labels, news, embeddings=False)


def parts(panel, prior):
    ranges = prior["dates"]
    start, end, _ = ranges["assessment"]
    dates = panel.days[(panel.days >= start) & (panel.days <= end)]
    i = panel.days.get_loc(dates[0])
    fold = Fold(prior["fold"], panel.days[i-312:i-5], dates)
    fit, val = training_partitions(fold, prior["config"])
    for d, k in [(fit, "fit"), (val, "validation"), (dates, "assessment")]:
        assert [str(d[0].date()), str(d[-1].date()), len(d)] == ranges[k]
    assert panel.days.get_loc(val[0])-panel.days.get_loc(fit[-1]) == 6
    assert panel.days.get_loc(dates[0])-panel.days.get_loc(val[-1]) == 6
    return fold, fit, val


def score(panel, state, dates, h, info):
    idx = panel.day_index(dates)
    p = infer(state.model, panel, state.features, state.graph, idx, h, state.batch_days)
    m = evaluate(p, info)
    m["daily_auc"] = {str(d.date()): auc(g.y_true, g.p_up) for d, g in p.groupby("date")}
    if p.y_true.nunique() == 2:
        calls = p.p_up.ge(info["threshold"])
        m["balanced_accuracy"] = float((calls[p.y_true.eq(1)].mean() + (~calls[p.y_true.eq(0)]).mean())/2)
    return p, m


def emit(f, r):
    print(json.dumps(clean(r), allow_nan=False), file=f, flush=True)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, _, panel = data()
    out = ROOT / "runs"
    out.mkdir()
    prior = records(OLD / "ledger.jsonl")
    src = [Path(__file__), ROOT / "trainer.py", ROOT / "loss.py", ROOT / "protocol.md", OLD / "ledger.jsonl",
           SMALL / "ledger.jsonl", *[Path(f"data/{x}.parquet") for x in ("prices", "labels", "membership")],
           Path("src/models/train.py"), Path("src/models/gat_lstm.py"), Path("src/graph/build_edges.py")]
    (out / "manifest.json").write_text(json.dumps({str(p): file_hash(p) for p in src}, indent=2))
    with (out / "candidates.jsonl").open("x") as log, (out / "selected.jsonl").open("x") as selected:
        for w, h, seed in [(0, 5, 42), (1, 5, 42), (0, 1, 42), (1, 1, 42),
                           (0, 5, 43), (1, 5, 43), (0, 1, 43), (1, 1, 43)]:
            ref = prior[f"w{w}_h{h}_s{seed}_original"]
            fold, fit, val = parts(panel, ref)
            cfg = deepcopy(ref["config"])
            cfg["model"] = "joint_direction_rank"
            candidates = []
            for weight in (.10, .30):
                if shutil.disk_usage(ROOT).free < 2.1*1024**3:
                    raise OSError("disk safety margin")
                name = f"w{w}_h{h}_s{seed}_lambda{weight:.2f}"
                print("start", name, flush=True)
                _, info, state = train_fold(panel, GraphSources(prices=prices), fold, h, cfg,
                                            objective=partial(joint, weight=weight), predict_test=False,
                                            return_model=True, log=lambda *_: None)
                ck = out / f"{name}.pt"
                torch.save(state.model.state_dict(), ck)
                vf, vm = score(panel, state, val, h, info)
                path = out / f"{name}_validation.parquet"
                vf.to_parquet(path, index=False, compression="zstd")
                row = dict(id=name, lambda_rank=weight, config=cfg, fit=score(panel, state, fit, h, info)[1],
                           validation_metrics=vm, ranges=ref["dates"], seed=seed,
                           files={str(p): file_hash(p) for p in (ck, path)}, **info)
                emit(log, row)
                candidates.append((row, state))
            best, model = max(candidates, key=lambda x: (x[0]["best_val_within_day_auc"], -x[0]["lambda_rank"]))
            # Commit the chosen checkpoint before any outer-window forward pass.
            with (out / "selection.jsonl").open("a") as choices:
                emit(choices, {k: best[k] for k in ("id", "lambda_rank", "best_val_within_day_auc", "files")})
            p, m = score(panel, model, fold.test, h, best)
            for folder, key in [(OLD, ref["id"]), (SMALL, f"w{w}_h{h}_s{seed}_small")]:
                b = pd.read_parquet(folder / f"{key}.parquet")
                pd.testing.assert_frame_equal(p[["date", "ticker", "y_true"]], b[["date", "ticker", "y_true"]])
            pf = out / f'{best["id"]}_assessment.parquet'
            p.to_parquet(pf, index=False, compression="zstd")
            emit(selected, best | {"assessment": m, "prediction": str(pf), "prediction_sha256": file_hash(pf)})
            print("selected", best["id"], m["within_day_auc"], flush=True)


if __name__ == "__main__":
    main()
