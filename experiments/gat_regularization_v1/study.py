"""Two isolated regularization interventions on matched historical GAT fits."""
import json
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from experiments.gat_lstm_tuning.feature_comparison import clean
from experiments.historical_demo_v1.replay import evaluate, infer
from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold
from src.models.ladder_report import auc
from src.models.panel import build_panel
from src.models.train import GraphSources, build_model, train_fold, training_partitions
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent
PREVIOUS = ROOT.parent / "gat_edge_weight_v1/runs"


def drop_pairs(model, args):
    """Training-only Bernoulli mask shared by reverse arcs inside each disjoint graph."""
    if not model.training or not args[1].numel():
        return args
    src, dst = args[1]
    pairs = torch.stack([torch.minimum(src, dst), torch.maximum(src, dst)], dim=1)
    unique, inv = torch.unique(pairs, dim=0, return_inverse=True)
    keep = (torch.rand(len(unique), device=src.device) >= .25)[inv]
    return args[0], args[1][:, keep], args[2][keep], args[3][keep]


def with_drop_edge(n_features, edges, cfg):
    model = build_model(n_features, edges, cfg)
    model.register_forward_pre_hook(drop_pairs)
    return model


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, labels, news = load_inputs(include_news=False, legacy_labels=True)
    assert prices.date.max().year == 2023
    panel = build_panel(prices, labels, news, embeddings=False)
    prior = {r["id"]: r for r in map(json.loads, (PREVIOUS / "ledger.jsonl").read_text().splitlines())}
    out = ROOT / "runs"
    out.mkdir()
    provenance = [Path(__file__), ROOT / "protocol.md", PREVIOUS / "ledger.jsonl", PREVIOUS / "manifest.json",
                  Path("src/models/train.py"), Path("src/models/gat_lstm.py"), Path("src/graph/build_edges.py"),
                  *[Path("data") / f"{n}.parquet" for n in ("prices", "labels", "membership")]]
    (out / "manifest.json").write_text(json.dumps({str(p): file_hash(p) for p in provenance}, indent=2))
    first = (0, 5, 42, "dropedge")
    full = [(w, h, seed, arm) for seed in (42, 43) for w in (0, 1) for h in (1, 5)
            for arm in ("small", "dropedge")]
    with (out / "ledger.jsonl").open("x") as stream:
        for w, h, seed, arm in [first, *[x for x in full if x != first]]:
            if shutil.disk_usage(ROOT).free < 2.1 * 1024**3:
                raise OSError("disk safety margin")
            ref = prior[f"w{w}_h{h}_s{seed}_original"]
            ranges = ref["dates"]
            assess = panel.days[(panel.days >= ranges["assessment"][0]) & (panel.days <= ranges["assessment"][1])]
            start = panel.days.get_loc(assess[0])
            fold = Fold(w, panel.days[start-312:start-5], assess)
            cfg = deepcopy(ref["config"])
            cfg["model"] = f"regularization_{arm}"
            fit, val = training_partitions(fold, cfg)
            for d, k in ((fit, "fit"), (val, "validation"), (assess, "assessment")):
                assert ranges[k] == [str(d[0].date()), str(d[-1].date()), len(d)]
            if arm == "small":
                cfg["encoder"]["hidden"] = 32
            name = f"w{w}_h{h}_s{seed}_{arm}"
            print("start", name, flush=True)
            build = build_model if arm == "small" else with_drop_edge
            with patch("src.models.train.build_model", side_effect=build):
                p, info, state = train_fold(panel, GraphSources(prices=prices), fold, h, cfg,
                                            return_model=True, log=lambda *_: None)
            a = pd.read_parquet(PREVIOUS / f'{ref["id"]}.parquet')
            pd.testing.assert_frame_equal(p[["date", "ticker", "y_true"]], a[["date", "ticker", "y_true"]])
            pm = evaluate(p, info)
            pm["daily_auc"] = {str(d.date()): auc(g.y_true, g.p_up) for d, g in p.groupby("date")}
            related = {}
            for kind, dates in (("fit", fit), ("validation", val)):
                b = infer(state.model, panel, state.features, state.graph, panel.day_index(dates), h, state.batch_days)
                related[kind] = evaluate(b, info)
            pf, ck = out / f"{name}.parquet", out / f"{name}.pt"
            p.to_parquet(pf, index=False, compression="zstd")
            torch.save(state.model.state_dict(), ck)
            r = dict(id=name, seed=seed, arm=arm, config=cfg, fit_eval=related, assessment=pm, ranges=ranges,
                     source_control=ref["id"], files={str(x): file_hash(x) for x in (pf, ck)}, **info)
            print(json.dumps(clean(r), allow_nan=False), file=stream, flush=True)
            print("done", name, pm["within_day_auc"], flush=True)


if __name__ == "__main__":
    main()
