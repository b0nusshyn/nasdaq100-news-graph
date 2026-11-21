"""Train matched chronological subsets and audit historical sample utilization."""
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from experiments.gat_joint_v1.study import OLD, data, emit, parts, records, score
from src.core.io import read_table
from src.eval.splitter import Fold
from src.graph.build_edges import corr_edges
from src.models.train import GraphSources, train_fold, training_partitions
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent
FRACTIONS = (.25, .50, .75, 1.)


def nested(history, fraction):
    if not 0 < fraction <= 1 or not len(history) or not history.is_monotonic_increasing or history.has_duplicates:
        raise ValueError("need unique ordered historical dates and a valid fraction")
    return history[-math.ceil(len(history)*fraction):]


def expand(panel, ref, fraction):
    fold, fit, val = parts(panel, ref)
    available = panel.days[(panel.days >= "2019-01-01") & (panel.days <= fit[-1])]
    days = nested(available, fraction)
    span = fold.train[fold.train > fit[-1]]
    outer = Fold(fold.index, days.append(span), fold.test)
    cfg = deepcopy(ref["config"])
    cfg["train"]["validation_fraction"] = (len(val) - .25)/len(outer.train)
    real, check = training_partitions(outer, cfg)
    assert real.equals(days) and check.equals(val)
    assert panel.days.get_loc(val[0]) - panel.days.get_loc(days[-1]) == 6
    return outer, days, val, cfg, len(available)


def audit(prices, panel):
    p = prices[prices.date.ge("2019-01-01")]
    m = read_table("membership")
    expected = []
    days = panel.days[panel.days >= "2019-01-01"]
    for r in m.itertuples(index=False):
        d = days[(days >= r.start) & ((days <= r.end) if pd.notna(r.end) else True)]
        if len(d):
            expected.append(pd.DataFrame({"date": d, "ticker": r.ticker}))
    keys = pd.concat(expected).drop_duplicates().merge(
        p.loc[p.in_index, ["date", "ticker"]], how="outer", indicator=True)
    results = {"rows": len(p), "tickers": p.ticker.nunique(), "sessions": len(days),
               "member_rows": int(p.in_index.sum()), "nonmember_rows": int((~p.in_index).sum()),
               "member_tickers": p.loc[p.in_index, "ticker"].nunique(),
               "historical_membership_join": keys["_merge"].value_counts().to_dict()}
    for h in (1, 5):
        r = panel.rows(days, h)
        results[str(h)] = {"eligible": len(r), "tickers": r.ticker.nunique(), "sessions": r.date.nunique(),
                           "member_rows_missing_label": int(p.in_index.sum())-len(r)}
    return results


def run():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, _, panel = data()
    prior = records(OLD / "ledger.jsonl")
    out = ROOT / "runs"
    out.mkdir()
    (out / "audit.json").write_text(json.dumps(audit(prices, panel), indent=2))
    paths = [Path(__file__), ROOT / "protocol.md", Path("src/models/train.py"), Path("src/graph/build_edges.py"),
             Path("src/models/panel.py"), OLD / "ledger.jsonl",
             *[Path(f"data/{n}.parquet") for n in ("prices", "labels", "membership")]]
    (out / "manifest.json").write_text(json.dumps({str(p): file_hash(p) for p in paths}, indent=2))
    with (out / "ledger.jsonl").open("x") as log:
        for w in (0, 1):
            for fraction in FRACTIONS:
                fold, fit, val, cfg, full = expand(panel, prior[f"w{w}_h5_s42_original"], fraction)
                name = f"w{w}_p{round(fraction*100):03}"
                g = corr_edges(prices, [Fold(w, fit, fold.test)])
                if g.empty:
                    emit(log, {"id": name, "skipped": "no valid fit-only graph", "fit_days": len(fit)})
                    continue
                g.to_parquet(out / f"{name}_graph.parquet", index=False)
                for seed in (42, 43):
                    if shutil.disk_usage(ROOT).free < 2.1*1024**3:
                        raise OSError("disk safety margin")
                    cfg.update(seed=seed, model="history_scaling")
                    ident = f"{name}_s{seed}"
                    print("start", ident, len(fit), flush=True)
                    preds, info, state = train_fold(panel, GraphSources(prices=prices), fold, 5, cfg,
                                                    return_model=True, log=lambda *_: None)
                    old = pd.read_parquet(OLD / f"w{w}_h5_s{seed}_original.parquet")
                    cols = ["date", "ticker", "y_true"]
                    pd.testing.assert_frame_equal(preds[cols], old[cols])
                    af, am = score(panel, state, fold.test, 5, info)
                    pd.testing.assert_frame_equal(preds[cols+["p_up"]], af[cols+["p_up"]])
                    tr = panel.rows(fit, 5)
                    d, k = panel.day_index(tr.date), panel.tickers.get_indexer(tr.ticker)
                    incomplete = int((state.features.x[d, k, state.features.names.index("has_full_window")] == 0).sum())
                    missing_graph = int((~tr.ticker.isin(g.src)).sum())
                    cp, pf = out / f"{ident}.pt", out / f"{ident}.parquet"
                    torch.save(state.model.state_dict(), cp)
                    preds.to_parquet(pf, index=False, compression="zstd")
                    fit_m, val_m = [score(panel, state, dates, 5, info)[1] for dates in (fit, val)]
                    ranges = {key: [str(d[0].date()), str(d[-1].date()), len(d)]
                              for key, d in (("fit", fit), ("validation", val), ("assessment", fold.test))}
                    emit(log, dict(id=ident, config=deepcopy(cfg), seed=seed, fraction=fraction,
                                   eligible_history_sessions=full, train_rows=len(tr),
                                   train_tickers=tr.ticker.nunique(), incomplete_kept=incomplete,
                                   no_fit_graph_peer_kept=missing_graph, ranges=ranges, train=fit_m,
                                   validation_metrics=val_m, assessment=am,
                                   files={str(p): file_hash(p) for p in (cp, pf)}, **info))
                    print("done", ident, am["within_day_auc"], flush=True)


if __name__ == "__main__":
    run()
