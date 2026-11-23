"""Gated, matched GAT-LSTM comparison with SEC FSDS features (SEC_LAGGED_PROXY policy).

Refuses to fit unless the coverage gate passes on the exact original 2022 and 2023 5-day
assessment keys (archived `w{0,1}_h5_s42_original.parquet`). Arms, otherwise identical to
experiments/gat_edge_weight_v1 `original` (same windows, partitions, graph, epochs, selection):

  original  archived control (re-trained here; training is bit-deterministic)
  event     + sec_has_event, sec_recency (exp(-age/10)), sec_filing_window, sec_is_annual
  fund      + sec_has_fund and five as-filed fundamentals
  both      event + fund

Encoding: fundamentals are clipped to fixed bounds chosen before any fit (no data-fitted
statistics), missing values become 0 next to an explicit has-flag, so `has_full_window`
and the row population stay exactly those of the control. z-scoring is the panel's own
fit-window-only scaling.

    .venv/bin/python -B -m experiments.sec_fsds_v1.train_sec <store dir> <new output dir>
"""
import argparse
import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RANGES = (("2022-07-18", "2022-12-30"), ("2023-07-24", "2023-12-20"))
CLIP = {"sec_rev_growth": (-0.9, 2.0), "sec_op_margin": (-1.0, 1.0), "sec_op_margin_chg": (-0.5, 0.5),
        "sec_net_margin_chg": (-0.5, 0.5), "sec_eps_chg": (-1.0, 1.0)}
EVENT = ["sec_has_event", "sec_recency", "sec_filing_window", "sec_is_annual"]
FUND = ["sec_has_fund", *CLIP]
ARMS = {"original": [], "event": EVENT, "fund": FUND, "both": EVENT + FUND}
GATE = {"fund_row_share": 0.25, "fund_tickers": 40, "fund_sectors": 6, "event_row_share": 0.25}


def encode(store: pd.DataFrame) -> pd.DataFrame:
    s = store[["ticker", "date"]].copy()
    ev, fu = store.sec_has_event.astype(bool), store.sec_has_fund.astype(bool)
    s["sec_has_event"] = ev.astype(float)
    s["sec_recency"] = np.where(ev, np.exp(-store.sec_sessions_since_filing.fillna(0) / 10), 0.0)
    s["sec_filing_window"] = store.sec_filing_window.where(ev, 0.0).fillna(0.0)
    s["sec_is_annual"] = store.sec_is_annual.where(ev, 0.0).fillna(0.0)
    s["sec_has_fund"] = fu.astype(float)
    for c, (lo, hi) in CLIP.items():
        s[c] = store[c].clip(lo, hi).where(fu, 0.0).fillna(0.0)
    return s


def gate(store: pd.DataFrame, keys: pd.DataFrame) -> dict:
    sectors = pd.read_csv(ROOT / "src/graph/gics.csv").set_index("ticker").gics_sector
    a = keys.merge(store, on=["ticker", "date"], how="left", validate="one_to_one")
    out = {}
    for y, g in a.groupby(a.date.dt.year):
        f = g.loc[g.sec_has_fund.fillna(False).astype(bool)]
        out[int(y)] = {"rows": len(g), "fund_row_share": len(f) / len(g),
                       "event_row_share": g.sec_has_event.fillna(False).astype(bool).mean(),
                       "fund_tickers": f.ticker.nunique(), "fund_sectors": f.ticker.map(sectors).nunique()}
        out[int(y)]["pass"] = all(out[int(y)][k] >= v for k, v in GATE.items())
    return out


def augment(panel, enc: pd.DataFrame, names: list[str]):
    extra = np.zeros((len(panel.days), len(panel.tickers), len(names)))
    di = pd.Series(np.arange(len(panel.days)), index=panel.days)
    ki = pd.Series(np.arange(len(panel.tickers)), index=[str(t) for t in panel.tickers])
    e = enc.loc[enc.date.isin(di.index) & enc.ticker.isin(ki.index)]
    extra[di[e.date].to_numpy(), ki[e.ticker].to_numpy()] = e[names].to_numpy(float)
    return replace(panel, price=np.concatenate([panel.price, extra], axis=2),
                   price_names=tuple(panel.price_names) + tuple(names))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("store", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = p.parse_args()

    import torch

    from experiments.gat_lstm_tuning.feature_comparison import clean
    from experiments.gat_lstm_tuning.run import digest
    from experiments.sec_fsds_v1.features import assessment_keys
    from src.core.config import load_config
    from src.eval.run_baselines import load_inputs
    from src.eval.splitter import Fold
    from src.models.ladder_report import auc
    from src.models.panel import build_panel
    from src.models.train import GraphSources, fold_report, train_fold, training_partitions

    store = pd.read_parquet(args.store / "store.parquet")
    store["ticker"] = store.ticker.astype(str)
    g = gate(store, assessment_keys())
    print(json.dumps(g, indent=1))
    if not all(v["pass"] for v in g.values()):
        raise SystemExit("Coverage gate failed: no model fitted")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    panel = build_panel(prices, labels, empty, embeddings=False)
    enc = encode(store)
    names = list(dict.fromkeys(EVENT + FUND))
    big = augment(panel, enc, names)
    base = load_config(ROOT / "configs/gat_lstm.yaml")
    base["embargo_days"] = 5
    base["features"].update(news=[], embeddings=False)
    base["train"].update(epochs=12, validate_every_steps=None, validation_fraction=50 / 307,
                         preprocessing_scope="weight_fit", record_history=True)
    original_price = list(base["features"]["price"])
    sources = GraphSources(prices=prices)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(clean({
        "gate": g, "arms": ARMS, "clip": CLIP, "ranges": RANGES, "seeds": args.seeds,
        "store_sha256": digest(args.store / "store.parquet"), "code_sha256": digest(Path(__file__))}), indent=2))
    with (args.output / "ledger.jsonl").open("x") as ledger:
        for seed in args.seeds:
            for w, (first, last) in enumerate(RANGES):
                days = panel.days[(panel.days >= first) & (panel.days <= last)]
                start = panel.days.get_loc(days[0])
                fold = Fold(w, panel.days[start - 312:start - 5], days)
                fit, val = training_partitions(fold, base)
                assert (len(fit), len(val)) == (252, 50)
                for arm, extra in ARMS.items():
                    if shutil.disk_usage(args.output).free < 2.1 * 1024**3:
                        raise OSError("Disk space below 2 GiB margin")
                    cfg = deepcopy(base)
                    cfg.update(seed=seed, model=f"sec_fsds_{arm}")
                    cfg["features"]["price"] = original_price + extra
                    cfg["gat"].update(edge_types=["corr"], signed_messages=False, center_graph_inputs=False)
                    name = f"w{w}_h5_s{seed}_{arm}"
                    preds, info, state = train_fold(big, sources, fold, 5, cfg, return_model=True,
                                                    log=lambda *_: None)
                    m = fold_report(preds, info)
                    m["majority"] = max(m["base_rate"], 1 - m["base_rate"])
                    m["daily_auc"] = {str(d.date()): auc(x.y_true, x.p_up) for d, x in preds.groupby("date")}
                    preds.to_parquet(args.output / f"{name}.parquet", index=False, compression="zstd")
                    torch.save(state.model.state_dict(), args.output / f"{name}.pt")
                    print(json.dumps(clean(dict(id=name, arm=arm, seed=seed, config=cfg, rows=len(preds),
                                                metrics=m, **info)), allow_nan=False), file=ledger, flush=True)
                    print(f"done {name}: wdAUC {m['within_day_auc']:.4f} acc {m['accuracy']:.4f} "
                          f"maj {m['majority']:.4f} mcc {m['mcc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
