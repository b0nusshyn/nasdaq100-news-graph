"""Pre-registered: combine the GAT-LSTM with two stable, theory-backed price factors.

Same task (5-day up/down), same assessment keys, same labels, same thresholds/selection.

Factors (from daily returns up to and including day t; ratios, so split adjustment cancels):
  lowvol = -std(ret_1d, 60 sessions)      low-volatility anomaly
  lowmax = -max(ret_1d, 20 sessions)      MAX / lottery effect (Bali, Cakici & Whitelaw 2011)
Screened on 2019-2021 (not an assessment year): within-day AUC > 0.5 in every year 2019-2023
for both, ~0.52. composite = mean of their within-day z-scores.

Arms (all fixed before seeing 2022/2023 results of any combination):
  factor    composite only; accuracy uses threshold 0.5 on a logistic of the composite z
  blend     z_blend = 0.5*z_gat + 0.5*z_factor within each day; p = day mean(p_gat) +
            day std(p_gat) * z_blend, so the GAT's validation-selected threshold still applies
  input     GAT-LSTM re-trained with lowvol and lowmax as two extra inputs (panel's fit-only
            scaling); missing (new listings) filled with that day's cross-sectional median
GAT base for `blend`: the archived original arm (bit-reproducible) and, secondary, macd_norm.
"""
import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RANGES = (("2022-07-18", "2022-12-30"), ("2023-07-24", "2023-12-20"))
BASES = {"original": "experiments/gat_edge_weight_v1/runs/w{w}_h5_s{s}_original.parquet",
         "macd_norm": "experiments/macd_scale_v1/runs/w{w}_h5_s{s}_macd_norm.parquet"}


def factors(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices[["ticker", "date", "close", "in_index"]].copy()
    p["ticker"] = p.ticker.astype(str)
    p = p.sort_values(["ticker", "date"])
    r = p.groupby("ticker").close.pct_change()
    p["lowvol"] = -r.groupby(p.ticker).transform(lambda s: s.rolling(60, min_periods=20).std())
    p["lowmax"] = -r.groupby(p.ticker).transform(lambda s: s.rolling(20, min_periods=10).max())
    for f in ("lowvol", "lowmax"):  # same-day cross-sectional median for new listings
        p[f] = p[f].fillna(p.groupby("date")[f].transform("median"))
    return p[["ticker", "date", "lowvol", "lowmax"]]


def zday(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def metrics(p: pd.DataFrame, threshold: float) -> dict:
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    yhat = (p.p_up >= threshold).astype(int)
    wd = [roc_auc_score(g.y_true, g.p_up) for _, g in p.groupby("date") if g.y_true.nunique() == 2]
    br = p.y_true.mean()
    return {"within_day_auc": float(np.mean(wd)), "accuracy": float((yhat == p.y_true).mean()),
            "majority": float(max(br, 1 - br)), "mcc": float(matthews_corrcoef(p.y_true, yhat)),
            "rows": len(p), "daily_auc": dict(zip([str(d.date()) for d, _ in p.groupby("date")
                                                  if _.y_true.nunique() == 2], wd, strict=True))}


def main():
    import torch

    from experiments.gat_lstm_tuning.feature_comparison import clean
    from src.core.config import load_config
    from src.eval.run_baselines import load_inputs
    from src.eval.splitter import Fold
    from src.models.panel import build_panel
    from src.models.train import GraphSources, fold_report, train_fold, training_partitions

    out = ROOT / "runs"
    out.mkdir()
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    fac = factors(prices)
    ledger = (out / "ledger.jsonl").open("x")
    edge = {json.loads(x)["id"]: json.loads(x) for x in
            open(REPO / "experiments/gat_edge_weight_v1/runs/ledger.jsonl")}
    macd = {json.loads(x)["id"]: json.loads(x) for x in
            open(REPO / "experiments/macd_scale_v1/runs/ledger.jsonl")}

    def log(rec):
        print(json.dumps(clean(rec), allow_nan=False), file=ledger, flush=True)
        m = rec["metrics"]
        print(f"{rec['id']:28s} wdAUC {m['within_day_auc']:.4f} acc {m['accuracy']:.4f} "
              f"maj {m['majority']:.4f} mcc {m['mcc']:.4f}", flush=True)

    # ---- factor-only and blend arms (no fitting)
    for w in (0, 1):
        for s in (42, 43):
            for base, pattern in BASES.items():
                g = pd.read_parquet(REPO / pattern.format(w=w, s=s))
                g["ticker"] = g.ticker.astype(str)
                g = g.merge(fac, on=["ticker", "date"], how="left", validate="one_to_one")
                assert g[["lowvol", "lowmax"]].notna().all().all()
                src = edge if base == "original" else macd
                thr = src[f"w{w}_h5_s{s}_{base}"]["metrics"]["threshold"]
                zf = (zday(g, "lowvol") + zday(g, "lowmax")) / 2
                zf = (zf - zf.groupby(g.date).transform("mean")) / zf.groupby(g.date).transform("std")
                if base == "original" and s == 42:
                    f = g.assign(p_up=1 / (1 + np.exp(-zf)))
                    log({"id": f"w{w}_factor", "arm": "factor", "threshold": 0.5,
                         "metrics": metrics(f, 0.5)})
                zb = 0.5 * zday(g, "p_up").fillna(0) + 0.5 * zf.fillna(0)
                day = g.groupby("date").p_up
                b = g.assign(p_up=day.transform("mean") + day.transform("std") * zb)
                b[["date", "ticker", "p_up", "y_true"]].to_parquet(out / f"w{w}_h5_s{s}_blend_{base}.parquet")
                log({"id": f"w{w}_h5_s{s}_blend_{base}", "arm": f"blend_{base}", "threshold": thr,
                     "metrics": metrics(b, thr)})

    # ---- input arm (re-train)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    panel = build_panel(prices, labels, empty, embeddings=False)
    di = pd.Series(np.arange(len(panel.days)), index=panel.days)
    ki = pd.Series(np.arange(len(panel.tickers)), index=[str(t) for t in panel.tickers])
    extra = np.full((len(panel.days), len(panel.tickers), 2), np.nan)
    f = fac.loc[fac.date.isin(di.index) & fac.ticker.isin(ki.index)]
    extra[di[f.date].to_numpy(), ki[f.ticker].to_numpy()] = f[["lowvol", "lowmax"]].to_numpy(float)
    big = replace(panel, price=np.concatenate([panel.price, extra], axis=2),
                  price_names=tuple(panel.price_names) + ("lowvol", "lowmax"))
    base = load_config(REPO / "configs/gat_lstm.yaml")
    base["embargo_days"] = 5
    base["features"].update(news=[], embeddings=False)
    base["train"].update(epochs=12, validate_every_steps=None, validation_fraction=50 / 307,
                         preprocessing_scope="weight_fit", record_history=True)
    sources = GraphSources(prices=prices)
    for s in (42, 43):
        for w, (first, last) in enumerate(RANGES):
            if shutil.disk_usage(out).free < 2.1 * 1024**3:
                raise OSError("Disk below 2 GiB margin")
            days = panel.days[(panel.days >= first) & (panel.days <= last)]
            start = panel.days.get_loc(days[0])
            fold = Fold(w, panel.days[start - 312:start - 5], days)
            fit, val = training_partitions(fold, base)
            assert (len(fit), len(val)) == (252, 50)
            cfg = deepcopy(base)
            cfg.update(seed=s, model="factor_input")
            cfg["features"]["price"] = list(base["features"]["price"]) + ["lowvol", "lowmax"]
            cfg["gat"].update(edge_types=["corr"], signed_messages=False, center_graph_inputs=False)
            preds, info, state = train_fold(big, sources, fold, 5, cfg, return_model=True, log=lambda *_: None)
            name = f"w{w}_h5_s{s}_input"
            preds.to_parquet(out / f"{name}.parquet", index=False)
            torch.save(state.model.state_dict(), out / f"{name}.pt")
            rep = fold_report(preds, info)
            m = metrics(preds.assign(ticker=preds.ticker.astype(str)), rep["threshold"])
            log({"id": name, "arm": "input", "threshold": rep["threshold"], "config": cfg, "metrics": m, **info})
    ledger.close()


if __name__ == "__main__":
    main()
