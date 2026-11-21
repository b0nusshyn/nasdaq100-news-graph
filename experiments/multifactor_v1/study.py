"""Pre-registered walk-forward multi-factor model, alone and blended with the GAT-LSTM.

Hypothesis: individually weak price/fundamental/short-volume factors whose sign varies by year
combine into a steadier cross-sectional ranker when weights are learned on several years.

Fixed before any 2022/2023 result of this study was seen:
- 16 factors (below), each ranked within day to [-0.5, 0.5]; missing -> 0 (the day's middle).
- Logistic regression, L2, C=0.1, no tuning. Target y_5d (original label).
- Training rows: index members from 2019-01-02 whose 5-day label ends at least 5 sessions
  (embargo) before the first assessment day: last train date = assessment start - 11 sessions.
- Assessment: the exact archived keys of gat_edge_weight_v1 windows (2022, 2023).
- Arms: `mf` (model alone; accuracy threshold 0.5), `mf_blend` (50/50 within-day z blend with
  the scale-free-MACD GAT seeds 42/43, GAT level and validation threshold kept, as in
  factor_blend_v1).
All inputs are point-in-time: price factors use data to day t; SEC features from sec_fsds_v1
store_v3 (SEC_LAGGED_PROXY); FINRA features lagged one session.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.factor_blend_v1.study import metrics, zday

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RANGES = (("2022-07-18", "2022-12-30"), ("2023-07-24", "2023-12-20"))
FACTORS = ["lowvol", "lowmax", "rev5", "rev20", "mom_12_1", "dollar_vol", "overnight_20",
           "neg_intraday_20", "ind_rev5", "peer_lag", "vz5", "sec_op_margin", "sec_rev_growth",
           "sec_eps_chg", "short_ratio_5", "short_abn"]


def price_factors() -> pd.DataFrame:
    p = pd.read_parquet(REPO / "data/prices.parquet").sort_values(["ticker", "date"])
    p["ticker"] = p.ticker.astype(str)
    p["sector"] = p.ticker.map(pd.read_csv(REPO / "src/graph/gics.csv").set_index("ticker").gics_sector)
    g = p.groupby("ticker")
    pc = g.close.shift(1)
    p["r"] = p.close / pc - 1
    p["on"] = p.open / pc - 1
    p["intra"] = p.close / p.open - 1

    def roll(col, n, how="sum", minp=None):
        mp = minp or int(n * .8)
        return p.groupby("ticker")[col].transform(lambda s: getattr(s.rolling(n, min_periods=mp), how)())
    p["lowvol"] = -roll("r", 60, "std", 20)
    p["lowmax"] = -roll("r", 20, "max", 10)
    p["r5"] = roll("r", 5)
    p["rev5"] = -p.r5
    p["rev20"] = -roll("r", 20)
    p["mom_12_1"] = g.close.shift(21) / g.close.shift(252) - 1
    p["dv"] = np.log((p.close * p.volume).replace(0, np.nan))
    p["dollar_vol"] = roll("dv", 20, "mean")
    p["overnight_20"] = roll("on", 20)
    p["neg_intraday_20"] = -roll("intra", 20)
    lv = np.log(p.volume.replace(0, np.nan))
    mu = lv.groupby(p.ticker).transform(lambda s: s.shift(1).rolling(60, min_periods=40).mean())
    sd = lv.groupby(p.ticker).transform(lambda s: s.shift(1).rolling(60, min_periods=40).std())
    p["vz"] = (lv - mu) / sd
    p["vz5"] = roll("vz", 5, "mean")
    m = p.in_index
    p.loc[m, "ind_rev5"] = -(p.loc[m, "r5"] - p[m].groupby(["date", "sector"]).r5.transform("mean"))
    p["r5_lag"] = p.groupby("ticker").r5.shift(1)
    p.loc[m, "peer_lag"] = p[m].groupby(["date", "sector"]).r5_lag.transform("mean")
    return p


def panel() -> pd.DataFrame:
    p = price_factors()
    sec = pd.read_parquet(REPO / "experiments/sec_fsds_v1/store_v3/store.parquet",
                          columns=["ticker", "date", "sec_op_margin", "sec_rev_growth", "sec_eps_chg"])
    fin = pd.read_parquet(REPO / "experiments/finra_short_v1/features.parquet",
                          columns=["ticker", "date", "short_ratio_5", "short_abn"])
    for t in (sec, fin):
        t["ticker"] = t.ticker.astype(str)
        t["date"] = pd.to_datetime(t.date).astype("datetime64[ns]")
    p["date"] = p.date.astype("datetime64[ns]")
    d = p.loc[p.in_index].merge(sec, on=["ticker", "date"], how="left").merge(fin, on=["ticker", "date"], how="left")
    d["short_ratio_5"] = -d.short_ratio_5  # hypothesised direction (more shorting -> lower return)
    d["short_abn"] = -d.short_abn
    lb = pd.read_parquet(REPO / "data/labels.parquet", columns=["ticker", "date", "y_5d"])
    lb["ticker"] = lb.ticker.astype(str)
    lb["date"] = lb.date.astype("datetime64[ns]")
    d = d.merge(lb, on=["ticker", "date"], how="left")
    for f in FACTORS:  # within-day rank in [-0.5, 0.5], missing -> 0
        d[f] = (d.groupby("date")[f].rank(pct=True) - 0.5).fillna(0.0)
    return d[["ticker", "date", "y_5d", *FACTORS]]


def main():
    from sklearn.linear_model import LogisticRegression

    out = ROOT / "runs"
    out.mkdir()
    d = panel()
    sessions = pd.DatetimeIndex(sorted(d.date.unique()))
    gat = {json.loads(x)["id"]: json.loads(x) for x in open(REPO / "experiments/macd_scale_v1/runs/ledger.jsonl")}
    ledger = (out / "ledger.jsonl").open("x")

    def log(rec):
        print(json.dumps(rec), file=ledger, flush=True)
        m = rec["metrics"]
        print(f"{rec['id']:22s} wdAUC {m['within_day_auc']:.4f} acc {m['accuracy']:.4f} "
              f"maj {m['majority']:.4f} mcc {m['mcc']:.4f}", flush=True)

    for w, (first, _) in enumerate(RANGES):
        keys = pd.read_parquet(REPO / f"experiments/gat_edge_weight_v1/runs/w{w}_h5_s42_original.parquet",
                               columns=["ticker", "date", "y_true"])
        keys["ticker"] = keys.ticker.astype(str)
        keys["date"] = keys.date.astype("datetime64[ns]")
        last_train = sessions[sessions.get_loc(pd.Timestamp(first)) - 11]
        tr = d.loc[(d.date >= "2019-01-02") & (d.date <= last_train) & d.y_5d.notna()]
        model = LogisticRegression(C=0.1, max_iter=1000).fit(tr[FACTORS], tr.y_5d.astype(int))
        te = keys.merge(d, on=["ticker", "date"], how="left", validate="one_to_one")
        assert te[FACTORS].notna().all().all() and (te.y_true == te.y_5d).all()
        te["p_up"] = model.predict_proba(te[FACTORS])[:, 1]
        coef = dict(zip(FACTORS, np.round(model.coef_[0], 4).tolist(), strict=True))
        te[["date", "ticker", "p_up", "y_true"]].to_parquet(out / f"w{w}_mf.parquet", index=False)
        log({"id": f"w{w}_mf", "train_rows": len(tr), "train_last": str(last_train.date()),
             "coef": coef, "threshold": 0.5, "metrics": metrics(te, 0.5)})
        for s in (42, 43):
            g = pd.read_parquet(REPO / f"experiments/macd_scale_v1/runs/w{w}_h5_s{s}_macd_norm.parquet")
            g["ticker"] = g.ticker.astype(str)
            g["date"] = g.date.astype("datetime64[ns]")
            g = g.merge(te[["ticker", "date"]].assign(mf=te.p_up), on=["ticker", "date"], validate="one_to_one")
            zb = 0.5 * zday(g, "p_up").fillna(0) + 0.5 * zday(g, "mf").fillna(0)
            day = g.groupby("date").p_up
            b = g.assign(p_up=day.transform("mean") + day.transform("std") * zb)
            thr = gat[f"w{w}_h5_s{s}_macd_norm"]["metrics"]["threshold"]
            b[["date", "ticker", "p_up", "y_true"]].to_parquet(out / f"w{w}_s{s}_mf_blend.parquet", index=False)
            log({"id": f"w{w}_s{s}_mf_blend", "threshold": thr, "metrics": metrics(b, thr)})
    ledger.close()


if __name__ == "__main__":
    main()
