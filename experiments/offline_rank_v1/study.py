"""Fit predeclared absolute-direction models and a date-grouped ranking ablation."""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.boosted_decision_v1.replay import metrics, threshold
from experiments.cohort_outperformance_v1.study import dataset, partition, targets
from experiments.market_direction_v1.study import TREE, features, finite
from src.models.ladder_report import auc, within_day_auc
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent
ARMS = ("logistic", "boosted", "pairwise", "pairwise_residual")
RANK = dict(objective="rank:pairwise", n_estimators=90, max_depth=2, learning_rate=.05,
            reg_lambda=8, lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=30,
            random_state=42, n_jobs=2, tree_method="hist")


def residual_features(prices):
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = np.log(close).diff()
    market = returns.mean(axis=1)
    beta = returns.rolling(20).cov(market).div(market.rolling(20).var(), axis=0).shift(1)
    residual = returns - beta.mul(market, axis=0)
    values = {}
    for n in (5, 20):
        s = residual.rolling(n).sum()
        values[f"res_{n}"] = s.stack()
        values[f"res_rank_{n}"] = s.rank(axis=1, pct=True).stack()
    return pd.DataFrame(values).reset_index()


def qids(frame):
    if not frame.date.is_monotonic_increasing:
        raise ValueError("ranking dates must be sorted")
    return pd.factorize(frame.date, sort=False)[0]


def compute(f, score, cut):
    r = metrics(f.y, pd.Series(score >= cut, index=f.index))
    p = f.assign(y_true=f.y, p_up=score)
    r["auc"] = within_day_auc(p)
    r["daily"] = {str(d.date()): auc(g.y_true, g.p_up) for d, g in p.groupby("date")}
    return r


def main():
    assert shutil.disk_usage(ROOT).free > 2.1 * 1024**3
    prices, calendar, cohort = dataset()
    stock, _ = features(prices)
    cols = list(stock.columns[2:])
    extra = residual_features(prices)
    stock = stock.merge(extra, on=["date", "ticker"], how="left", validate="one_to_one").sort_values(["date","ticker"])
    prices_wide = prices.pivot(index="date", columns="ticker", values="close")
    out = ROOT / "runs"
    out.mkdir(exist_ok=True)
    log = out / "ledger.jsonl"
    completed = {json.loads(line)["id"] for line in log.read_text().splitlines()} if log.exists() else set()
    sources = [Path(__file__), ROOT / "protocol.md",
               ROOT.parent / "post2023_validation_v1/external_raw_panel.parquet",
               ROOT.parent / "market_direction_v1/study.py", ROOT.parent / "cohort_outperformance_v1/study.py"]
    manifest = out / ("resume_manifest.json" if completed else "manifest.json")
    manifest.write_text(json.dumps(dict(rank=RANK, tree=TREE, cohort=cohort, features=cols,
        residual=list(extra.columns[2:]), xgboost=xgb.__version__, hashes={str(p): file_hash(p) for p in sources}),
        indent=2))
    with log.open("a" if completed else "x") as ledger, threadpool_limits(limits=2):
        for year in (2024, 2025):
            fold, cal = partition(calendar, year)
            fit = fold.train[:252]
            assert len(fit) == 252 and len(cal) == 50
            assert calendar.get_loc(fit[-1])+5 < calendar.get_loc(cal[0])
            assert calendar.get_loc(cal[-1])+5 < calendar.get_loc(fold.test[0])
            assert calendar[calendar.get_loc(fold.test[-1])+5].year == year
            for h in (1, 5):
                y = targets(prices_wide, calendar, h).assign(y=lambda p: p.absolute.gt(0).astype("int8"))
                data = stock.merge(y[["date", "ticker", "y"]], on=["date", "ticker"], validate="one_to_one")
                f, c, a = [data[data.date.isin(d)] for d in (fit, cal, fold.test)]
                assert all(len(s)==len(d)*30 for s,d in zip((f,c,a),(fit,cal,fold.test),strict=True))
                for arm in ARMS:
                    name = f"{year}_{h}d_{arm}"
                    if name in completed:
                        continue
                    print("start", name, flush=True)
                    used = cols + (list(extra.columns[2:]) if arm == "pairwise_residual" else [])
                    if arm == "logistic":
                        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                              LogisticRegression(C=.25, max_iter=300))
                    elif arm == "boosted":
                        model = HistGradientBoostingClassifier(**TREE)
                    else:
                        model = xgb.XGBRanker(**RANK)
                    ranking = arm.startswith("pairwise")
                    model.fit(f[used].astype(float), f.y, **({"qid": qids(f)} if ranking else {}))
                    score = model.predict if ranking else model.decision_function
                    fs, cs, sa = [score(p[used].astype(float)) for p in (f,c,a)]
                    cut = threshold(c.y, cs)
                    joblib.dump(model, out / f"{name}.joblib", compress=3)
                    assessed = a[["date","ticker","y","market_vol","market_20"]].assign(score=sa, up=sa>=cut)
                    assessed.to_parquet(out / f"{name}.parquet", index=False)
                    median_vol = f.drop_duplicates("date").market_vol.median()
                    masks = {"high_vol": a.market_vol.ge(median_vol), "low_vol": a.market_vol.lt(median_vol),
                             "rising": a.market_20.ge(0), "falling": a.market_20.lt(0)}
                    regime = {k: {"dates": a[mask].date.nunique(), **compute(a[mask], sa[mask], cut)}
                              for k, mask in masks.items() if a[mask].date.nunique() >= 10}
                    report = dict(id=name, year=year, horizon=h, arm=arm, rows=len(a), cut=cut,
                                  fit_auc=compute(f,fs,cut)["auc"],
                                  calibration_auc=compute(c,cs,cut)["auc"], assessment=compute(a,sa,cut),
                                  regimes=regime, features=used,
                                  dates=[[str(d[0].date()),str(d[-1].date()),len(d)] for d in (fit,cal,fold.test)])
                    print(json.dumps(finite(report), allow_nan=False),file=ledger,flush=True)
                    print("done", name, report["assessment"]["auc"], flush=True)


if __name__ == "__main__":
    main()
