"""Train the preregistered two-component models on retrospective windows."""
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.boosted_decision_v1.replay import threshold
from experiments.cohort_outperformance_v1.study import dataset, partition, targets
from experiments.market_direction_v1.study import features, finite
from experiments.offline_rank_v1.study import compute
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent
BASE = ROOT.parent / "offline_rank_v1" / "runs"
ALPHA = 50


def center(f, columns):
    x = f[columns].astype(float)
    return x - x.groupby(f.date).transform("mean")


def build():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ALPHA))


def components(f, stock_cols, market_cols):
    by_date = f.groupby("date")
    prevalence = by_date.y.transform("mean")
    stock = build().fit(center(f, stock_cols), f.y - prevalence)
    daily = by_date[market_cols].first().astype(float)
    market = build().fit(daily, by_date.y.mean())
    return stock, market


def scores(f, stock, market, stock_cols, market_cols):
    u = stock.predict(center(f, stock_cols))
    daily = f.groupby("date")[market_cols].first().astype(float)
    z = pd.Series(market.predict(daily), index=daily.index)
    return u, u + f.date.map(z).to_numpy()


def main():
    if shutil.disk_usage(ROOT).free < 2.1 * 1024**3:
        raise OSError("disk margin")
    prices, calendar, cohort = dataset()
    x, market_cols = features(prices)
    stock_cols = [c for c in x.columns if c not in ["date", "ticker", *market_cols]]
    closes = prices.pivot(index="date", columns="ticker", values="close")
    dst = ROOT / "runs"
    dst.mkdir()
    inputs = [Path(__file__), ROOT / "protocol.md", BASE / "ledger.jsonl",
              ROOT.parent / "post2023_validation_v1/external_raw_panel.parquet"]
    (dst / "manifest.json").write_text(json.dumps(dict(alpha=ALPHA, stock=stock_cols, market=market_cols,
        cohort=cohort, inputs_sha256={str(p): file_hash(p) for p in inputs}), indent=2))
    with (dst / "ledger.jsonl").open("x") as log, threadpool_limits(limits=2):
        for year in (2024, 2025):
            fold, cal = partition(calendar, year)
            fit = fold.train[:252]
            assert calendar.get_loc(fit[-1])+5 < calendar.get_loc(cal[0])
            assert calendar.get_loc(cal[-1])+5 < calendar.get_loc(fold.test[0])
            for h in (1, 5):
                y = targets(closes, calendar, h).assign(y=lambda p: p.absolute.gt(0).astype("int8"))
                data = x.merge(y[["date", "ticker", "y"]], on=["date", "ticker"], validate="one_to_one")
                f, c, a = [data[data.date.isin(d)].sort_values(["date", "ticker"]).reset_index(drop=True)
                           for d in (fit, cal, fold.test)]
                name = f"{year}_{h}d"
                ref = pd.read_parquet(BASE / f"{name}_logistic.parquet")
                pd.testing.assert_frame_equal(ref[["date", "ticker", "y"]], a[["date", "ticker", "y"]])
                stock, market = components(f, stock_cols, market_cols)
                joblib.dump({"stock": stock, "market": market}, dst / f"{name}.joblib", compress=3)
                cs, sa = [scores(q, stock, market, stock_cols, market_cols) for q in (c, a)]
                z = a[["date", "ticker", "y"]].copy()
                results = {}
                for k, cv, av in zip(("stock", "combined"), cs, sa, strict=True):
                    cut = threshold(c.y, cv)
                    results[k] = dict(cutoff=cut, metrics=compute(a, av, cut))
                    z[k], z[f"{k}_up"] = av, av >= cut
                z.to_parquet(dst / f"{name}.parquet", index=False)
                print(json.dumps(finite(dict(id=name, year=year, horizon=h, rows=len(z), results=results,
                    ranges=[[str(d[0].date()), str(d[-1].date())] for d in (fit, cal, fold.test)]))),
                    file=log, flush=True)
                print(name, results["combined"]["metrics"]["accuracy"], flush=True)


if __name__ == "__main__":
    main()
