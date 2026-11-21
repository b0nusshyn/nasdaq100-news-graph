"""Twelve frozen restricted-cohort development fits. Never reads post-2025 bars into the study."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.market_direction_v1.study import TREE, features, finite, lstm_scores, selective
from src.core.config import load_config
from src.core.io import read_table
from src.data.build_index_history import trading_days
from src.data.build_labels import make_labels
from src.eval.metrics import classification_metrics
from src.eval.splitter import Fold
from src.models.ladder_report import auc, within_day_auc
from src.models.panel import build_panel
from src.models.train import train_fold, training_partitions
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    acquire = json.loads((ROOT / "acquisition.json").read_text())
    old = read_table("prices")
    prior = old[old.date.dt.year.eq(2023) & old.ticker.isin(acquire["cohort"])]
    cohort = prior.assign(notional=prior.close * prior.volume).groupby("ticker").notional.median().nlargest(30).index
    src = ROOT / "external_raw_panel.parquet"
    prices = pd.read_parquet(src, filters=[("date", "<", pd.Timestamp("2026-01-01")), ("ticker", "in", list(cohort))])
    calendar = trading_days("2023-01-01", "2025-12-31")
    keys = pd.MultiIndex.from_frame(prices[["date", "ticker"]])
    expected = pd.MultiIndex.from_product([calendar, cohort], names=["date", "ticker"])
    assert len(keys) == len(expected) and keys.is_unique and keys.isin(expected).all()
    for col in ["open", "high", "low", "close"]:
        prices[col] = prices[col] * prices.adj_close / prices.close
    prices["volume"] = prices.volume.astype(float)
    prices = prices.drop(columns="adj_close").assign(in_index=True)
    labels = make_labels(prices)
    base, _ = features(prices)
    cols = list(base.columns[2:])
    base = base.merge(labels, on=["date", "ticker"], validate="one_to_one")
    empty = pd.DataFrame(columns=["date", "ticker", "has_news", "sentiment", "confidence"])
    panel = build_panel(prices, labels, empty, embeddings=False)
    cfg_path = Path("configs/lstm_price.yaml")
    config = load_config(cfg_path)
    config.update(seed=42, embargo_days=5)
    config["train"].update(validation_fraction=50 / 307, epochs=16,
                           preprocessing_scope="weight_fit", record_history=True)
    pred_dir = ROOT / "development_predictions"
    pred_dir.mkdir(exist_ok=True)
    ledger = ROOT / "development_runs.jsonl"
    old_runs = [json.loads(line) for line in ledger.read_text().splitlines()[1:]] if ledger.exists() else []
    completed = {(r["year"], r["horizon"], r["arm"]) for r in old_runs}
    with ledger.open("a" if completed else "x") as stream, threadpool_limits(limits=2):
        def emit(r):
            print(json.dumps(finite(r), allow_nan=False), file=stream, flush=True)
        if not completed:
            emit({"development_only": True, "cohort": cohort.tolist(), "tree": TREE, "config": config,
              "source_hashes": {str(p): file_hash(p) for p in [src, cfg_path, ROOT / "development_protocol.md"]}})
        for w, year in enumerate((2024, 2025)):
            test = calendar[(calendar >= f"{year}-07-15") & (calendar <= f"{year}-12-20")]
            pos = calendar.get_loc(test[0])
            fold = Fold(w, calendar[pos - 312:pos - 5], test)
            fit, val = training_partitions(fold, config)
            assert calendar.get_loc(val[0]) - calendar.get_loc(fit[-1]) == 6
            assert calendar.get_loc(test[0]) - calendar.get_loc(val[-1]) == 6
            assert calendar[calendar.get_loc(test[-1]) + 5].year == year
            for h in (1, 5):
                y, ret = f"y_{h}d", f"fwd_ret_{h}d"
                f, v, a = [base[base.date.isin(d)].sort_values(["date", "ticker"]).reset_index(drop=True)
                           for d in (fit, val, test)]
                assert all(q[y].notna().all() for q in (f, v, a))
                for arm in ("stock_logistic", "relative_regressor", "price_lstm"):
                    if (year, h, arm) in completed:
                        continue
                    # Reserve one fit for the first interrupted LSTM attempt.
                    if (year, h, arm) == (2025, 5, "price_lstm") and completed:
                        continue
                    info = {}
                    if arm == "price_lstm":
                        preds, info, state = train_fold(panel, None, fold, h, config, return_model=True,
                                                        log=lambda *_: None)
                        cv = v.merge(lstm_scores(state, panel, val, h), on=["date", "ticker"],
                                     validate="one_to_one").score.to_numpy()
                        assert np.array_equal(preds[["date", "ticker"]].to_numpy(), a[["date", "ticker"]].to_numpy())
                        p = preds.p_up.to_numpy(dtype=float).clip(1e-8, 1 - 1e-8)
                        ca = np.log(p / (1-p))
                    else:
                        if arm == "stock_logistic":
                            mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                LogisticRegression(C=.25, max_iter=300))
                            mdl.fit(f[cols], f[y])
                            score = mdl.decision_function
                        else:
                            mdl = HistGradientBoostingRegressor(**TREE)
                            mdl.fit(f[cols], f[ret] - f.groupby("date")[ret].transform("mean"))
                            score = mdl.predict
                        cv, ca = score(v[cols]), score(a[cols])
                    c = make_pipeline(StandardScaler(), LogisticRegression(C=1))
                    c.fit(cv.reshape(-1, 1), v[y])
                    pv, pa = [c.predict_proba(q.reshape(-1, 1))[:, 1] for q in (cv, ca)]
                    out = a[["date", "ticker", y, ret]].rename(columns={y: "y_true", ret: "return"})
                    out = out.assign(p_up=pa, confidence=abs(pa-.5), pred_up=pa >= .5,
                                     correct=(pa >= .5) == out.y_true.to_numpy())
                    out.to_parquet(pred_dir / f"{year}_{h}_{arm}.parquet", compression="zstd", index=False)
                    daily = [auc(g.y_true, g.p_up) for _, g in out.groupby("date")]
                    ic = [g.p_up.corr(g["return"], method="spearman") for _, g in out.groupby("date")
                          if g.p_up.nunique() > 1]
                    emit({"year": year, "horizon": h, "arm": arm, "within_day_auc": within_day_auc(out),
                          "daily_auc": daily, "rank_ic": float(np.nanmean(ic)) if ic else None,
                          "fit": [str(fit[0].date()), str(fit[-1].date())],
                          "calibrate": [str(val[0].date()), str(val[-1].date())],
                          "assessment": [str(test[0].date()), str(test[-1].date())],
                          "majority": float(max(out.y_true.mean(), 1-out.y_true.mean())),
                          "metrics": classification_metrics(out.y_true.to_numpy(), pa),
                          "selective_20": selective(out, np.quantile(abs(pv-.5), .8)), "train": info})
                    print(year, h, arm, within_day_auc(out), flush=True)


if __name__ == "__main__":
    main()
