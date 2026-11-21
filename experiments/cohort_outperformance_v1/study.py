"""Retrospective 2024–2025 cohort-relative targets, learners and abstention."""
import json
import shutil
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.market_direction_v1.study import TREE, features, finite, lstm_scores, selective
from src.core.config import load_config
from src.core.paths import table_path
from src.data.build_index_history import trading_days
from src.eval.metrics import classification_metrics
from src.eval.splitter import Fold
from src.models.ladder_report import auc, within_day_auc
from src.models.panel import build_panel
from src.models.train import GraphSources, train_fold, training_partitions
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent
PRIOR = ROOT.parent / "post2023_validation_v1"
LEVELS = (.05, .10, .20, .50, 1.)
ARMS = ("logistic", "boosted_binary", "boosted_relative")
LOCK = pd.Timestamp("2026-01-01")


def targets(close, calendar, horizon):
    """Use exact scheduled endpoints and require the same complete benchmark each day."""
    close = close.reindex(calendar).rename_axis(index="date", columns="ticker")
    close = close.where(np.isfinite(close) & close.gt(0))
    raw = close.shift(-horizon) / close - 1
    excess = raw.sub(raw.mean(axis=1, skipna=False), axis=0)
    label = excess.gt(0).where(excess.notna())
    result = pd.concat([
        x.stack().rename(name)
        for x, name in [(raw, "absolute"), (excess, "excess"), (label, "y")]
    ], axis=1).dropna()
    return result.reset_index().astype({"y": "int8"})


def dataset():
    header = json.loads((PRIOR / "development_runs.jsonl").read_text().splitlines()[0])
    cohort = header["cohort"]
    path = PRIOR / "external_raw_panel.parquet"
    prices = pd.read_parquet(path, filters=[("date", "<", LOCK), ("ticker", "in", cohort)])
    calendar = trading_days("2023-01-01", "2025-12-31")
    assert prices.date.lt(LOCK).all()
    keys = pd.MultiIndex.from_frame(prices[["date", "ticker"]])
    expected = pd.MultiIndex.from_product([calendar, cohort], names=keys.names)
    assert len(cohort) == 30 and len(keys) == len(expected) and keys.is_unique and keys.isin(expected).all()
    factor = prices.adj_close / prices.close
    assert np.isfinite(factor).all() and factor.gt(0).all()
    for col in ("open", "high", "low", "close"):
        prices[col] *= factor
    prices = prices.drop(columns="adj_close").assign(in_index=True).sort_values(["date", "ticker"])
    return prices, calendar, cohort


def partition(calendar, year):
    assessment = calendar[(calendar >= f"{year}-07-15") & (calendar <= f"{year}-12-20")]
    start = calendar.get_loc(assessment[0])
    cal = calendar[start - 55:start - 5]
    fold = Fold(year, calendar[start - 342:start - 60], assessment)
    return fold, cal


def describe(data, label, score):
    y = data[label].to_numpy()
    pred = data.p_up.to_numpy()
    local = data.assign(y_true=y, p_up=data[score])
    metrics = classification_metrics(y, pred)
    metrics.update(base_rate=float(y.mean()), majority=float(max(y.mean(), 1-y.mean())),
                   within_day_auc=within_day_auc(local),
                   daily_auc=[auc(g.y_true, g.p_up) for _, g in local.groupby("date")])
    return metrics


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, calendar, cohort = dataset()
    features_frame, _ = features(prices)
    cols = list(features_frame.columns[2:])
    close = prices.pivot(index="date", columns="ticker", values="close")
    labels = {h: targets(close, calendar, h) for h in (1, 5)}
    labels_panel = labels[5][["date", "ticker", "y"]].rename(columns={"y": "y_5d"})
    empty = pd.DataFrame(columns=["date", "ticker", "has_news", "sentiment", "confidence"])
    panel = build_panel(prices, labels_panel, empty, embeddings=False)
    cfg_path = Path("configs/gat_lstm.yaml")
    config = load_config(cfg_path)
    config.update(seed=42, embargo_days=5)
    config["train"].update(validation_fraction=25 / 282, epochs=12, validate_every_steps=None,
                           preprocessing_scope="weight_fit", record_history=True)
    config["features"].update(news=[], embeddings=False)
    out = ROOT / "runs"
    out.mkdir()
    paths = [Path(__file__), ROOT / "protocol.md", PRIOR / "external_raw_panel.parquet", cfg_path,
             PRIOR / "development_runs.jsonl", table_path("membership"),
             Path("experiments/market_direction_v1/study.py"), *Path("src/models").glob("*.py")]
    (out / "manifest.json").write_text(json.dumps({
        "seed": 42, "cohort": cohort, "config": config, "boosting": TREE, "tabular_features": cols,
        "levels": LEVELS, "torch": str(torch.__version__), "sha256": {str(p): file_hash(p) for p in paths},
    }, indent=2))

    with (out / "ledger.jsonl").open("x") as ledger, threadpool_limits(limits=2):
        for year in (2024, 2025):
            fold, cal = partition(calendar, year)
            fit, val = training_partitions(fold, config)
            assert list(map(len, (fit, val, cal))) == [252, 25, 50]
            for earlier, later in [(fit, val), (val, cal), (cal, fold.test)]:
                assert calendar.get_loc(later[0]) - calendar.get_loc(earlier[-1]) == 6
            for h in (1, 5):
                data = features_frame.merge(labels[h], on=["date", "ticker"], validate="one_to_one")
                f, v, c, a = [data[data.date.isin(d)].sort_values(["date", "ticker"]).reset_index(drop=True)
                              for d in (fit, val, cal, fold.test)]
                assert all(len(q) == 30 * len(d) for q, d in zip((f, v, c, a), (fit, val, cal, fold.test), strict=True))
                for arm in ARMS + (("price_lstm", "gat_corr") if h == 5 else ()):
                    assert shutil.disk_usage(ROOT).free > 2.1 * 1024**3
                    run_id = f"{year}_{h}d_{arm}"
                    print("start", run_id, flush=True)
                    t = time.perf_counter()
                    info, cfg = {}, None
                    if arm in ("price_lstm", "gat_corr"):
                        cfg = deepcopy(config)
                        cfg["gat"]["edge_types"] = ["corr"] if arm == "gat_corr" else []
                        preds, info, state = train_fold(panel, GraphSources(prices=prices), fold, h, cfg,
                                                        return_model=True, log=lambda *_: None)
                        np.testing.assert_array_equal(preds[["date", "ticker"]].to_numpy(),
                                                      a[["date", "ticker"]].to_numpy())
                        np.testing.assert_array_equal(preds.y_true, a.y)

                        def score(frame, fitted=state, horizon=h):
                            s = lstm_scores(fitted, panel, pd.DatetimeIndex(frame.date.unique()), horizon)
                            return frame.merge(s, on=["date", "ticker"], validate="one_to_one").score.to_numpy()

                        torch.save(state.model.state_dict(), out / f"{run_id}.pt")
                    else:
                        if arm == "logistic":
                            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                  LogisticRegression(C=.25, max_iter=300))
                            model.fit(f[cols], f.y)
                            predict = model.decision_function
                        else:
                            model = (HistGradientBoostingClassifier if arm == "boosted_binary"
                                     else HistGradientBoostingRegressor)(**TREE)
                            model.fit(f[cols], f.y if arm == "boosted_binary" else f.excess)
                            predict = model.decision_function if arm == "boosted_binary" else model.predict

                        def score(frame, predict=predict):
                            return predict(frame[cols])
                        joblib.dump(model, out / f"{run_id}.joblib", compress=3)
                    sf, sv, sc, sa = map(score, (f, v, c, a))
                    calibration = make_pipeline(StandardScaler(), LogisticRegression(C=1))
                    calibration.fit(sc[:, None], c.y)
                    joblib.dump(calibration, out / f"{run_id}_calibrator.joblib", compress=3)
                    pc, pa = [calibration.predict_proba(q[:, None])[:, 1] for q in (sc, sa)]
                    z = a[["date", "ticker", "absolute", "excess"]].assign(
                        y_true=a.y, direction=a.absolute.gt(0), score=sa, p_up=pa,
                        confidence=abs(pa - .5), pred_up=pa >= .5, correct=(pa >= .5) == a.y)
                    limits = {str(k): float(np.quantile(abs(pc-.5), 1-k)) if k < 1 else 0. for k in LEVELS}
                    selected = {k: selective(z, threshold) for k, threshold in limits.items()}
                    metrics = describe(z, "y_true", "score")
                    secondary = describe(z, "direction", "score")
                    ranks = [g.score.corr(g.excess, method="spearman") for _, g in z.groupby("date")
                             if g.score.nunique() > 1]
                    z.to_parquet(out / f"{run_id}.parquet", compression="zstd", index=False)
                    ranges = {k: [str(d[0].date()), str(d[-1].date()), len(d)]
                              for k, d in zip(("fit", "validation", "calibration", "assessment"),
                                              (fit, val, cal, fold.test), strict=True)}
                    row = dict(id=run_id, year=year, horizon=h, arm=arm, seed=42, rows=len(z), dates=ranges,
                               metrics=metrics, secondary_direction=secondary, rank_ic=float(np.mean(ranks)),
                               fit_auc=within_day_auc(f.assign(y_true=f.y, p_up=sf)),
                               validation_auc=within_day_auc(v.assign(y_true=v.y, p_up=sv)),
                               calibration_auc=within_day_auc(c.assign(y_true=c.y, p_up=sc)),
                               calibration_slope=float(calibration[-1].coef_[0, 0]), selective=selected,
                               seconds=time.perf_counter()-t, config=cfg, history=info)
                    print(json.dumps(finite(row), allow_nan=False), file=ledger, flush=True)
                    print("done", run_id, metrics["within_day_auc"], flush=True)


if __name__ == "__main__":
    main()
