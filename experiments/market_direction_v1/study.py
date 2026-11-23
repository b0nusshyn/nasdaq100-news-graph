"""Matched chronological direction and abstention study on the historical panel."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.core.config import load_config
from src.core.paths import table_path
from src.eval.metrics import classification_metrics
from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold
from src.models.ladder_report import within_day_auc
from src.models.panel import build_panel
from src.models.train import make_batch, train_fold, training_partitions
from src.nlp.rebuild_timed_news import file_hash

ROOT = Path(__file__).parent
MODELS = ("market_logistic", "stock_logistic", "boosted_binary", "boosted_relative", "price_lstm")
WINDOWS = (("2021-07-12", "2021-12-31"), ("2022-07-18", "2022-12-30"))
LEVELS = (.05, .10, .20, .50)
TREE = dict(max_iter=90, learning_rate=.05, max_leaf_nodes=7, min_samples_leaf=150,
            l2_regularization=8, early_stopping=False, random_state=42)


def features(prices):
    """Use adjacent observed exchange-session bars, no fill and no fit-dependent operations."""
    blocks = {name: prices.pivot(index="date", columns="ticker", values=name).sort_index()
              for name in ("close", "high", "low", "volume", "in_index")}
    close, members = blocks["close"], blocks["in_index"].eq(True)
    ret = np.log(close).diff()
    market = ret.where(members).mean(axis=1)
    vol = ret.rolling(20).std()
    momentum = np.log(close).diff(20)
    short = np.log(close).diff(5)
    market_trend = market.rolling(20).sum()
    gaps = {
        "stock_return": ret, "stock_5": short, "risk_adj_momentum": momentum / (vol * np.sqrt(20)),
        "relative_20": momentum.sub(market_trend, axis=0), "vol_20": vol,
        "beta_20": ret.rolling(20).cov(market).div(market.rolling(20).var(), axis=0),
        "range_5": np.log(blocks["high"] / blocks["low"]).rolling(5).mean(),
        "volume_relative": np.log(blocks["volume"].where(blocks["volume"].gt(0))).diff(5),
        "trend_distance": close / close.rolling(40).mean() - 1,
        "relative_reversal_5": short.sub(market.rolling(5).sum(), axis=0),
        "rank_5": short.where(members).rank(axis=1, pct=True),
        "rank_vol": vol.where(members).rank(axis=1, pct=True),
    }
    cross = pd.DataFrame({"market_1": market, "market_5": market.rolling(5).sum(), "market_20": market_trend,
                          "market_vol": market.rolling(20).std(), "breadth": ret.gt(0).where(ret.notna() & members)
                          .mean(axis=1), "dispersion": ret.where(members).std(axis=1)})
    out = prices.loc[prices.in_index, ["date", "ticker"]].copy()
    d, t = close.index.get_indexer(out.date), close.columns.get_indexer(out.ticker)
    for k, v in gaps.items():
        out[k] = v.to_numpy()[d, t]
    out = out.join(cross, on="date").replace([np.inf, -np.inf], np.nan)
    return out, list(cross)


def selective(data, threshold):
    taken = data[data.confidence.ge(threshold)]
    up = taken.pred_up
    accuracy = taken.correct
    by_date = taken.groupby("date").correct.agg(["sum", "size"]).reindex(
        pd.DatetimeIndex(data.date.unique()).sort_values(), fill_value=0)
    n_dates = len(by_date)
    starts = np.random.default_rng(42).integers(n_dates, size=(1500, (n_dates + 9) // 10, 1))
    indices = ((starts + np.arange(10)) % n_dates).reshape(1500, -1)[:, :n_dates]
    totals = by_date.to_numpy()[indices].sum(axis=1)
    valid = totals[:, 1] > 0
    interval = np.quantile(totals[valid, 0] / totals[valid, 1], [.025, .975]).tolist() if valid.any() else []

    def proportion(s):
        return float(s.mean()) if len(s) else None

    prior = proportion(taken.y_true)
    return {"threshold": float(threshold), "coverage": len(taken) / len(data), "rows": len(taken),
            "dates": taken.date.nunique(), "tickers": taken.ticker.nunique(), "accuracy": proportion(accuracy),
            "majority": max(prior, 1 - prior) if prior is not None else None, "up_share": proportion(up),
            "up_accuracy": proportion(accuracy[up]), "down_accuracy": proportion(accuracy[~up]),
            "up_rows": int(up.sum()), "down_rows": int((~up).sum()), "block_interval": interval}


def lstm_scores(fitted, panel, dates, horizon):
    values = []
    idx = panel.day_index(dates)
    with torch.no_grad():
        for i in range(0, len(idx), fitted.batch_days):
            b = make_batch(idx[i:i + fitted.batch_days], panel, fitted.features, fitted.graph, horizon)
            logits, _ = fitted.model(b["x"], b["edge_index"], b["edge_type"], b["edge_weight"])
            at = b["at"][b["target"].numpy()]
            values.append(pd.DataFrame({"date": panel.days[at[:, 0]], "ticker": panel.tickers[at[:, 1]],
                                        "score": logits[b["target"]].numpy()}))
    return pd.concat(values, ignore_index=True)


def finite(obj):
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite(v) for v in obj]
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    calendar = pd.DatetimeIndex(sorted(prices.date.unique()))
    base, macro = features(prices)
    base = base.merge(labels, on=["date", "ticker"], validate="one_to_one")
    panel = build_panel(prices, labels, empty, embeddings=False)
    cfg_path = Path("configs/lstm_price.yaml")
    config = load_config(cfg_path)
    config.update(seed=42, embargo_days=5)
    config["train"].update(validation_fraction=50 / 307, epochs=16,
                           preprocessing_scope="weight_fit", record_history=True)
    assert not config["gat"]["edge_types"] and not config["features"]["embeddings"]
    # Isolate this study and refuse overwrite. Prediction keys and scores are persisted for reproducibility.
    pred_dir = ROOT / ("followup_predictions" if args.followup else "predictions")
    pred_dir.mkdir()
    inputs = [table_path(n) for n in ("prices", "labels", "membership")]
    inputs += [cfg_path, ROOT / "protocol.md", Path(__file__)]
    if args.followup:
        inputs.append(ROOT / "followup_protocol.md")
    ledger = ROOT / ("followup_runs.jsonl" if args.followup else "runs.jsonl")
    with ledger.open("x", encoding="utf-8") as stream, threadpool_limits(limits=2):
        def emit(x):
            print(json.dumps(finite(x), allow_nan=False), file=stream, flush=True)

        emit({"development_only": True, "seed": 42, "tree": TREE, "logistic_C": .25,
              "calibration_C": 1, "lstm": config, "source_hashes": {str(p): file_hash(p) for p in inputs}})
        windows = (*WINDOWS, ("2023-01-17", "2023-06-30")) if args.followup else WINDOWS
        for w, (lo, hi) in enumerate(windows):
            assessment = calendar[(calendar >= lo) & (calendar <= hi)]
            pos = calendar.get_loc(assessment[0])
            fold = Fold(w, calendar[pos - 312:pos - 5], assessment)
            fit, val = training_partitions(fold, config)
            assert (len(fit), len(val)) == (252, 50)
            assert calendar.get_loc(val[0]) - calendar.get_loc(fit[-1]) == 6
            assert calendar.get_loc(assessment[0]) - calendar.get_loc(val[-1]) == 6
            for h in ((1,) if args.followup else (1, 5)):
                y, ret = f"y_{h}d", f"fwd_ret_{h}d"
                rows = base[base[y].notna()].rename(columns={y: "y_true", ret: "return"})
                f, v, a = [rows[rows.date.isin(d)].reset_index(drop=True) for d in (fit, val, assessment)]
                f = f.sort_values(["date", "ticker"])
                v = v.sort_values(["date", "ticker"]).reset_index(drop=True)
                a = a.sort_values(["date", "ticker"]).reset_index(drop=True)
                expected = panel.rows(assessment, h)
                np.testing.assert_array_equal(a[["date", "ticker"]], expected[["date", "ticker"]])
                np.testing.assert_array_equal(a.y_true, expected.y)
                names = MODELS
                if args.followup:
                    names = ("relative_45",) if w < 2 else ("relative_45", "boosted_relative",
                                                          "stock_logistic", "price_lstm")
                for name in names:
                    start = time.perf_counter()
                    info = {}
                    if name == "price_lstm":
                        preds, info, fitted = train_fold(panel, None, fold, h, config, return_model=True,
                                                         log=lambda *_: None)
                        raw_val = v.merge(lstm_scores(fitted, panel, val, h), on=["date", "ticker"],
                                          validate="one_to_one").score.to_numpy()
                        np.testing.assert_array_equal(preds[["date", "ticker"]], a[["date", "ticker"]])
                        p = preds.p_up.to_numpy(dtype=float).clip(1e-8, 1 - 1e-8)
                        raw_assess = np.log(p / (1 - p))
                    else:
                        regression = name in ("boosted_relative", "relative_45")
                        cols = macro if name == "market_logistic" else list(base.columns[2:-4])
                        if "logistic" in name:
                            mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                LogisticRegression(C=.25, max_iter=300))
                        else:
                            tree_cls = HistGradientBoostingRegressor if regression else (
                                HistGradientBoostingClassifier
                            )
                            mdl = tree_cls(**(TREE | {"max_iter": 45} if name == "relative_45" else TREE))
                        target = f.y_true
                        if regression:
                            target = f["return"] - f.groupby("date")["return"].transform("mean")
                        mdl.fit(f[cols], target)
                        score = mdl.predict if regression else mdl.decision_function
                        raw_val, raw_assess = score(v[cols]), score(a[cols])
                        info["train_auc"] = within_day_auc(f.assign(p_up=score(f[cols])))
                    calibration = make_pipeline(StandardScaler(), LogisticRegression(C=1))
                    calibration.fit(np.asarray(raw_val).reshape(-1, 1), v.y_true)
                    val_p, p = [calibration.predict_proba(np.asarray(z).reshape(-1, 1))[:, 1]
                                for z in (raw_val, raw_assess)]
                    out = a[["date", "ticker", "y_true", "return", "market_20"]].assign(p_up=p, raw_score=raw_assess,
                            confidence=np.abs(p - .5), pred_up=p >= .5, correct=(p >= .5) == a.y_true.to_numpy())
                    out.to_parquet(pred_dir / f"w{w}_h{h}_{name}.parquet", compression="zstd", index=False)
                    with np.errstate(invalid="ignore"):
                        metrics = classification_metrics(out.y_true.to_numpy(), p)
                    ranks = [g.raw_score.corr(g["return"], method="spearman")
                             for _, g in out.groupby("date") if g.raw_score.nunique() > 1]
                    risks = {}
                    for level in LEVELS:
                        threshold = np.quantile(np.abs(val_p - .5), 1 - level)
                        risks[str(level)] = selective(out, threshold)
                    emit({"window": w, "horizon": h, "model": name, "seconds": time.perf_counter()-start,
                          "fit": [str(fit[0].date()), str(fit[-1].date())],
                          "validation": [str(val[0].date()), str(val[-1].date())], "assessment": [lo, hi],
                          "fit_rows": len(f), "assessment_rows": len(a),
                          "fit_majority_accuracy": float(out.y_true.eq(f.y_true.mean() >= .5).mean()),
                          "majority": float(max(out.y_true.mean(), 1 - out.y_true.mean())),
                          "up_share": float(out.pred_up.mean()), "validation_auc": within_day_auc(v.assign(p_up=val_p)),
                          "within_day_auc": within_day_auc(out), "rank_ic": float(np.nanmean(ranks)) if ranks else None,
                          "brier": float(np.mean((p-out.y_true.to_numpy(dtype=float))**2)),
                          "selective": risks, "metrics": metrics, "training": info})
                    print(w, h, name, metrics["accuracy"], within_day_auc(out), flush=True)


if __name__ == "__main__":
    main()
