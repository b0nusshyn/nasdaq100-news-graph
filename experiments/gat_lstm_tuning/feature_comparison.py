"""Run the shared weight-fit pipeline on the two retrospective development windows."""
import argparse
import json
import math
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric

from experiments.gat_lstm_tuning.run import digest
from src.core.config import load_config
from src.core.io import read_table
from src.core.paths import table_path
from src.eval.run_baselines import load_inputs
from src.eval.splitter import walk_forward
from src.models.ladder_report import auc
from src.models.panel import build_panel
from src.models.train import GraphSources, fold_report, train_fold, training_partitions

CONFIG = Path("configs/gat_lstm.yaml")
ARMS = {
    "price": (False, []),
    "news": (True, []),
    "graph": (True, ["comention", "corr"]),
}
CONFIRMATION_ARMS = {**ARMS, "price_corr": (False, ["corr"])}


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="matched_feature_comparison.jsonl",
                        help="new JSONL filename in this experiment directory")
    parser.add_argument("--confirm-5d", action="store_true", help="compare four fixed arms on development window D2")
    args = parser.parse_args()
    if Path(args.output).name != args.output or not args.output.endswith(".jsonl"):
        parser.error("--output must be a JSONL filename in this experiment directory")
    output = Path(__file__).with_name(args.output)
    if output.exists():
        parser.error(f"{output.name} already exists")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, labels, news = load_inputs(include_news=True, legacy_labels=True)
    links = read_table("article_tickers")
    membership = read_table("membership")
    prices = prices[prices.date <= "2020-12-31"]
    calendar = pd.DatetimeIndex(sorted(prices.date.unique()))
    joined = prices[["date", "ticker", "in_index"]].reset_index(drop=True).reset_index().merge(membership, on="ticker")
    active = joined.date.ge(joined.start) & (joined.end.isna() | joined.date.le(joined.end))
    flags = active.groupby(joined["index"]).any().reindex(range(len(prices)), fill_value=False)
    if flags.tolist() != prices.in_index.tolist():
        raise ValueError("Membership differs from recorded price flags")

    next_session = pd.Series(calendar[1:], index=calendar[:-1])

    def lag(frame):
        return frame.loc[frame.date.isin(next_session.index)].assign(date=lambda f: f.date.map(next_session))

    panel = build_panel(prices, labels[labels.date <= "2020-12-31"], lag(news))
    sources = GraphSources(prices, lag(links))
    days = calendar[calendar >= "2019-01-01"]
    folds = walk_forward(days, train_days=302, test_days=55, step_days=65, horizon=5, embargo_days=5)
    selected_folds, horizons, arms = (folds[2:3], (5,), CONFIRMATION_ARMS) if args.confirm_5d else (
        folds[:2], (1, 5), ARMS,
    )
    if len(selected_folds) != (1 if args.confirm_5d else 2):
        raise ValueError("Missing required chronological development windows")
    config = load_config(CONFIG)
    config["embargo_days"] = 5
    config["train"].update(epochs=16, validation_fraction=45 / 302,
                           preprocessing_scope="weight_fit", record_history=True)
    paths = [table_path(n) for n in ("prices", "labels", "membership", "news_features", "article_tickers")]
    paths += [CONFIG, Path(__file__), *sorted(Path("src").rglob("*.py"))]
    with output.open("x") as stream:
        def emit(row):
            print(json.dumps(clean(row), allow_nan=False), file=stream, flush=True)

        emit({"development_only": True, "config": config, "arms": arms, "news_lag_sessions": 1, "seeds": [42, 43],
              "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "pyg": torch_geometric.__version__, "device": "cpu", "torch_threads": 2},
              "sha256": {str(path): digest(path) for path in paths}})
        for fold in selected_folds:
            fit, val = training_partitions(fold, config)
            assert (len(fit), len(val), len(fold.test)) == (252, 45, 55)
            for before, after in [(fit, val), (val, fold.test)]:
                assert calendar.get_loc(after[0]) - calendar.get_loc(before[-1]) == 6
            assert calendar.get_loc(fold.test[-1]) + 5 < len(calendar)
            ranges = {key: [str(d[0].date()), str(d[-1].date()), len(d)]
                      for key, d in (("fit", fit), ("validation", val), ("assessment", fold.test))}
            for horizon in horizons:
                expected = panel.rows(fold.test, horizon).sort_values(["date", "ticker"]).reset_index(drop=True)
                majority_up = bool(panel.rows(fit, horizon)["y"].mean() >= 0.5)
                for seed in (42, 43):
                    for arm, (has_news, edge_types) in arms.items():
                        cfg = deepcopy(config)
                        cfg.update(seed=seed, model=f"development_{arm}")
                        cfg["gat"]["edge_types"] = edge_types
                        if not has_news:
                            cfg["features"].update(news=[], embeddings=False)
                        preds, info = train_fold(panel, sources, fold, horizon, cfg, log=lambda *_: None)
                        np.testing.assert_array_equal(preds.date, expected.date)
                        np.testing.assert_array_equal(preds.ticker, expected.ticker)
                        np.testing.assert_array_equal(preds.y_true, expected.y)
                        metrics = fold_report(preds, info)
                        metrics["fit_majority_accuracy"] = float(preds.y_true.eq(majority_up).mean())
                        metrics["assessment_majority_accuracy"] = max(metrics["base_rate"], 1 - metrics["base_rate"])
                        metrics["brier"] = float(np.mean((preds.p_up - preds.y_true) ** 2))
                        metrics["daily_auc"] = {str(d.date()): auc(g.y_true, g.p_up)
                                                for d, g in preds.groupby("date")}
                        emit({"arm": arm, "seed": seed, "ranges": ranges, "assessment": metrics, **info})
                        print(f"D{fold.index} {horizon}d seed {seed} {arm}: "
                              f"{info['best_step']}/{info['steps']}, AUC {metrics['within_day_auc']:.5f}, "
                              f"{info['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
