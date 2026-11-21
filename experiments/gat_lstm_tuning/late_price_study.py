"""Twelve matched exploratory price fits after the last official prediction date."""
import argparse
import json
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.gat_lstm_tuning.feature_comparison import clean
from experiments.gat_lstm_tuning.run import digest
from src.core.config import load_config
from src.core.io import read_table
from src.core.paths import table_path
from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold
from src.models.ladder_report import auc
from src.models.panel import build_panel
from src.models.price_features import EXTENDED_PRICE
from src.models.train import fold_report, train_fold, training_partitions

OFFICIAL_END = "2023-07-12"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="late_price_study.jsonl")
    args = parser.parse_args()
    if Path(args.output).name != args.output or not args.output.endswith(".jsonl"):
        parser.error("--output must be a new JSONL filename in the experiment directory")
    output = Path(__file__).with_name(args.output)
    if output.exists():
        parser.error("Output already exists")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, labels, empty_news = load_inputs(include_news=False, legacy_labels=True)
    membership = read_table("membership")
    panel = build_panel(prices, labels, empty_news, embeddings=False, extended_price=True)
    calendar = panel.days
    after = calendar[calendar > OFFICIAL_END]
    if len(after) != 119:
        raise ValueError("Expected the original 119-session late-2023 tail")
    assessment = (after[5:57], after[62:-5])
    plan = []
    for i, assess in enumerate(assessment):
        start = calendar.get_loc(assess[0])
        fold = Fold(i, calendar[start - 307:start - 5], assess)
        # Keep the project population and disclose member rows with unavailable targets.
        expected = {(date, row.ticker) for row in membership.itertuples(index=False)
                    for date in assess if date >= row.start and (pd.isna(row.end) or date <= row.end)}
        quoted = prices.loc[prices.date.isin(assess) & prices.in_index, ["date", "ticker"]]
        if set(quoted.itertuples(index=False, name=None)) != expected:
            raise ValueError(f"Recorded membership and quoted member rows disagree for window {i}")
        unavailable = {}
        for horizon in (1, 5):
            observed = panel.rows(assess, horizon)
            unavailable[horizon] = [(str(date.date()), ticker) for date, ticker in
                                    sorted(expected - set(zip(observed.date, observed.ticker, strict=True)))]
        plan.append((fold, unavailable))

    config_path = Path("configs/gat_lstm.yaml")
    config = load_config(config_path)
    config.update(embargo_days=5)
    config["train"].update(epochs=16, validation_fraction=45 / 302,
                           preprocessing_scope="weight_fit", record_history=True)
    config["features"].update(news=[], embeddings=False)
    config["gat"]["edge_types"] = []
    paths = [table_path(name) for name in ("prices", "labels", "membership")]
    paths += [config_path, Path(__file__), *sorted(Path("src").rglob("*.py"))]
    with output.open("x") as stream:
        def emit(row):
            print(json.dumps(clean(row), allow_nan=False), file=stream, flush=True)

        emit({"development_only": True, "last_official_prediction_date": OFFICIAL_END, "config": config,
              "features_added": EXTENDED_PRICE, "seeds_by_horizon": {"1": [42], "5": [42, 43]},
              "runtime": {"device": "cpu", "torch_threads": 2, "torch": str(torch.__version__),
                          "python": platform.python_version()},
              "sha256": {str(path): digest(path) for path in paths}})
        for fold, unavailable in plan:
            fit, val = training_partitions(fold, config)
            assert (len(fit), len(val), len(fold.test)) == (252, 45, 52)
            ranges = {kind: [str(dates[0].date()), str(dates[-1].date()), len(dates)]
                      for kind, dates in (("fit", fit), ("validation", val), ("assessment", fold.test))}
            for h in (1, 5):
                expected = panel.rows(fold.test, h)
                majority_up = bool(panel.rows(fit, h).y.mean() >= 0.5)
                for seed in (42,) if h == 1 else (42, 43):
                    for arm in ("reference", "extended"):
                        cfg = deepcopy(config)
                        cfg.update(seed=seed, model=f"late_price_{arm}")
                        if arm == "extended":
                            cfg["features"]["price"] += list(EXTENDED_PRICE)
                        preds, info = train_fold(panel, None, fold, h, cfg, log=lambda *_: None)
                        assert preds[["date", "ticker"]].equals(expected[["date", "ticker"]])
                        np.testing.assert_array_equal(preds.y_true, expected.y)
                        metrics = fold_report(preds, info)
                        metrics["fit_majority_accuracy"] = float(preds.y_true.eq(majority_up).mean())
                        metrics["assessment_majority_accuracy"] = max(metrics["base_rate"], 1 - metrics["base_rate"])
                        metrics["brier"] = float(np.mean((preds.p_up - preds.y_true) ** 2))
                        metrics["daily_auc"] = {str(d.date()): auc(g.y_true, g.p_up) for d, g in preds.groupby("date")}
                        emit({"arm": arm, "seed": seed, "ranges": ranges, "assessment": metrics,
                              "unavailable_member_targets": unavailable[h], **info})
                        print(f"{fold.index=} {h=} {seed=} {arm=} AUC {metrics['within_day_auc']:.5f}, "
                              f"{info['best_step']}/{info['steps']} steps, {info['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
