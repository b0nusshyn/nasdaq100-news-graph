"""Run up to twelve exploratory matched fits once full candidate encoding is approved."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gat_lstm_tuning.feature_comparison import clean
from src.core.config import load_config
from src.core.paths import table_path
from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold
from src.models.ladder_report import within_day_auc
from src.models.panel import build_panel
from src.models.train import fold_report, train_fold, training_partitions
from src.nlp.external_news_candidate import POLICIES
from src.nlp.rebuild_timed_news import file_hash

WINDOWS = (("2023-07-20", "2023-10-02"), ("2023-10-10", "2023-12-21"))
ARMS = ("price_only", *POLICIES)


def fixed_folds(calendar):
    for i, (lo, hi) in enumerate(WINDOWS):
        assessment = calendar[calendar.to_series().between(lo, hi).to_numpy()]
        pos = calendar.get_loc(assessment[0])
        if len(assessment) != 52 or pos < 307 or calendar.get_loc(assessment[-1]) + 5 >= len(calendar):
            raise ValueError("Development calendar changed")
        yield Fold(i, calendar[pos - 307:pos - 5], assessment)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoded-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--windows", nargs="+", type=int, choices=[0, 1], default=[0])
    p.add_argument("--horizons", nargs="+", choices=[1, 5], type=int, default=[1, 5])
    args = p.parse_args()
    filename_seed = re.search(r"(?:^|_)seed(\d+)(?:_|$)", args.output.stem)
    if filename_seed and int(filename_seed[1]) != args.seed:
        p.error("Output filename seed disagrees with --seed")
    root = args.encoded_dir
    manifest = root / "manifest.json"
    if args.output.exists():
        p.error("Output already exists")
    if not manifest.is_file() or not json.loads(manifest.read_text()).get("full"):
        p.error("A full encoded candidate is required. Smoke outputs cannot be used for training.")
    news_paths = {policy: root / f"{policy}_news.parquet" for policy in POLICIES}

    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    reference = build_panel(prices, labels, empty, embeddings=False)
    panels = {"price_only": reference}
    coverage = {}
    population = prices.loc[prices.in_index & prices.date.between("2019-01-01", "2023-12-31"), ["date", "ticker"]]
    expected_keys = pd.MultiIndex.from_frame(population)
    for policy, path in news_paths.items():
        data = pd.read_parquet(path)
        keys = pd.MultiIndex.from_frame(data[["date", "ticker"]])
        if not keys.is_unique or len(keys) != len(expected_keys) or not keys.isin(expected_keys).all():
            raise ValueError("Candidate news differs from the historical in-index ticker-day population")
        coverage[policy] = data[["date", "ticker", "has_news", "missing_source_count"]]
        panels[policy] = build_panel(prices, labels, data)
    config_path = Path("configs/gat_lstm.yaml")
    config = load_config(config_path)
    config.update(seed=args.seed, embargo_days=5)
    config["gat"]["edge_types"] = []
    config["train"].update(epochs=16, validation_fraction=45 / 302,
                           preprocessing_scope="weight_fit", record_history=True)
    predictions_dir = args.output.with_suffix(".predictions")
    if predictions_dir.exists():
        p.error("Prediction output directory already exists")
    if shutil.disk_usage(args.output.parent).free < 2 * 1024**3 + 16 * 1024**2:
        p.error("Insufficient free disk space above the 2 GiB safety margin")
    predictions_dir.mkdir()
    with args.output.open("x", encoding="utf-8") as stream:
        def emit(row):
            print(json.dumps(clean(row), allow_nan=False), file=stream, flush=True)

        hashed = [manifest, *news_paths.values(), config_path, Path(__file__)]
        hashed += [table_path(name) for name in ("prices", "labels", "membership")]
        hashed += sorted(Path("src/models").glob("*.py"))
        emit({"development_only": True, "config": config, "windows": args.windows, "horizons": args.horizons,
              "files_sha256": {str(path): file_hash(path) for path in hashed}})
        for fold in fixed_folds(reference.days):
            if fold.index not in args.windows:
                continue
            fit, val = training_partitions(fold, config)
            assert reference.days.get_loc(val[0]) - reference.days.get_loc(fit[-1]) == 6
            assert reference.days.get_loc(fold.test[0]) - reference.days.get_loc(val[-1]) == 6
            ranges = {role: [str(dates[0].date()), str(dates[-1].date()), len(dates)]
                      for role, dates in (("fit", fit), ("validation", val), ("assessment", fold.test))}
            for horizon in sorted(set(args.horizons)):
                expected = reference.rows(fold.test, horizon)
                majority_up = reference.rows(fit, horizon).y.mean() >= 0.5
                for arm in ARMS:
                    cfg = deepcopy(config)
                    cfg["model"] = f"candidate_{arm}"
                    if arm == "price_only":
                        cfg["features"].update(news=[], embeddings=False)
                    preds, info = train_fold(panels[arm], None, fold, horizon, cfg, log=lambda *_: None)
                    assert preds[["date", "ticker"]].equals(expected[["date", "ticker"]])
                    np.testing.assert_array_equal(preds.y_true, expected.y)
                    metrics = fold_report(preds, info)
                    metrics["fit_majority_accuracy"] = float(preds.y_true.eq(majority_up).mean())
                    metrics["assessment_majority_accuracy"] = max(metrics["base_rate"], 1 - metrics["base_rate"])
                    metrics["brier"] = float(np.mean((preds.p_up.astype(float) - preds.y_true) ** 2))
                    metrics["score_distribution"] = {
                        "mean": float(preds.p_up.mean()), "sd": float(preds.p_up.std()),
                        "p05": float(preds.p_up.quantile(.05)), "p95": float(preds.p_up.quantile(.95)),
                    }
                    strata = {}
                    for policy, table in coverage.items():
                        joined = preds.merge(table, on=["date", "ticker"], validate="one_to_one")
                        assert len(joined) == len(preds)
                        strata[policy] = {
                            f"news={has_news},missing_source={lost}": {
                                "rows": len(group), "within_day_auc": within_day_auc(group),
                            }
                            for (has_news, lost), group in joined.groupby(
                                ["has_news", joined.missing_source_count.gt(0)]
                            )
                        }
                    prediction_path = predictions_dir / f"w{fold.index}_h{horizon}_{arm}.parquet"
                    if shutil.disk_usage(predictions_dir).free < 2 * 1024**3 + 1024**2:
                        raise RuntimeError("Insufficient disk space above the 2 GiB safety margin")
                    with prediction_path.open("xb") as dest:
                        preds.to_parquet(dest, compression="zstd", index=False)
                    emit({"arm": arm, "seed": args.seed, "ranges": ranges, "assessment": metrics,
                          "coverage_strata": strata,
                          "predictions": str(prediction_path),
                          "predictions_sha256": file_hash(prediction_path), **info})
                    print(f"seed {args.seed} window {fold.index} h{horizon} {arm}: "
                          f"AUC {metrics['within_day_auc']:.5f}, {info['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
