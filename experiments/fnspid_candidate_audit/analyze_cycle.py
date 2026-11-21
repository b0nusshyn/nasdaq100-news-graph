"""Summarize matched predictions, with paired date-block uncertainty for exploratory contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.ladder_report import auc
from src.nlp.external_news_candidate import POLICIES
from src.nlp.rebuild_timed_news import file_hash


def paired_days(candidate, baseline, mask=None):
    keys = ["date", "ticker", "y_true"]
    for key in keys:
        np.testing.assert_array_equal(candidate[key], baseline[key])
    data = candidate[keys].assign(news=candidate.p_up, price=baseline.p_up)
    if mask is not None:
        data = data.loc[mask.to_numpy()]
    return {
        str(date.date()): float(auc(g.y_true, g.news) - auc(g.y_true, g.price))
        for date, g in data.groupby("date") if g.y_true.nunique() == 2
    }


def block_interval(by_window, rng):
    trials = []
    for vals in by_window:
        n = len(vals)
        starts = rng.integers(n, size=(4000, (n + 9) // 10, 1))
        idx = ((starts + np.arange(10)) % n).reshape(4000, -1)[:, :n]
        trials.append(np.asarray(vals)[idx].mean(axis=1))
    return np.quantile(np.mean(trials, axis=0), [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledgers", nargs="+", type=Path, required=True)
    parser.add_argument("--encoded-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    grouped = {}
    for path in args.ledgers:
        header, *lines = path.read_text().splitlines()
        seed_from_config = json.loads(header)["config"]["seed"]
        for line in lines:
            row = json.loads(line)
            if row.setdefault("seed", seed_from_config) != seed_from_config:
                raise ValueError(f"Run seed disagrees with manifest {path}")
            key = (row["fold"], row["horizon"], row["seed"], row["arm"])
            if key in grouped:
                raise ValueError(f"Duplicate training run {key}")
            pred_path = Path(row["predictions"])
            if file_hash(pred_path) != row["predictions_sha256"]:
                raise ValueError(f"Prediction hash mismatch {pred_path}")
            grouped[key] = (row, pd.read_parquet(pred_path))
    results = {"fits": len(grouped), "contrasts": [], "means": [], "paired_block_intervals": []}
    rng = np.random.default_rng(441)
    for h in (1, 5):
        for arm in ("price_only", *POLICIES):
            subset = [row for (f, horizon, seed, a), (row, _) in grouped.items() if horizon == h and a == arm]
            if not subset:
                continue
            def mean(values):
                numbers = [v for v in values if v is not None]
                return float(np.mean(numbers)) if numbers else None
            results["means"].append({
                "horizon": h, "arm": arm, "fits": len(subset),
                **{k: mean([r["assessment"][k] for r in subset]) for k in (
                    "within_day_auc", "accuracy", "mcc", "f1", "fit_majority_accuracy",
                    "assessment_majority_accuracy", "brier",
                )},
                "defined_mcc": sum(r["assessment"]["mcc"] is not None for r in subset),
                "validation_auc": mean([r["best_val_within_day_auc"] for r in subset]),
                "training_probe_auc": mean([
                    next(v["train_probe_within_day_auc"] for v in r["history"] if v["step"] == r["best_step"])
                    for r in subset
                ]),
            })
        for policy in POLICIES:
            coverage = pd.read_parquet(args.encoded_dir / f"{policy}_news.parquet",
                                       columns=["date", "ticker", "has_news", "missing_source_count"])
            by_window = {}
            for (fold, horizon, seed, arm), (row, pred) in grouped.items():
                if h != horizon or arm != policy:
                    continue
                ref, reference = grouped[fold, h, seed, "price_only"]
                c = pred[["date", "ticker"]].merge(coverage, on=["date", "ticker"], validate="one_to_one")
                assert len(c) == len(pred)
                values = paired_days(pred, reference)
                by_window.setdefault(fold, []).append(pd.Series(values))
                strata = {}
                for name, mask in {
                    "candidate_news": c.has_news, "no_candidate_news": ~c.has_news,
                    "missing_source": c.missing_source_count.gt(0),
                    "no_known_missing_source": c.missing_source_count.eq(0),
                }.items():
                    days = paired_days(pred, reference, mask)
                    strata[name] = {"rows": int(mask.sum()), "days": len(days),
                                    "auc_delta": float(np.mean(list(days.values()))) if days else None}
                results["contrasts"].append({"horizon": h, "arm": arm, "fold": fold, "seed": seed,
                                             "auc_delta": row["assessment"]["within_day_auc"]
                                             - ref["assessment"]["within_day_auc"], "strata": strata})
            if by_window:
                days = [pd.concat(seeds, axis=1).mean(axis=1).sort_index().to_list() for seeds in by_window.values()]
                results["paired_block_intervals"].append({
                    "horizon": h, "arm": policy, "window_mean_auc_deltas": [float(np.mean(v)) for v in days],
                    "interval": block_interval(days, rng), "block_sessions": 10,
                })
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(results["means"], indent=2))


if __name__ == "__main__":
    main()
