"""Paired fold-level MCC comparison for Task 4.9 (Issue #44).

The endpoint is the unweighted mean of five held-out fold MCC differences,
GAT-LSTM minus Baseline 6. This is separate from the model report's headline
within-day AUC. Thresholds come exclusively from validation run records.

The primary two-sided sign-flip test enumerates all 32 assignments. Its exact
p-value requires joint invariance under independent fold sign flips (for example,
independent symmetric contrasts under the null); equal means alone is insufficient.
Overlapping training windows, market regimes and overlapping five-day outcomes
can violate that assumption. Treat inference as conditional and exploratory.

The secondary 95% percentile interval enumerates all 5**5 paired whole-fold
bootstrap samples. It never resamples individual ticker-days. With five potentially
dependent folds its coverage can be poor, even degenerate; it must not override the
primary test. Holm adjusts the two horizon p-values at family alpha 0.05. Five folds
have minimum two-sided p = 2/32 = 0.0625, so cannot reject at this level.

Undefined MCC is reported as null; no fold is discarded and that horizon's inference
is withheld. Both horizons must be estimable before familywise conclusions are made.
No fitting, threshold selection, random sampling or artifact writing occurs here.
Prediction files cannot independently prove upstream membership or news causality.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.core.io import read_table
from src.core.paths import predictions_path
from src.core.schema import PREDICTIONS, validate
from src.eval.metrics import classification_metrics

MODELS = ("gat_lstm", "lstm_news")
HORIZONS = (1, 5)
FOLDS = tuple(range(5))
KEY = ["fold", "horizon", "date", "ticker"]


def paired_fold_inference(differences: list[float]) -> dict:
    """Enumerate sign flips and paired fold bootstrap; accept 2–5 finite contrasts.

    Discrete bootstrap endpoints use the empirical inverse CDF. Enumeration makes
    results reproducible without a random seed. Zero contrasts retain their signs.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not 2 <= len(values) <= 5 or not np.isfinite(values).all():
        raise ValueError("expected 2–5 finite fold differences")

    count = len(values)
    mean = float(values.mean())
    signs = np.array(list(itertools.product((-1, 1), repeat=count)))
    null_means = (signs * values).mean(axis=1)
    tolerance = 1e-12 * max(1.0, abs(mean))
    p_value = float(np.mean(np.abs(null_means) >= abs(mean) - tolerance))

    samples = np.array(list(itertools.product(range(count), repeat=count)))
    bootstrap_means = values[samples].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975], method="inverted_cdf")
    return {
        "mean_difference": mean,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": p_value,
        "sign_assignments": len(signs),
        "bootstrap_samples": len(samples),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment, returned in the original horizon order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("p-values must be a nonempty finite vector in [0, 1]")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be in [0, 1]")

    adjusted = np.empty(len(values))
    previous = 0.0
    for rank, index in enumerate(np.argsort(values)):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted.tolist()


def _prepare_predictions(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Validate before any casting, then order the complete official row set."""
    validate(frame, PREDICTIONS)
    if frame.empty or set(frame["model"]) != {model}:
        raise ValueError(f"expected nonempty predictions for {model}")
    if not frame["y_true"].isin([0, 1]).all():
        raise ValueError("y_true must contain binary labels")
    if not np.isfinite(frame["p_up"]).all():
        raise ValueError("p_up must contain finite probabilities")

    coverage = set(frame[["horizon", "fold"]].itertuples(index=False, name=None))
    if coverage != set(itertools.product(HORIZONS, FOLDS)):
        raise ValueError(f"{model}: require horizons 1 and 5, each with folds 0–4")

    for horizon in HORIZONS:
        previous_end = None
        for fold in FOLDS:
            dates = frame.loc[(frame.horizon == horizon) & (frame.fold == fold), "date"]
            if previous_end is not None and dates.min() <= previous_end:
                raise ValueError("test folds must be chronological and nonoverlapping")
            previous_end = dates.max()
    return frame.sort_values(KEY).reset_index(drop=True)


def _thresholds(record: dict, model: str, frame: pd.DataFrame) -> dict:
    """Read fixed validation thresholds; reject missing or ambiguous run entries."""
    if not isinstance(record, dict) or record.get("model") != model:
        raise ValueError(f"run record must identify {model}")
    entries = record.get("folds")
    if not isinstance(entries, list):
        raise ValueError("run record requires a folds list")

    thresholds = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("run fold entries must be objects")
        horizon, fold = entry.get("horizon"), entry.get("fold")
        if type(horizon) is not int or type(fold) is not int:
            raise ValueError("run horizon and fold must be integers")
        key = (horizon, fold)
        if key in thresholds:
            raise ValueError("duplicate run fold/horizon")
        threshold = entry.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not 0 <= threshold <= 1:
            raise ValueError("validation threshold must be finite and in [0, 1]")
        rows = frame.loc[(frame.horizon == horizon) & (frame.fold == fold)]
        if "test_rows" in entry and entry["test_rows"] != len(rows):
            raise ValueError("run test_rows does not match predictions")
        thresholds[key] = float(threshold)

    if set(thresholds) != set(itertools.product(HORIZONS, FOLDS)):
        raise ValueError("run record requires both horizons and all five folds")
    return thresholds


def _fold_scores(proposed: pd.DataFrame, baseline: pd.DataFrame, thresholds: list[dict], horizon: int) -> list:
    scores = []
    for fold in FOLDS:
        selected = (proposed.horizon == horizon) & (proposed.fold == fold)
        rows = proposed.loc[selected]
        mcc = []
        for frame, operating_points in zip((proposed, baseline), thresholds, strict=True):
            metrics = classification_metrics(
                frame.loc[selected, "y_true"].to_numpy(),
                frame.loc[selected, "p_up"].to_numpy(),
                threshold=operating_points[horizon, fold],
            )
            value = metrics["mcc"]
            mcc.append(float(value) if np.isfinite(value) else None)
        difference = None if None in mcc else mcc[0] - mcc[1]
        scores.append(
            {
                "fold": fold,
                "n_observations": len(rows),
                "first_date": rows.date.min().date().isoformat(),
                "last_date": rows.date.max().date().isoformat(),
                "mcc_gat_lstm": mcc[0],
                "mcc_lstm_news": mcc[1],
                "difference": difference,
                "threshold_gat_lstm": thresholds[0][horizon, fold],
                "threshold_lstm_news": thresholds[1][horizon, fold],
            }
        )
    return scores


def compare_predictions(proposed: pd.DataFrame, baseline: pd.DataFrame, proposed_run: dict, baseline_run: dict) -> dict:
    """Compare complete five-fold artifacts, without silently dropping any rows.

    Run records attest threshold provenance; their contents alone cannot prove that
    validation selection was causal or that the two runs used matched settings.
    Inspect recorded configs and upstream split provenance before reporting results.
    """
    proposed = _prepare_predictions(proposed, MODELS[0])
    baseline = _prepare_predictions(baseline, MODELS[1])
    if not proposed[KEY].equals(baseline[KEY]):
        raise ValueError("prediction keys do not match exactly")
    if not proposed.y_true.equals(baseline.y_true):
        raise ValueError("conflicting realised labels")
    thresholds = [
        _thresholds(record, model, frame)
        for record, model, frame in zip((proposed_run, baseline_run), MODELS, (proposed, baseline), strict=True)
    ]

    results = []
    for horizon in HORIZONS:
        scores = _fold_scores(proposed, baseline, thresholds, horizon)
        differences = [row["difference"] for row in scores]
        inference = None if None in differences else paired_fold_inference(differences)
        results.append(
            {
                "horizon": horizon,
                "n_folds": len(scores),
                "n_observations": sum(row["n_observations"] for row in scores),
                "folds": scores,
                "inference": inference,
                "status": "estimable" if inference else "undefined_mcc",
                "p_value_holm": None,
                "reject_under_sign_flip_assumptions": None,
            }
        )

    if all(result["inference"] is not None for result in results):
        adjusted = holm_adjust([result["inference"]["p_value"] for result in results])
        for result, p_value in zip(results, adjusted, strict=True):
            result["p_value_holm"] = p_value
            result["reject_under_sign_flip_assumptions"] = p_value <= 0.05
            result["interpretation"] = (
                "Conditional evidence of a difference; check direction and dependence assumptions."
                if p_value <= 0.05
                else "No rejection at family alpha 0.05; this does not establish equivalence."
            )
    else:
        for result in results:
            result["interpretation"] = "Family inference withheld: at least one horizon has undefined fold MCC."

    return {
        "metric": "mcc",
        "contrast": "gat_lstm minus lstm_news; unweighted mean across folds",
        "primary_test": "exact two-sided fold sign-flip; conditional on joint sign invariance",
        "secondary_interval": "95% paired whole-fold percentile bootstrap; descriptive, coverage uncertain",
        "multiplicity": "Holm across horizons 1 and 5",
        "family_alpha": 0.05,
        "limitations": [
            "Five folds: minimum two-sided sign-flip p-value 0.0625; limited power.",
            "Overlapping training windows and five-day outcomes can violate fold independence/sign invariance.",
            "MCC is a separate endpoint from headline within-day AUC; no test of that headline is made.",
            "Threshold provenance, causal features and historical membership require upstream audit.",
        ],
        "runs": {
            model: {
                field: record.get(field)
                for field in ("model", "config", "ablate", "git_commit", "git_dirty", "written_at_utc")
            }
            for model, record in zip(MODELS, (proposed_run, baseline_run), strict=True)
        },
        "horizons": results,
    }


def load_comparison(proposed_path: Path, baseline_path: Path) -> dict:
    """Read official artifacts and their associated run records, without writing."""
    paths = [Path(proposed_path), Path(baseline_path)]
    run_paths = [path.with_name(f"{path.stem}_run.json") for path in paths]
    missing = [str(path) for path in paths + run_paths if not path.is_file()]
    if missing:
        raise ValueError("missing required artifacts: " + ", ".join(missing))

    frames = []
    for path in paths:
        if set(pq.read_schema(path).names) != set(PREDICTIONS.column_names):
            raise ValueError(f"{path}: columns differ from official prediction schema")
        # The projected I/O path preserves raw dtypes; validate before any lossy casts.
        frames.append(read_table("predictions", path, columns=PREDICTIONS.column_names))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in run_paths]
    result = compare_predictions(*frames, *records)
    result["inputs"] = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths + run_paths
    ]
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paired fold-level MCC significance: GAT-LSTM vs Baseline 6")
    parser.add_argument("--proposed", type=Path, default=predictions_path("gat_lstm"))
    parser.add_argument("--baseline", type=Path, default=predictions_path("lstm_news"))
    args = parser.parse_args(argv)
    try:
        result = load_comparison(args.proposed, args.baseline)
    except (ValueError, OSError) as error:
        parser.exit(1, f"significance: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
