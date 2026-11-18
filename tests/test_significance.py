"""Task 4.9: exact paired calculations, strict artifacts and read-only CLI."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.core.io import write_table
from src.core.schema import conform
from src.eval.significance import compare_predictions, holm_adjust, load_comparison, paired_fold_inference


@pytest.fixture
def artifacts():
    frames = []
    records = []
    for model in ("gat_lstm", "lstm_news"):
        rows = []
        entries = []
        for horizon in (1, 5):
            for fold in range(5):
                for index, label in enumerate((0, 0, 1, 1)):
                    rows.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            "fold": fold,
                            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=10 * fold + index // 2),
                            "ticker": ("AAA", "BBB")[index % 2],
                            "p_up": (0.1, 0.2, 0.8, 0.9)[index],
                            "y_true": label,
                        }
                    )
                entries.append({"horizon": horizon, "fold": fold, "threshold": 0.5, "test_rows": 4})
        frames.append(conform(pd.DataFrame(rows), "predictions"))
        records.append({"model": model, "folds": entries, "config": {"seed": 42}})
    return (*frames, *records)


def test_identical_models_keep_pairing(artifacts):
    report = compare_predictions(*artifacts)
    for horizon in report["horizons"]:
        assert horizon["n_folds"] == 5
        assert horizon["n_observations"] == 20
        assert [row["difference"] for row in horizon["folds"]] == [0] * 5
        assert horizon["inference"] == {
            "mean_difference": 0,
            "ci_low": 0,
            "ci_high": 0,
            "p_value": 1,
            "sign_assignments": 32,
            "bootstrap_samples": 3125,
        }
        assert horizon["p_value_holm"] == 1
        assert not horizon["reject_under_sign_flip_assumptions"]


@pytest.mark.parametrize("better", [True, False])
def test_direction_and_coarse_p_value(artifacts, better):
    proposed, baseline, *_ = artifacts
    # Confusion matrix TP=TN=FP=FN=1 gives MCC=0, versus perfect MCC=1.
    changed = baseline if better else proposed
    changed["p_up"] = np.tile([0.1, 0.8, 0.4, 0.9], 10).astype("float32")
    result = compare_predictions(*artifacts)["horizons"][0]
    expected = 1 if better else -1
    assert result["inference"]["mean_difference"] == expected
    assert result["inference"]["ci_low"] == expected
    assert result["inference"]["ci_high"] == expected
    # Only the all-positive and all-negative assignments are as extreme: 2/32.
    assert result["inference"]["p_value"] == 0.0625
    assert result["p_value_holm"] == 0.125
    assert not result["reject_under_sign_flip_assumptions"]


@pytest.mark.parametrize("differences,mean,p_value", [([1, 2, 3], 2, 0.25), ([1, -2, 3], 2 / 3, 0.75), ([0, 2], 1, 1)])
def test_hand_calculated_sign_flips(differences, mean, p_value):
    result = paired_fold_inference(differences)
    assert result["mean_difference"] == pytest.approx(mean)
    assert result["p_value"] == p_value


def test_hand_calculated_paired_bootstrap_and_reproducibility():
    # Four ordered draws of the two paired contrasts give means [0, 1, 1, 2].
    first = paired_fold_inference([0, 2])
    assert first["bootstrap_samples"] == 4
    assert (first["ci_low"], first["ci_high"]) == (0, 2)
    assert paired_fold_inference([0, 2]) == first
    reverse = paired_fold_inference([0, -2])
    assert reverse["p_value"] == first["p_value"]
    assert reverse["ci_low"] == -first["ci_high"]
    assert reverse["ci_high"] == -first["ci_low"]


@pytest.mark.parametrize("values", [[], [1], [1] * 6, [1, np.nan], [[1, 2]], [1, np.inf]])
def test_inference_rejects_invalid_samples(values):
    with pytest.raises(ValueError, match="finite fold differences"):
        paired_fold_inference(values)


def test_holm_hand_calculation():
    assert holm_adjust([0.01, 0.04]) == [0.02, 0.04]
    assert holm_adjust([0.04, 0.03]) == [0.06, 0.06]


@pytest.mark.parametrize("values", [[], [np.nan], [-0.1], [1.1]])
def test_holm_invalid(values):
    with pytest.raises(ValueError):
        holm_adjust(values)


def test_order_does_not_matter(artifacts):
    expected = compare_predictions(*artifacts)
    proposed, baseline, proposed_run, baseline_run = copy.deepcopy(artifacts)
    proposed_run["folds"].reverse()
    baseline_run["folds"].reverse()
    actual = compare_predictions(
        proposed.sample(frac=1, random_state=1),
        baseline.sample(frac=1, random_state=2),
        proposed_run,
        baseline_run,
    )
    # Raw provenance ordering is immaterial; all calculated results are identical.
    assert actual["horizons"] == expected["horizons"]


def test_missing_row_is_not_silently_dropped(artifacts):
    proposed, baseline, *records = artifacts
    with pytest.raises(ValueError, match="keys do not match"):
        compare_predictions(proposed, baseline.iloc[1:], *records)


def test_duplicate_key(artifacts):
    proposed, baseline, *records = artifacts
    duplicated = pd.concat([proposed, proposed.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        compare_predictions(duplicated, baseline, *records)


def test_conflicting_labels(artifacts):
    artifacts[1].loc[0, "y_true"] = 1
    with pytest.raises(ValueError, match="conflicting realised labels"):
        compare_predictions(*artifacts)


@pytest.mark.parametrize(
    "column,value", [("y_true", 2), ("p_up", np.nan), ("p_up", np.inf), ("p_up", -0.1), ("p_up", 1.1)]
)
def test_invalid_prediction_values(artifacts, column, value):
    artifacts[0].loc[0, column] = value
    with pytest.raises(ValueError):
        compare_predictions(*artifacts)


@pytest.mark.parametrize("selection", ["horizon", "fold", "empty"])
def test_missing_coverage_even_when_both_models_match(artifacts, selection):
    proposed, baseline, *records = artifacts
    keep = proposed.horizon == 1 if selection == "horizon" else proposed.fold != 4
    if selection == "empty":
        keep[:] = False
    with pytest.raises(ValueError):
        compare_predictions(proposed.loc[keep], baseline.loc[keep], *records)


def test_fold_assignment_must_match(artifacts):
    artifacts[1].loc[0, "fold"] = 1
    with pytest.raises(ValueError):
        compare_predictions(*artifacts)


def test_nonchronological_folds(artifacts):
    for frame in artifacts[:2]:
        frame.loc[frame.fold == 4, "date"] -= pd.Timedelta(days=100)
    with pytest.raises(ValueError, match="chronological"):
        compare_predictions(*artifacts)


def test_model_specific_validation_thresholds(artifacts):
    for frame in artifacts[:2]:
        frame["p_up"] = np.tile([0.1, 0.4, 0.6, 0.9], 10).astype("float32")
    for entry in artifacts[2]["folds"]:
        entry["threshold"] = 0.85
    report = compare_predictions(*artifacts)
    row = report["horizons"][0]["folds"][0]
    assert row["threshold_gat_lstm"] == 0.85
    assert row["threshold_lstm_news"] == 0.5
    assert row["mcc_gat_lstm"] == pytest.approx(1 / np.sqrt(3))
    assert row["mcc_lstm_news"] == 1
    assert row["difference"] == pytest.approx(1 / np.sqrt(3) - 1)


@pytest.mark.parametrize("threshold", [None, True, np.nan, np.inf, -0.1, 1.1, "0.5"])
def test_invalid_threshold(artifacts, threshold):
    artifacts[2]["folds"][0]["threshold"] = threshold
    with pytest.raises(ValueError, match="validation threshold"):
        compare_predictions(*artifacts)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_model", "stale_rows", "missing_threshold"])
def test_invalid_run_record(artifacts, problem):
    record = artifacts[2]
    if problem == "missing":
        record["folds"].pop()
    elif problem == "duplicate":
        record["folds"].append(record["folds"][0])
    elif problem == "wrong_model":
        record["model"] = "other"
    elif problem == "stale_rows":
        record["folds"][0]["test_rows"] = 999
    else:
        del record["folds"][0]["threshold"]
    with pytest.raises(ValueError):
        compare_predictions(*artifacts)


@pytest.mark.parametrize("constant", ["labels", "predictions"])
def test_undefined_mcc_keeps_folds_and_withholds_family(artifacts, constant):
    for frame in artifacts[:2]:
        selected = (frame.fold == 0) & (frame.horizon == 1)
        frame.loc[selected, "y_true" if constant == "labels" else "p_up"] = 0
    with pytest.warns(UserWarning, match="undefined"):
        report = compare_predictions(*artifacts)
    first, second = report["horizons"]
    assert first["status"] == "undefined_mcc"
    assert first["inference"] is None
    assert len(first["folds"]) == 5
    assert second["inference"] is not None
    assert all(result["p_value_holm"] is None for result in report["horizons"])
    json.dumps(report, allow_nan=False)


def test_test_outcomes_do_not_select_thresholds(artifacts):
    before = compare_predictions(*artifacts)
    for frame in artifacts[:2]:
        frame["y_true"] = 1 - frame.y_true
    after = compare_predictions(*artifacts)
    for original, changed in zip(before["horizons"], after["horizons"], strict=True):
        for left, right in zip(original["folds"], changed["folds"], strict=True):
            assert right["threshold_gat_lstm"] == left["threshold_gat_lstm"] == 0.5
            assert right["threshold_lstm_news"] == left["threshold_lstm_news"] == 0.5
            assert right["mcc_gat_lstm"] == -left["mcc_gat_lstm"]


def write_artifacts(directory, artifacts):
    paths = []
    for frame, record in zip(artifacts[:2], artifacts[2:], strict=True):
        path = directory / f"{record['model']}.parquet"
        write_table(frame, "predictions", path)
        path.with_name(f"{path.stem}_run.json").write_text(json.dumps(record))
        paths.append(path)
    return paths


def test_actual_cli_reads_artifacts_without_writing(tmp_path, artifacts):
    paths = write_artifacts(tmp_path, artifacts)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = subprocess.run(
        [sys.executable, "-m", "src.eval.significance", "--proposed", str(paths[0]), "--baseline", str(paths[1])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert [row["horizon"] for row in json.loads(result.stdout)["horizons"]] == [1, 5]
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert load_comparison(*paths)["horizons"][0]["n_observations"] == 20


def test_missing_artifacts_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="missing required artifacts") as error:
        load_comparison(tmp_path / "gat_lstm.parquet", tmp_path / "lstm_news.parquet")
    assert all(
        name in str(error.value)
        for name in (
            "gat_lstm.parquet",
            "lstm_news.parquet",
            "gat_lstm_run.json",
            "lstm_news_run.json",
        )
    )


def test_loader_rejects_lossy_label_cast(tmp_path, artifacts):
    paths = write_artifacts(tmp_path, artifacts)
    invalid = artifacts[0].copy()
    invalid["y_true"] = invalid.y_true.astype(float) + 0.1
    invalid.to_parquet(paths[0], index=False)
    with pytest.raises(ValueError, match="dtype"):
        load_comparison(*paths)


def test_help_without_docstrings():
    result = subprocess.run(
        [sys.executable, "-OO", "-m", "src.eval.significance", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "--proposed" in result.stdout
    assert "--baseline" in result.stdout


def test_unrelated_undefined_run_metrics_do_not_break_json(artifacts):
    # Upstream json records may contain NaN diagnostics, unrelated to MCC inputs.
    artifacts[2]["folds"][0]["val_auc"] = float("nan")
    report = compare_predictions(*artifacts)
    json.dumps(report, allow_nan=False)


def test_paired_statistic_not_pooled_or_observation_weighted(artifacts):
    # Give one fold an MCC contrast of 2, all others 0: unweighted effect = 2/5.
    selected = artifacts[1].fold == 0
    artifacts[1].loc[selected, "p_up"] = 1 - artifacts[1].loc[selected, "p_up"]
    proposed, baseline, proposed_run, baseline_run = artifacts
    expanded = []
    for frame in (proposed, baseline):
        extra = frame.loc[frame.fold == 1].copy()
        extra["ticker"] = extra.ticker + "_extra"
        expanded.append(pd.concat([frame, extra], ignore_index=True))
    for record in (proposed_run, baseline_run):
        for entry in record["folds"]:
            if entry["fold"] == 1:
                entry["test_rows"] = 8
    result = compare_predictions(*expanded, proposed_run, baseline_run)["horizons"][0]
    assert result["n_observations"] == 24
    assert [row["difference"] for row in result["folds"]] == [2, 0, 0, 0, 0]
    assert result["inference"]["mean_difference"] == 0.4
    # Every sign assignment has the same absolute mean, including zero contrasts.
    assert result["inference"]["p_value"] == 1


def test_inputs_are_not_mutated(artifacts):
    before = copy.deepcopy(artifacts)
    compare_predictions(*artifacts)
    for actual, original in zip(artifacts[:2], before[:2], strict=True):
        pd.testing.assert_frame_equal(actual, original)
    assert artifacts[2:] == before[2:]


def test_unexpected_feature_columns_rejected(artifacts):
    artifacts[0]["future_return"] = 1.0
    with pytest.raises(ValueError, match="unexpected columns"):
        compare_predictions(*artifacts)


def test_official_keys_cannot_repeat_under_different_folds(artifacts):
    proposed, baseline, *records = artifacts
    duplicate = proposed.iloc[:1].copy()
    duplicate["fold"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        compare_predictions(pd.concat([proposed, duplicate]), baseline, *records)


def test_cli_missing_files_fails_without_outputs(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.eval.significance"],
        env={**os.environ, "NDX_DATA_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "missing required artifacts" in result.stderr
    assert not list(tmp_path.iterdir())


def test_future_fold_changes_do_not_change_earlier_fold_scores(artifacts):
    before = compare_predictions(*artifacts)
    for frame in artifacts[:2]:
        frame.loc[frame.fold == 4, "y_true"] = 1 - frame.loc[frame.fold == 4, "y_true"]
    after = compare_predictions(*artifacts)
    for original, changed in zip(before["horizons"], after["horizons"], strict=True):
        assert original["folds"][:4] == changed["folds"][:4]
        assert original["folds"][4]["mcc_gat_lstm"] == -changed["folds"][4]["mcc_gat_lstm"]
