"""Task 4.13: protocol, actual tiny fits, provenance and interruption safety."""

import copy
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_config
from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.eval import tune_baselines as tuning
from src.eval.splitter import walk_forward
from src.models.panel import build_panel, split_validation


@pytest.fixture
def study(tmp_path):
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    labels = read_table("labels", FIXTURES_DIR / "labels.parquet")
    news = read_table("news_features", FIXTURES_DIR / "news_features.parquet")
    panel = build_panel(prices, labels, news, embeddings=True)
    # Five full synthetic folds, with five sessions before validation and test.
    folds = walk_forward(panel.days[:55], train_days=20, test_days=6, horizon=5, embargo_days=5)
    assert len(folds) == 5
    configs = tuning.candidate_configs()
    for candidates in configs.values():
        for cfg in candidates.values():
            cfg["encoder"]["hidden"] = 4
            cfg["window_days"] = 3
            cfg["features"]["embedding_dim"] = 2
            cfg["train"].update(epochs=1, validate_every_steps=1, early_stopping_patience=1)
    input_file = tmp_path / "input-fingerprint.txt"
    input_file.write_text("synthetic inputs v1")
    return tuning.make_study(
        panel, folds, configs, {"synthetic-input": input_file}, {"environment": "synthetic tests"}, {"synthetic": True}
    )


def fake_info(study, fold, horizon, cfg, *, auc=0.6, mcc=0.2):
    fit, val = split_validation(fold.train, 0.1, 5)
    fit_count = int(study.panel.row_mask(fit, horizon).any(axis=1).sum())
    val_count = int(study.panel.row_mask(val, horizon).any(axis=1).sum())
    steps = math.ceil(fit_count / cfg["train"]["batch_days"])
    every = min(cfg["train"]["validate_every_steps"], steps)
    features = len(cfg["features"]["price"]) + 1 + len(cfg["features"]["news"])
    if cfg["features"]["embeddings"]:
        features += cfg["features"]["embedding_dim"]
    return {
        "fold": fold.index,
        "horizon": horizon,
        "seconds": 0.1,
        "stop_metric": "within_day_auc",
        "epochs": 1,
        "steps": steps,
        "steps_per_epoch": steps,
        "checks": steps // every,
        "best_step": every,
        "best_epoch": 1.0,
        "best_val_within_day_auc": auc,
        "val_loss_at_best": 0.69,
        "best_val_loss": 0.69,
        "loss_optimal_step": every,
        "val_within_day_auc_at_loss_optimum": auc,
        "min_val_loss": 0.69,
        "threshold": 0.45,
        "val_base_rate": 0.5,
        "val_mcc": mcc,
        "fit_days": fit_count,
        "val_days": val_count,
        "features": features,
        "edges_per_day": 0,
    }


def install_trainer(monkeypatch, study, *, scores=None, mcc=None):
    from src.models import train

    calls = []

    def measure(panel, edges, fold, horizon, cfg, *, predict_test, log):
        assert panel is study.panel and edges is None and predict_test is False
        assert cfg["model"] in tuning.MODELS and not cfg["gat"]["edge_types"]
        candidate = next(cid for cid, value in study.context["configs"][cfg["model"]].items() if value == cfg)
        calls.append((cfg["model"], candidate, horizon, fold.index))
        auc = scores[candidate][horizon] if scores else 0.6
        value = mcc.get(candidate, 0.2) if mcc else 0.2
        return None, fake_info(study, fold, horizon, cfg, auc=auc, mcc=value)

    monkeypatch.setattr(train, "train_fold", measure)
    return calls, measure


def test_exact_shared_candidate_protocol_and_no_config_mutation():
    original = {name: load_config(f"{name}.yaml") for name in tuning.MODELS}
    configs = tuning.candidate_configs()
    for model in tuning.MODELS:
        for candidate, settings in zip(tuning.CANDIDATES, tuning.SETTINGS, strict=True):
            cfg = configs[model][candidate]
            assert (cfg["train"]["batch_days"], cfg["train"]["lr"], cfg["encoder"]["dropout"]) == settings
            assert cfg["seed"] == 42 and cfg["horizons"] == [1, 5]
            assert cfg["encoder"]["hidden"] == 64 and cfg["encoder"]["layers"] == 1
            assert cfg["window_days"] == 20 and cfg["gat"]["edge_types"] == []
            assert cfg["train"]["epochs"] == 40
            assert cfg["train"]["early_stopping_patience"] == 4
            assert cfg["train"]["validate_every_steps"] == 14
            assert cfg["train"]["early_stopping_metric"] == "within_day_auc"
            assert cfg["train"]["weight_decay"] == 0.0001 and cfg["train"]["grad_clip"] == 1.0
        assert original[model] == load_config(f"{model}.yaml")


def test_changed_upstream_grid_is_rejected(monkeypatch):
    monkeypatch.setitem(tuning.GRID, "T6", {"batch_days": 8, "lr": 0.01})
    with pytest.raises(ValueError, match="agreed six candidates"):
        tuning.candidate_configs()


def test_all_120_jobs_delegate_to_shared_validation_only_trainer(study, tmp_path, monkeypatch):
    calls, _ = install_trainer(monkeypatch, study)
    output = tmp_path / "results"
    result = tuning.run_study(study, output)
    assert result == {"trained": 120, "reused": 0, "completed": 120, "remaining": 0}
    assert len(calls) == len(set(calls)) == 120
    for model in tuning.MODELS:
        for candidate in tuning.CANDIDATES:
            assert {(h, f) for m, c, h, f in calls if (m, c) == (model, candidate)} == {
                (h, f) for h in (1, 5) for f in range(5)
            }
    text = (output / "tuning_baselines.md").read_text()
    assert "Retrospective" in text and "including internal validation" in text
    assert "N/A" in text and "MCC 1d" in text and "MCC 5d" in text
    assert text.count("| selected |") == 2  # Exact ties choose T0 for each baseline.
    assert len(list((output / "tuning_baselines_runs").glob("*.json"))) == 121
    assert not list(output.rglob("*.parquet")) and not list(output.rglob("*.pt"))


def test_partial_candidate_cannot_win_or_publish(study, tmp_path, monkeypatch):
    install_trainer(monkeypatch, study)
    output = tmp_path / "results"
    stats = tuning.run_study(study, output, max_jobs=4)
    assert stats["remaining"] == 116
    assert not (output / "tuning_baselines.md").exists()
    records = tuning.read_records(output / "tuning_baselines_runs", study)
    rows, winners = tuning.summary_rows(study, records)
    assert winners == {} and all(row["score"] is None for row in rows)
    with pytest.raises(ValueError, match="all 120"):
        tuning.render_report(study, records)


def test_selection_uses_auc_equal_horizons_not_mcc(study, tmp_path, monkeypatch):
    scores = {cid: {1: 0.55, 5: 0.55} for cid in tuning.CANDIDATES}
    scores["T1"] = {1: 0.9, 5: 0.3}  # Average 0.6, despite leading at 1d.
    scores["T2"] = {1: 0.65, 5: 0.65}  # Hand-calculated shared score 0.65.
    install_trainer(monkeypatch, study, scores=scores, mcc={"T0": 0.9, "T2": -0.2})
    output = tmp_path / "results"
    tuning.run_study(study, output)
    records = tuning.read_records(output / "tuning_baselines_runs", study)
    rows, winners = tuning.summary_rows(study, records)
    assert winners == dict.fromkeys(tuning.MODELS, "T2")
    assert next(row["score"] for row in rows if row["candidate"] == "T1") == pytest.approx(0.6)
    assert next(row["score"] for row in rows if row["candidate"] == "T2") == 0.65
    assert all(row["val_mcc_1"] == -0.2 for row in rows if row["candidate"] == "T2")


def test_resume_does_not_duplicate_or_retrain(study, tmp_path, monkeypatch):
    calls, _ = install_trainer(monkeypatch, study)
    output = tmp_path / "results"
    first = tuning.run_study(study, output, max_jobs=4)
    second = tuning.run_study(study, output, max_jobs=4)
    assert first["completed"] == 4 and second["completed"] == 8 and second["reused"] == 4
    assert len(calls) == len(set(calls)) == 8
    tuning.run_study(study, output)
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    result = tuning.run_study(study, output)
    assert result["trained"] == 0 and result["reused"] == 120
    assert len(calls) == 120
    assert all(path.read_bytes() == content for path, content in before.items())


def test_interruption_releases_lock_and_preserves_committed_jobs(study, tmp_path, monkeypatch):
    from src.models import train

    calls, measure = install_trainer(monkeypatch, study)
    output = tmp_path / "results"

    def interrupt(*args, **kwargs):
        if len(calls) == 3:
            raise KeyboardInterrupt
        return measure(*args, **kwargs)

    monkeypatch.setattr(train, "train_fold", interrupt)
    with pytest.raises(KeyboardInterrupt):
        tuning.run_study(study, output)
    assert len(tuning.read_records(output / "tuning_baselines_runs", study)) == 3
    monkeypatch.setattr(train, "train_fold", measure)
    result = tuning.run_study(study, output, max_jobs=1)
    assert result["reused"] == 3 and result["completed"] == 4
    assert len(set(calls)) == 4


@pytest.mark.parametrize("change", ["data", "configuration", "runtime", "source"])
def test_resume_rejects_fingerprint_changes(study, tmp_path, monkeypatch, change):
    calls, _ = install_trainer(monkeypatch, study)
    output = tmp_path / "results"
    tuning.run_study(study, output, max_jobs=1)
    if change == "data":
        next(iter(study.files.values())).write_text("changed data")
    elif change == "configuration":
        study.context["configs"]["lstm_price"]["T0"]["train"]["lr"] = 0.02
    elif change == "runtime":
        study.context["runtime"]["environment"] = "other runtime"
    else:
        study.context["files"]["synthetic-source"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        tuning.run_study(study, output)
    assert len(calls) == 1


def test_data_change_during_fit_never_commits_a_result(study, tmp_path, monkeypatch):
    from src.models import train

    _, measure = install_trainer(monkeypatch, study)

    def change(*args, **kwargs):
        result = measure(*args, **kwargs)
        next(iter(study.files.values())).write_text("changed during fit")
        return result

    monkeypatch.setattr(train, "train_fold", change)
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="fingerprints changed"):
        tuning.run_study(study, output)
    assert not [p for p in (output / "tuning_baselines_runs").glob("*.json") if p.name != "manifest.json"]


@pytest.mark.parametrize("problem", ["checksum", "duplicate", "truncated", "foreign_fingerprint"])
def test_corrupt_or_duplicate_cache_is_not_silently_repaired(study, tmp_path, monkeypatch, problem):
    install_trainer(monkeypatch, study)
    output = tmp_path / "results"
    tuning.run_study(study, output, max_jobs=1)
    directory = output / "tuning_baselines_runs"
    path = directory / "lstm_price_T0_h1_fold0.json"
    original = path.read_text()
    if problem == "checksum":
        row = json.loads(original)
        row["info"]["val_mcc"] = 0.8
        path.write_text(json.dumps(row))
    elif problem == "duplicate":
        (directory / "duplicate.json").write_text(original)
    elif problem == "truncated":
        path.write_text('{"truncated":')
    else:
        row = json.loads(original)
        row["study_fingerprint"] = "other"
        path.write_text(json.dumps(row))
    before = {p: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError):
        tuning.run_study(study, output)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_pending_uncommitted_write_can_be_resumed(study, tmp_path, monkeypatch):
    install_trainer(monkeypatch, study)
    output = tmp_path / "results"
    directory = output / "tuning_baselines_runs"
    directory.mkdir(parents=True)
    (directory / ".pending-interrupted.tmp").write_text("partial uncommitted data")
    result = tuning.run_study(study, output, max_jobs=1)
    assert result["trained"] == 1
    assert len(tuning.read_records(directory, study)) == 1


def test_existing_unowned_report_and_unknown_directory_are_protected(study, tmp_path, monkeypatch):
    install_trainer(monkeypatch, study)
    output = tmp_path / "results"
    output.mkdir()
    report = output / "tuning_baselines.md"
    report.write_text("team report")
    with pytest.raises(ValueError, match="no matching study manifest"):
        tuning.run_study(study, output)
    assert report.read_text() == "team report"
    assert not (output / "tuning_baselines_runs").exists()
    report.unlink()
    directory = output / "tuning_baselines_runs"
    directory.mkdir()
    (directory / "unrelated.txt").write_text("preserve")
    with pytest.raises(ValueError, match="unrecognised"):
        tuning.run_study(study, output)
    assert (directory / "unrelated.txt").read_text() == "preserve"


def test_advisory_lock_prevents_concurrent_writer(study, tmp_path, monkeypatch):
    install_trainer(monkeypatch, study)
    output = tmp_path / "results"
    with tuning.study_lock(output / "tuning_baselines_runs"):
        with pytest.raises(ValueError, match="another tuning process"):
            tuning.run_study(study, output)


@pytest.mark.parametrize("problem", ["predictions", "test_score", "graph", "threshold", "missing_field"])
def test_bad_trainer_output_is_not_persisted(study, tmp_path, monkeypatch, problem):
    from src.models import train

    _, measure = install_trainer(monkeypatch, study)

    def wrong(*args, **kwargs):
        _, info = measure(*args, **kwargs)
        if problem == "test_score":
            info["test_auc"] = 0.99
        elif problem == "graph":
            info["edges_per_day"] = 1
        elif problem == "threshold":
            info["threshold"] = np.nan
        elif problem == "missing_field":
            del info["checks"]
        return (pd.DataFrame() if problem == "predictions" else None), info

    monkeypatch.setattr(train, "train_fold", wrong)
    output = tmp_path / "results"
    with pytest.raises(ValueError):
        tuning.run_study(study, output, max_jobs=1)
    assert len(tuning.read_records(output / "tuning_baselines_runs", study)) == 0


def test_undefined_mcc_is_null_not_zero(study, tmp_path, monkeypatch):
    from src.models import train

    _, measure = install_trainer(monkeypatch, study)

    def undefined(*args, **kwargs):
        _, info = measure(*args, **kwargs)
        info["val_mcc"] = np.nan
        return None, info

    monkeypatch.setattr(train, "train_fold", undefined)
    output = tmp_path / "results"
    tuning.run_study(study, output)
    rows, winners = tuning.summary_rows(study, tuning.read_records(output / "tuning_baselines_runs", study))
    assert all(row["val_mcc_1"] is None and row["val_mcc_5"] is None for row in rows)
    assert winners == dict.fromkeys(tuning.MODELS, "T0")
    assert "| N/A | N/A |" in (output / "tuning_baselines.md").read_text()


def test_legacy_t1_is_not_reused_without_original_fingerprints(study, tmp_path, monkeypatch):
    paths = {}
    for model in tuning.MODELS:
        path = tmp_path / f"{model}_run.json"
        entries = [
            fake_info(study, fold, h, study.context["configs"][model]["T1"]) for h in (1, 5) for fold in study.folds
        ]
        for entry in entries:
            entry["test"] = {"mcc": 1, "auc": 1}
        path.write_text(json.dumps({"config": study.context["configs"][model]["T1"], "folds": entries}))
        paths[model] = path.with_name(model + ".parquet")
    monkeypatch.setattr(tuning, "predictions_path", lambda model: paths[model])
    audit = tuning.legacy_t1_audit(study)
    assert all(item["configuration_matches"] and item["complete_coverage"] for item in audit)
    assert all(item["reusable"] is False and "not imported" in item["reason"] for item in audit)


@pytest.mark.parametrize("flags", [[], ["-OO"]])
def test_help_needs_no_neural_imports(flags):
    result = subprocess.run(
        [sys.executable, *flags, "-m", "src.eval.tune_baselines", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "--run" in result.stdout and "--report" in result.stdout


def test_default_cli_is_read_only_plan(study, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tuning, "prepare_study", lambda legacy: study)
    monkeypatch.setattr(tuning, "legacy_t1_audit", lambda _: [])
    output = tmp_path / "results"
    tuning.main(["--output-dir", str(output)])
    result = json.loads(capsys.readouterr().out)
    assert result["remaining"] == 120 and result["completed"] == 0
    assert not output.exists()
    with pytest.raises(SystemExit) as error:
        tuning.main(["--report", "--output-dir", str(output)])
    assert error.value.code == 1 and not output.exists()


def test_actual_tiny_neural_fits_both_baselines_horizons_and_future_invariance(study, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.models import train

    original_batch = train.make_batch
    original_build = train.build_model
    trained_models = []
    first_test = study.panel.day_index(study.folds[0].test[:1])[0]

    def historical_batches(days, *args, **kwargs):
        assert np.max(days) < first_test, "test observations must never reach a model batch"
        return original_batch(days, *args, **kwargs)

    def graph_disabled(*args, **kwargs):
        model = original_build(*args, **kwargs)
        assert not model.has_graph and len(model.convs) == 0
        trained_models.append(model)
        return model

    monkeypatch.setattr(train, "make_batch", historical_batches)
    monkeypatch.setattr(train, "build_model", graph_disabled)
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first = tmp_path / "first"
        result = tuning.run_study(study, first, candidates=("T1",), max_jobs=4)
        assert result["trained"] == 4
        first_records = tuning.read_records(first / "tuning_baselines_runs", study)
        # Poison future prices, news and labels; fresh validation fits must stay identical.
        altered = copy.deepcopy(study.panel)
        altered.price[first_test:] += 100
        altered.news[first_test:] += 100
        future_embeddings = altered.emb_at[:, 0] >= first_test
        altered.emb[future_embeddings] += 100
        for values in altered.labels.values():
            values[first_test:] = 1 - values[first_test:]
        second_study = tuning.Study(altered, study.folds, study.context, study.files)
        second = tmp_path / "second"
        tuning.run_study(second_study, second, candidates=("T1",), max_jobs=4)
        second_records = tuning.read_records(second / "tuning_baselines_runs", second_study)
    finally:
        torch.set_num_threads(prior_threads)
    assert {key[0] for key in first_records} == set(tuning.MODELS)
    assert {key[2] for key in first_records} == {1, 5}
    assert len(trained_models) == 8
    for key in first_records:
        left, right = first_records[key]["info"], second_records[key]["info"]
        assert all(left[name] == right[name] for name in left if name != "seconds")
        assert left["steps"] > 0 and left["checks"] > 0
        assert left["edges_per_day"] == 0 and left["best_val_within_day_auc"] is not None
    assert not (first / "tuning_baselines.md").exists()


def test_undefined_auc_does_not_produce_a_false_winner(study, tmp_path, monkeypatch):
    scores = {cid: {1: 0.6, 5: 0.6} for cid in tuning.CANDIDATES}
    scores["T3"][5] = np.nan
    install_trainer(monkeypatch, study, scores=scores)
    output = tmp_path / "results"
    tuning.run_study(study, output)
    records = tuning.read_records(output / "tuning_baselines_runs", study)
    _, winners = tuning.summary_rows(study, records)
    assert winners == {}
    text = (output / "tuning_baselines.md").read_text()
    assert "| selected |" not in text and "selection unavailable" in text


def test_modified_completed_report_is_not_overwritten(study, tmp_path, monkeypatch):
    calls, _ = install_trainer(monkeypatch, study)
    output = tmp_path / "results"
    tuning.run_study(study, output)
    path = output / "tuning_baselines.md"
    path.write_text("manual team edits")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        tuning.run_study(study, output)
    assert path.read_text() == "manual team edits" and len(calls) == 120


def test_atomic_record_publication_never_overwrites_existing(tmp_path):
    path = tmp_path / "result.json"
    tuning.atomic_create(path, "original")
    with pytest.raises(FileExistsError):
        tuning.atomic_create(path, "replacement")
    assert path.read_text() == "original"
    assert not list(tmp_path.glob(".pending-*.tmp"))


def test_zero_graph_guard_precedes_shared_training(study, tmp_path, monkeypatch):
    calls, _ = install_trainer(monkeypatch, study)
    study.context["configs"]["lstm_price"]["T0"]["gat"]["edge_types"] = ["sector"]
    with pytest.raises(ValueError, match="refusing to train a graph"):
        tuning.run_study(study, tmp_path / "results")
    assert not calls


def test_default_and_report_commands_cannot_train(study, tmp_path, monkeypatch, capsys):
    calls, _ = install_trainer(monkeypatch, study)
    output = tmp_path / "results"
    tuning.run_study(study, output)
    capsys.readouterr()
    monkeypatch.setattr(tuning, "prepare_study", lambda legacy: study)
    tuning.main(["--report", "--output-dir", str(output)])
    assert len(calls) == 120
    assert json.loads(capsys.readouterr().out)["training"] == "none"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--run", "--max-jobs", "0"],
        ["--max-jobs", "1"],
        ["--models", "gat_lstm"],
        ["--only", "T6"],
        ["--only", "T0", "T0"],
    ],
)
def test_invalid_cli_fails_before_preflight(arguments, monkeypatch):
    def must_not_run(*args):
        raise AssertionError("invalid arguments reached data loading")

    monkeypatch.setattr(tuning, "prepare_study", must_not_run)
    with pytest.raises(SystemExit) as error:
        tuning.main(arguments)
    assert error.value.code == 2


def test_changed_production_fold_settings_rejected_before_loading_data(monkeypatch):
    monkeypatch.setattr(tuning, "file_hashes", lambda _: {})
    monkeypatch.setattr(tuning, "walk_forward_config", lambda: {"train_days": 20})
    with pytest.raises(ValueError, match="official 504/5/126"):
        tuning.prepare_study()
