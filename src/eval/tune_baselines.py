"""Retrospective LSTM baseline tuning, Evaluation Task 4.13 (#59).

No arguments: read-only plan. --run explicitly enables validation-only training;
--report rebuilds the completed table without training. Official predictions,
checkpoints and configurations are never written. Historical T1 sidecars without
verifiable original input/runtime fingerprints are not imported.

The study mirrors Task 3.12: six candidates per baseline, five folds, both
horizons, seed 42, within-day AUC checkpoint/configuration selection, and MCC
at a validation-selected threshold. Later validation windows overlap earlier test
periods: this is retrospective, not prospective out-of-sample validation. Shared
preprocessing includes the internal validation portion of the outer training fold.
The shared trainer constructs test eligibility indices but, with predict_test=False,
never predicts or scores test observations. This module does not read test predictions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from src.core.config import load_config, walk_forward_config
from src.core.paths import CONFIGS_DIR, REPO_ROOT, RESULTS_DIR, predictions_path, table_path
from src.models.panel import build_panel, split_validation
from src.models.tune import GRID, config_for

MODELS = ("lstm_price", "lstm_news")
CANDIDATES = ("T0", "T1", "T2", "T3", "T4", "T5")
HORIZONS = (1, 5)
# Guard the agreed protocol if the upstream grid changes in a future merge.
SETTINGS = ((32, 1e-3, 0.1), (8, 1e-3, 0.1), (8, 3e-4, 0.1), (8, 1e-4, 0.1), (16, 3e-4, 0.1), (8, 3e-4, 0.3))
SOURCE_FILES = (
    "src/eval/tune_baselines.py",
    "src/models/tune.py",
    "src/models/train.py",
    "src/models/panel.py",
    "src/models/encoder.py",
    "src/models/gat_lstm.py",
    "src/models/baselines/lstm_price.py",
    "src/models/baselines/lstm_news.py",
    "src/data/indicators.py",
    "src/data/build_labels.py",
    "src/data/build_index_history.py",
    "src/data/validation.py",
    "src/eval/run_baselines.py",
    "src/eval/metrics.py",
    "src/models/ladder_report.py",
    "src/eval/splitter.py",
    "src/graph/comention_report.py",
    "src/core/io.py",
    "src/core/schema.py",
    "src/core/config.py",
    "src/core/paths.py",
    "configs/walk_forward.yaml",
    "configs/gat_lstm.yaml",
    "configs/lstm_price.yaml",
    "configs/lstm_news.yaml",
)
INFO_FIELDS = (
    "fold",
    "horizon",
    "seconds",
    "stop_metric",
    "epochs",
    "steps",
    "steps_per_epoch",
    "checks",
    "best_step",
    "best_epoch",
    "best_val_within_day_auc",
    "val_loss_at_best",
    "best_val_loss",
    "loss_optimal_step",
    "val_within_day_auc_at_loss_optimum",
    "min_val_loss",
    "threshold",
    "val_base_rate",
    "val_mcc",
    "fit_days",
    "val_days",
    "features",
    "edges_per_day",
)
INT_FIELDS = (
    "fold",
    "horizon",
    "epochs",
    "steps",
    "steps_per_epoch",
    "checks",
    "best_step",
    "loss_optimal_step",
    "fit_days",
    "val_days",
    "features",
    "edges_per_day",
)
METRIC_BOUNDS = {"best_val_within_day_auc": (0, 1), "val_within_day_auc_at_loss_optimum": (0, 1), "val_mcc": (-1, 1)}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hashes(files: dict[str, Path]) -> dict[str, str]:
    result = {}
    for name, path in sorted(files.items()):
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        result[name] = checksum.hexdigest()
    return result


def candidate_configs() -> dict:
    """Use the shared grid utility, retaining all unvaried baseline settings."""
    if tuple(GRID) != CANDIDATES:
        raise ValueError("upstream tuning grid differs from the agreed six candidates")
    configs = {}
    proposed = load_config(CONFIGS_DIR / "gat_lstm.yaml")
    for model in MODELS:
        base = load_config(CONFIGS_DIR / f"{model}.yaml")
        if base["features"]["price"] != proposed["features"]["price"]:
            raise ValueError("baseline price representation differs from the proposed model")
        if model == "lstm_news" and base["features"] != proposed["features"]:
            raise ValueError("Baseline 6 must preserve the shared price/news representation")
        if model == "lstm_price" and (base["features"]["news"] or base["features"]["embeddings"]):
            raise ValueError("Baseline 5 must be price only")
        configs[model] = {}
        for candidate, settings in zip(CANDIDATES, SETTINGS, strict=True):
            cfg = config_for(base, GRID[candidate])
            expected = {
                "epochs": 40,
                "early_stopping_patience": 4,
                "validate_every_steps": 14,
                "early_stopping_metric": "within_day_auc",
                "weight_decay": 0.0001,
                "grad_clip": 1.0,
                "validation_fraction": 0.1,
            }
            if any(cfg["train"].get(key) != value for key, value in expected.items()):
                raise ValueError("shared training settings differ from the agreed Task 3.12 protocol")
            actual = (cfg["train"]["batch_days"], cfg["train"]["lr"], cfg["encoder"]["dropout"])
            if actual != settings or cfg["encoder"]["hidden"] != 64 or cfg["encoder"]["layers"] != 1:
                raise ValueError("candidate settings differ from the agreed search")
            if cfg["seed"] != 42 or cfg["horizons"] != [1, 5] or cfg["window_days"] != 20:
                raise ValueError("seed, horizons and sequence length must remain fixed")
            if cfg["model"] != model or cfg["gat"]["edge_types"] or cfg.get("embargo_days", 5) != 5:
                raise ValueError("only graph-disabled LSTM baselines with a five-session embargo are allowed")
            configs[model][candidate] = cfg
    return configs


@dataclass(frozen=True)
class Study:
    panel: object
    folds: list
    context: dict
    files: dict[str, Path]

    @property
    def fingerprint(self) -> str:
        return digest(self.context)

    def verify_files(self) -> None:
        if file_hashes(self.files) != self.context["files"]:
            raise ValueError("source/configuration/data fingerprints changed; refusing reuse or training")


def make_study(panel, folds: list, configs: dict, files: dict, runtime: dict, settings: dict) -> Study:
    """Freeze explicit context; small synthetic studies use this same orchestration."""
    if [f.index for f in folds] != list(range(5)):
        raise ValueError("require exactly the five shared folds, numbered 0–4")
    if set(configs) != set(MODELS) or any(tuple(configs[m]) != CANDIDATES for m in MODELS):
        raise ValueError("require both baselines and the six agreed candidates")
    splits = []
    previous_end = None
    for fold in folds:
        if fold.train.empty or fold.test.empty:
            raise ValueError("fold dates must be nonempty")
        train_at, test_at = panel.day_index(fold.train), panel.day_index(fold.test)
        if np.any(np.diff(train_at) != 1) or np.any(np.diff(test_at) != 1):
            raise ValueError("folds must contain contiguous trading sessions")
        if test_at[0] - train_at[-1] - 1 != 5:
            raise ValueError("require the same five-session outer embargo")
        if previous_end is not None and fold.test[0] <= previous_end:
            raise ValueError("test folds must be chronological and nonoverlapping")
        previous_end = fold.test[-1]
        fit, val = split_validation(fold.train, 0.1, 5)
        splits.append(
            {
                "fold": fold.index,
                "train": fold.train.strftime("%Y-%m-%d").tolist(),
                "fit": fit.strftime("%Y-%m-%d").tolist(),
                "validation": val.strftime("%Y-%m-%d").tolist(),
                "test": fold.test.strftime("%Y-%m-%d").tolist(),
            }
        )
    for model in MODELS:
        for cfg in configs[model].values():
            if cfg["model"] != model or cfg["gat"]["edge_types"]:
                raise ValueError("only the two graph-disabled baselines may be tuned")
            if cfg["seed"] != 42 or cfg["horizons"] != [1, 5] or cfg["train"]["validation_fraction"] != 0.1:
                raise ValueError("require shared seed, horizons and validation fraction")
            if cfg["train"]["early_stopping_metric"] != "within_day_auc":
                raise ValueError("checkpoint selection must use validation within-day AUC")
    context = {
        "task": "4.13",
        "version": 1,
        "study": "retrospective",
        "objective": "equal mean of five validation fold AUCs per horizon, then equal mean over horizons",
        "tie_break": "first in T0–T5 order",
        "candidate_order": list(CANDIDATES),
        "configs": copy.deepcopy(configs),
        "folds": splits,
        "walk_forward": settings,
        "files": file_hashes(files),
        "runtime": runtime,
    }
    return Study(panel, folds, context, files)


def prepare_study(legacy_labels: bool = False) -> Study:
    """Production preflight only: no model fitting and no output writes."""
    files = {name: REPO_ROOT / name for name in SOURCE_FILES}
    files.update({f"data/{name}": table_path(name) for name in ("prices", "labels", "news_features")})
    before = file_hashes(files)
    configs = candidate_configs()
    settings = walk_forward_config()
    expected = {"train_days": 504, "embargo_days": 5, "test_days": 126, "step_days": 126, "expanding": False}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("production tuning requires the official 504/5/126 rolling folds")
    from src.eval.run_baselines import load_inputs
    from src.graph.comention_report import shared_folds

    prices, labels, news = load_inputs(include_news=True, legacy_labels=legacy_labels)
    import pandas as pd

    price_keys = pd.MultiIndex.from_frame(prices[["date", "ticker"]])
    label_keys = pd.MultiIndex.from_frame(labels[["date", "ticker"]])
    if not price_keys.sort_values().equals(label_keys.sort_values()):
        raise ValueError("prices and labels must have identical keys")
    study_rows = prices.loc[
        prices.in_index
        & prices.date.between(pd.Timestamp(settings["period"]["start"]), pd.Timestamp(settings["period"]["end"]))
    ]
    missing_news = pd.MultiIndex.from_frame(study_rows[["date", "ticker"]]).difference(
        pd.MultiIndex.from_frame(news[["date", "ticker"]])
    )
    if len(missing_news):
        raise ValueError("news_features omits historical-member study rows")
    panel = build_panel(prices, labels, news, embeddings=True)
    folds = shared_folds(prices, settings)
    import torch

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "pyarrow", "torch", "torch-geometric", "pandas-market-calendars")
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    }
    study = make_study(panel, folds, configs, files, runtime, json.loads(json.dumps(settings, default=str)))
    if study.context["files"] != before:
        raise ValueError("input or source files changed during preflight")
    return study


def legacy_t1_audit(study: Study) -> list[dict]:
    """Inspect provenance, not test scores. Legacy scores are never silently imported."""
    audit = []
    for model in MODELS:
        path = predictions_path(model).with_name(f"{model}_run.json")
        entry = {"model": model, "path": str(path), "reusable": False}
        if not path.is_file():
            entry["reason"] = "historical T1 run record is absent"
        else:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                entry["configuration_matches"] = record["config"] == study.context["configs"][model]["T1"]
                entries = record["folds"]
                entry["complete_coverage"] = len(entries) == 10 and {(e["horizon"], e["fold"]) for e in entries} == {
                    (h, f) for h in HORIZONS for f in range(5)
                }
                entry["training_commit"] = record.get("git_commit")
                entry["protocol_matches"] = entry["complete_coverage"] and all(
                    e.get("stop_metric") == "within_day_auc"
                    and e.get("edges_per_day") == 0
                    and e.get("fit_days") == len(study.context["folds"][e["fold"]]["fit"])
                    and e.get("val_days") == len(study.context["folds"][e["fold"]]["validation"])
                    for e in entries
                )
                entry["reason"] = "original training data/source/runtime fingerprints cannot be verified; not imported"
            except (ValueError, KeyError, TypeError) as error:
                entry["reason"] = f"invalid historical run metadata: {error}"
        audit.append(entry)
    return audit


def clean_info(info: dict, study: Study, key: tuple) -> dict:
    """Validate shared diagnostics and represent undefined metrics as JSON null."""
    model, candidate, horizon, fold_id = key
    if any(name.startswith("test") for name in info):
        raise ValueError("test information must not enter tuning records")
    if any(name not in info for name in INFO_FIELDS):
        raise ValueError("shared trainer returned incomplete validation diagnostics")
    result = {}
    for name in INFO_FIELDS:
        value = info[name]
        if name == "stop_metric":
            if value != "within_day_auc":
                raise ValueError("unexpected checkpoint objective")
            result[name] = value
        elif name in INT_FIELDS:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"invalid integer diagnostic {name}")
            result[name] = int(value)
        elif name in METRIC_BOUNDS and (value is None or isinstance(value, Real) and np.isnan(value)):
            result[name] = None
        else:
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"invalid finite diagnostic {name}")
            result[name] = float(value)
    cfg = study.context["configs"][model][candidate]
    fold = study.folds[fold_id]
    fit, val = split_validation(fold.train, cfg["train"]["validation_fraction"], 5)
    fit_count = int(study.panel.row_mask(fit, horizon).any(axis=1).sum())
    val_count = int(study.panel.row_mask(val, horizon).any(axis=1).sum())
    features = len(cfg["features"]["price"]) + 1 + len(cfg["features"]["news"])
    if cfg["features"]["embeddings"]:
        features += cfg["features"]["embedding_dim"]
    expected = {
        "horizon": horizon,
        "fold": fold_id,
        "edges_per_day": 0,
        "fit_days": fit_count,
        "val_days": val_count,
        "features": features,
        "steps_per_epoch": math.ceil(fit_count / cfg["train"]["batch_days"]),
    }
    if not fit_count or not val_count or any(result[name] != value for name, value in expected.items()):
        raise ValueError("validation diagnostics differ from the shared fold/configuration or graph is enabled")
    if not 1 <= result["epochs"] <= cfg["train"]["epochs"]:
        raise ValueError("invalid epoch count")
    if not 1 <= result["steps"] <= result["epochs"] * result["steps_per_epoch"]:
        raise ValueError("invalid optimizer step count")
    every = min(cfg["train"]["validate_every_steps"], result["steps_per_epoch"])
    if result["checks"] != result["steps"] // every:
        raise ValueError("validation-check schedule differs from shared training")
    for name in ("best_step", "loss_optimal_step"):
        if not 1 <= result[name] <= result["steps"] or result[name] % every:
            raise ValueError(f"invalid {name}")
    if not 0 <= result["threshold"] <= 1 or not 0 <= result["val_base_rate"] <= 1 or result["seconds"] < 0:
        raise ValueError("invalid threshold/base rate/runtime")
    for name, (low, high) in METRIC_BOUNDS.items():
        if result[name] is not None and not low <= result[name] <= high:
            raise ValueError(f"invalid validation metric {name}")
    return result


def record_name(key: tuple) -> str:
    model, candidate, horizon, fold = key
    return f"{model}_{candidate}_h{horizon}_fold{fold}.json"


def read_records(directory: Path, study: Study) -> dict:
    """Only complete, fingerprint-verified, canonically named records are reusable."""
    manifest = directory / "manifest.json"
    if not manifest.exists():
        if directory.exists() and any(
            p.name != ".lock" and not (p.name.startswith(".pending-") and p.suffix == ".tmp")
            for p in directory.iterdir()
        ):
            raise ValueError("unrecognised tuning directory without a manifest")
        return {}
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if meta.get("fingerprint") != digest(meta.get("context")) or meta["fingerprint"] != study.fingerprint:
        raise ValueError("study fingerprint mismatch; use a new output directory, never merge incompatible runs")
    records = {}
    for path in sorted(directory.glob("*.json")):
        if path == manifest:
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row.get("model"), row.get("candidate"), row.get("horizon"), row.get("fold"))
        if key[0] not in MODELS or key[1] not in CANDIDATES or type(key[2]) is not int or type(key[3]) is not int:
            raise ValueError("invalid tuning record key")
        if key[2] not in HORIZONS or key[3] not in range(5) or path.name != record_name(key) or key in records:
            raise ValueError("invalid, duplicate or noncanonical tuning record")
        checksum = row.pop("record_sha256", None)
        if checksum != digest(row) or row.get("study_fingerprint") != study.fingerprint:
            raise ValueError("tuning record checksum/fingerprint mismatch")
        if row.get("config_sha256") != digest(study.context["configs"][key[0]][key[1]]):
            raise ValueError("cached configuration differs from current candidate")
        if row.get("origin") != "measured_shared_validation_only":
            raise ValueError("unverified record origin")
        clean_info(row["info"], study, key)
        records[key] = row
    return records


def atomic_create(path: Path, value: str) -> None:
    """Publish one complete file without overwriting; ignore uncommitted .tmp files."""
    fd, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".tmp", dir=path.parent)
    pending = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)  # Atomic, exclusive publication; existing files raise.
    finally:
        pending.unlink(missing_ok=True)


@contextmanager
def study_lock(directory: Path):
    """POSIX advisory locking releases automatically even after process termination."""
    try:
        import fcntl
    except ImportError as error:
        raise ValueError("tuning execution requires POSIX file locking (macOS/Linux)") from error
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another tuning process owns this study") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def summary_rows(study: Study, records: dict) -> tuple[list[dict], dict]:
    rows, winners = [], {}
    for model in MODELS:
        model_rows = []
        for candidate in CANDIDATES:
            entries = [
                records[model, candidate, h, f]["info"]
                for h in HORIZONS
                for f in range(5)
                if (model, candidate, h, f) in records
            ]
            cfg = study.context["configs"][model][candidate]
            row = {
                "model": model,
                "candidate": candidate,
                "records": len(entries),
                "score": None,
                "batch_days": cfg["train"]["batch_days"],
                "lr": cfg["train"]["lr"],
                "hidden": cfg["encoder"]["hidden"],
                "dropout": cfg["encoder"]["dropout"],
            }
            for metric in ("best_val_within_day_auc", "val_mcc"):
                for horizon in HORIZONS:
                    values = [e[metric] for e in entries if e["horizon"] == horizon]
                    row[f"{metric}_{horizon}"] = sum(values) / 5 if len(values) == 5 and None not in values else None
            if len(entries) == 10 and all(row[f"best_val_within_day_auc_{h}"] is not None for h in HORIZONS):
                row["score"] = sum(row[f"best_val_within_day_auc_{h}"] for h in HORIZONS) / 2
            model_rows.append(row)
        if all(row["records"] == 10 and row["score"] is not None for row in model_rows):
            winners[model] = max(model_rows, key=lambda row: row["score"])["candidate"]
        rows.extend(model_rows)
    return rows, winners


def render_report(study: Study, records: dict) -> str:
    if len(records) != 120:
        raise ValueError("all 120 measured fold/horizon records are required before publishing the table")
    rows, winners = summary_rows(study, records)
    lines = [
        "# Retrospective tuning study: LSTM baselines (Task 4.13, Issue #59)",
        "",
        f"Study fingerprint: `{study.fingerprint}`. Full configurations, data/source hashes, runtime and fold dates:",
        (
            "[`manifest.json`](tuning_baselines_runs/manifest.json). Per-fit JSON records retain "
            "thresholds and stopping diagnostics."
        ),
        "",
        "Selection and early stopping use validation within-day AUC; MCC uses each validation-selected threshold.",
        "Five folds and both horizons receive equal weight; one configuration per baseline, shared across horizons.",
        "Exact ties use T0–T5 order. All six candidates need ten valid records before a winner is selected.",
        "Undefined metrics are N/A, never zero; an undefined AUC prevents selection for that baseline.",
        "",
        "**Retrospective limitation:** later validation windows overlap earlier test periods. These are not",
        (
            "prospectively out-of-sample validated configurations. No test predictions or "
            "performance scores enter selection."
        ),
        (
            "Shared preprocessing/PCA is fitted on the outer training fold, including internal "
            "validation; it is unchanged."
        ),
        (
            "Validation scores are optimistic because checkpoints, thresholds and configurations "
            "reuse validation observations."
        ),
        "",
        "Historical T1 sidecars lack verifiable original data/source/runtime fingerprints and were not imported.",
        "All rows use measured Task 4.13 records; resumption reuses them only after exact fingerprint verification.",
        "",
        (
            "| Baseline | Candidate | Batch days | Learning rate | Hidden | Heads | Dropout | Val "
            "MCC 1d | Val MCC 5d | Val within-day AUC 1d | Val within-day AUC 5d | "
            "Selection score | Outcome | Provenance |"
        ),
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    def show(value):
        return "N/A" if value is None else f"{value:.6f}"

    for row in rows:
        outcome = "selection unavailable"
        if row["model"] in winners:
            outcome = "selected" if winners[row["model"]] == row["candidate"] else "not selected"
        lines.append(
            f"| {row['model']} | {row['candidate']} | {row['batch_days']} | {row['lr']:.0e} | "
            f"{row['hidden']} | N/A | {row['dropout']:.1f} | {show(row['val_mcc_1'])} | {show(row['val_mcc_5'])} | "
            f"{show(row['best_val_within_day_auc_1'])} | {show(row['best_val_within_day_auc_5'])} | "
            f"{show(row['score'])} | {outcome} | 10 measured, fingerprint-verified records |"
        )
    return "\n".join(lines) + "\n"


def publish_report(output: Path, study: Study, records: dict) -> None:
    path = output / "tuning_baselines.md"
    text = render_report(study, records)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("existing tuning_baselines.md differs; refusing to overwrite it")
        return
    atomic_create(path, text)


def run_study(study: Study, output: Path, *, models=MODELS, candidates=CANDIDATES, max_jobs: int | None = None) -> dict:
    """Fit only missing jobs; selections affect scheduling, not the frozen study."""
    for selected, allowed in ((models, MODELS), (candidates, CANDIDATES)):
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(allowed):
            raise ValueError("select unique, supported baselines and candidates")
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs <= 0):
        raise ValueError("max_jobs must be a positive integer")
    directory = output / "tuning_baselines_runs"
    if (output / "tuning_baselines.md").exists() and not (directory / "manifest.json").exists():
        raise ValueError("existing report has no matching study manifest; refusing to overwrite")
    study.verify_files()
    read_records(directory, study)  # Reject incompatible state before creating files.
    trained = 0
    with study_lock(directory):
        records = read_records(directory, study)
        reused = sum(key[0] in models and key[1] in candidates for key in records)
        manifest = directory / "manifest.json"
        if not manifest.exists():
            value = {
                "context": study.context,
                "fingerprint": study.fingerprint,
                "historical_t1_audit": legacy_t1_audit(study),
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            atomic_create(manifest, json.dumps(value, indent=2, allow_nan=False) + "\n")
        from src.models.train import train_fold

        for candidate in candidates:
            for fold in study.folds:
                for model in models:
                    for horizon in HORIZONS:
                        key = (model, candidate, horizon, fold.index)
                        if key in records or max_jobs is not None and trained >= max_jobs:
                            continue
                        study.verify_files()
                        cfg = copy.deepcopy(study.context["configs"][model][candidate])
                        if cfg["model"] != model or cfg["gat"]["edge_types"]:
                            raise ValueError("refusing to train a graph model or a different model")
                        predictions, info = train_fold(
                            study.panel, None, fold, horizon, cfg, predict_test=False, log=lambda *_: None
                        )
                        if predictions is not None:
                            raise ValueError("shared trainer returned test predictions in validation-only mode")
                        info = clean_info(info, study, key)
                        study.verify_files()
                        row = {
                            "study_fingerprint": study.fingerprint,
                            "model": model,
                            "candidate": candidate,
                            "horizon": horizon,
                            "fold": fold.index,
                            "config_sha256": digest(cfg),
                            "origin": "measured_shared_validation_only",
                            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "info": info,
                        }
                        checksum = digest(row)
                        atomic_create(
                            directory / record_name(key), canonical({**row, "record_sha256": checksum}) + "\n"
                        )
                        records[key] = row
                        trained += 1
                        print(f"Completed {model} {candidate} h={horizon} fold={fold.index}", flush=True)
        study.verify_files()
        if len(records) == 120:
            publish_report(output, study, records)
    return {"trained": trained, "reused": reused, "completed": len(records), "remaining": 120 - len(records)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Retrospective validation-only LSTM baseline tuning; default is plan only"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="explicitly authorize training missing validation jobs")
    mode.add_argument("--report", action="store_true", help="publish the fully measured table without training")
    parser.add_argument("--legacy-labels", action="store_true", help="verify and map legacy forward returns in memory")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "tables")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--only", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--max-jobs", type=int, help="commit at most this many new jobs; resume with the same command")
    args = parser.parse_args(argv)
    if args.max_jobs is not None and (not args.run or args.max_jobs <= 0):
        parser.error("--max-jobs must be positive and requires --run")
    if len(set(args.models)) != len(args.models) or len(set(args.only)) != len(args.only):
        parser.error("models and candidates must be unique")
    try:
        study = prepare_study(args.legacy_labels)
        directory = args.output_dir / "tuning_baselines_runs"
        if args.run:
            result = run_study(study, args.output_dir, models=args.models, candidates=args.only, max_jobs=args.max_jobs)
        elif args.report:
            study.verify_files()
            records = read_records(directory, study)
            if len(records) != 120:
                raise ValueError("all 120 measured records are required; no table written")
            with study_lock(directory):
                records = read_records(directory, study)
                study.verify_files()
                publish_report(args.output_dir, study, records)
            result = {"report": str(args.output_dir / "tuning_baselines.md"), "training": "none"}
        else:
            study.verify_files()
            records = read_records(directory, study)
            result = {
                "mode": "read-only plan; use --run only after production authorization",
                "study_fingerprint": study.fingerprint,
                "completed": len(records),
                "remaining": 120 - len(records),
                "historical_t1_audit": legacy_t1_audit(study),
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"tune_baselines: {error}\n")


if __name__ == "__main__":
    main()
