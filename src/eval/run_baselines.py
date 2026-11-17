from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.core.config import load_config, walk_forward_config
from src.core.io import read_table, write_predictions, write_table
from src.core.paths import CONFIGS_DIR, data_dir, table_path
from src.core.schema import conform, validate

# Keep in sync with src.models.baselines.BASELINES (not imported: heavy deps would slow --help).
BASELINE_NAMES = ("buy_and_hold", "arima", "logistic", "xgboost", "lstm_price", "lstm_news")


def input_hashes(include_news: bool) -> dict:
    """Detect changes to input files while separate workers read them."""
    names = ["prices", "labels", "news_features"] if include_news else ["prices", "labels"]
    hashes = {}
    for name in names:
        digest = hashlib.sha256()
        with table_path(name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


def selected_folds(panel, settings: dict, fold_indices):
    """Use the shared generator once per process and validate an optional subset."""
    from src.models.train import folds_for

    if settings["embargo_days"] != 5:
        raise ValueError("all horizons require the shared five-session embargo")
    folds = folds_for(panel, settings)
    if fold_indices is not None:
        requested = list(fold_indices)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("fold indices must be nonempty and unique")
        if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= len(folds) for i in requested):
            raise ValueError(f"fold indices must be between 0 and {len(folds) - 1}")
        folds = [folds[i] for i in requested]
    if not folds:
        raise ValueError("no complete walk-forward folds; production requires at least 635 study sessions")
    return folds


def fold_manifest(folds) -> list[dict]:
    return [
        {
            "index": fold.index,
            "train": fold.train.strftime("%Y-%m-%d").tolist(),
            "test": fold.test.strftime("%Y-%m-%d").tolist(),
        }
        for fold in folds
    ]


def training_summary(model, info: dict) -> dict:
    """Expose fitted versus fallback states without changing any learner."""
    if model.name == "buy_and_hold":
        return {"method": "constant_up"}
    if model.name == "arima":
        return {
            "fitted_tickers": len(model.models),
            "constant_tickers": len(model.constant_returns),
            "fallback_tickers": len(model.last_prices) - len(model.models) - len(model.constant_returns),
        }
    if model.name == "logistic":
        return {"fitted": model.pipeline is not None}
    if model.name == "xgboost":
        return {"fitted": model.model is not None}
    return {"epochs": info.get("epochs"), "edges_per_day": info.get("edges_per_day")}


def load_inputs(*, include_news: bool, legacy_labels: bool = False) -> tuple:
    """Read official artifacts; optionally verify and map legacy returns in memory."""
    prices = read_table("prices")
    if legacy_labels:
        from src.data.build_labels import make_labels

        # This explicit exception reads a legacy file that cannot pass official I/O.
        raw = pd.read_parquet(table_path("labels"))
        rename = {f"ret_{h}d": f"fwd_ret_{h}d" for h in (1, 5) if f"fwd_ret_{h}d" not in raw}
        labels = conform(raw.rename(columns=rename), "labels")
        validate(labels, "labels")
        expected = make_labels(prices).set_index(["date", "ticker"])
        actual = labels.set_index(["date", "ticker"])
        if not actual.index.sort_values().equals(expected.index.sort_values()):
            raise ValueError("legacy labels must have exactly the prices' date/ticker keys")
        expected = expected.reindex(actual.index)
        for column in ("fwd_ret_1d", "fwd_ret_5d", "y_1d", "y_5d"):
            observed = actual[column].to_numpy(dtype=float, na_value=np.nan)
            calculated = expected[column].to_numpy(dtype=float, na_value=np.nan)
            if not np.allclose(observed, calculated, rtol=1e-7, atol=1e-10, equal_nan=True):
                raise ValueError(f"legacy labels: {column} differs from exact future-session closes")
    else:
        try:
            labels = read_table("labels")
        except ValueError as error:
            raise ValueError(
                f"labels must follow SCHEMA.md; use --legacy-labels only for verified ret_* mapping: {error}"
            ) from error

    if include_news:
        news = read_table("news_features")
    else:
        news = pd.DataFrame(columns=["date", "ticker", "has_news", "sentiment", "confidence"])
    return prices, labels, news


def checked_predictions(predictions, panel, fold, horizon: int, name: str) -> pd.DataFrame:
    """Reject changed row sets, labels or metadata before assembling model files."""
    for column, value in (("model", name), ("horizon", horizon), ("fold", fold.index)):
        if not predictions[column].eq(value).all():
            raise ValueError(f"incorrect {column} in predictions")
    if not predictions["y_true"].isin([0, 1]).all():
        raise ValueError("prediction labels must be binary and nonmissing")
    output = conform(predictions, "predictions").sort_values(["date", "ticker"]).reset_index(drop=True)
    validate(output, "predictions")
    expected = panel.rows(fold.test, horizon)
    if expected.empty:
        raise ValueError("no eligible labelled test rows in this fold")
    if not output[["date", "ticker"]].equals(expected[["date", "ticker"]]):
        raise ValueError("predictions differ from the shared eligible date/ticker rows")
    if not np.array_equal(output["y_true"].to_numpy(), expected["y"].to_numpy()):
        raise ValueError("prediction labels differ from the shared labels")
    return output


def run_baselines(
    prices: pd.DataFrame,
    labels: pd.DataFrame,
    news: pd.DataFrame,
    model_names,
    horizons=(1, 5),
    *,
    fold_indices=None,
    walk_forward=None,
    model_configs=None,
    expected_folds=None,
) -> tuple[dict, dict]:
    """In-process worker engine; use the CLI for isolated multi-model execution.

    Optional configurations permit predeclared small synthetic tests; production
    defaults remain unchanged. No model receives a random train/test split.
    """
    from src.data.indicators import FEATURES
    from src.models.baselines import BASELINES
    from src.models.baselines.base import run_frame_baseline
    from src.models.panel import build_panel

    model_names, horizons = tuple(model_names), tuple(horizons)
    if not model_names or len(set(model_names)) != len(model_names) or set(model_names) - set(BASELINE_NAMES):
        raise ValueError("select unique names from the six supported baselines")
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("select unique horizons 1 and/or 5")
    if any(isinstance(h, bool) or not isinstance(h, Integral) or h not in (1, 5) for h in horizons):
        raise ValueError("horizons must be 1 and/or 5")

    settings = walk_forward_config() if walk_forward is None else walk_forward
    price_keys = pd.MultiIndex.from_frame(prices[["date", "ticker"]])
    label_keys = pd.MultiIndex.from_frame(labels[["date", "ticker"]])
    if not price_keys.sort_values().equals(label_keys.sort_values()):
        raise ValueError("prices and labels must have identical date/ticker keys")
    include_news = "lstm_news" in model_names
    if include_news:
        study = prices[
            prices["in_index"]
            & prices["date"].between(pd.Timestamp(settings["period"]["start"]), pd.Timestamp(settings["period"]["end"]))
        ]
        missing = pd.MultiIndex.from_frame(study[["date", "ticker"]]).difference(
            pd.MultiIndex.from_frame(news[["date", "ticker"]])
        )
        if len(missing):
            raise ValueError(f"news_features omits {len(missing)} in-index study rows")

    panel = build_panel(prices, labels, news, embeddings=include_news)
    folds = selected_folds(panel, settings, fold_indices)
    splits = fold_manifest(folds)
    if expected_folds is not None and splits != expected_folds:
        raise ValueError("worker folds differ from the parent's frozen fold definitions")

    # Frame baselines use the same causal indicators already computed by build_panel.
    frame = prices.merge(labels[["date", "ticker", "y_1d", "y_5d"]], on=["date", "ticker"], validate="one_to_one")
    days = panel.day_index(frame["date"])
    tickers = panel.tickers.get_indexer(frame["ticker"])
    frame[list(FEATURES)] = panel.price[days, tickers]
    frame = frame.set_index(["date", "ticker"]).sort_index()
    validation_fraction = load_config(CONFIGS_DIR / "gat_lstm.yaml")["train"]["validation_fraction"]
    outputs, records = {}, {}

    for name in model_names:
        parts, infos = [], []
        for horizon in horizons:
            for fold in folds:
                try:
                    config_path = (model_configs or {}).get(name)
                    model = BASELINES[name](Path(config_path)) if config_path else BASELINES[name]()
                    if hasattr(model, "run_fold"):
                        if model.cfg["train"]["validation_fraction"] != validation_fraction:
                            raise ValueError("LSTM validation fraction differs from the proposed model")
                        predictions, info = model.run_fold(panel, fold, horizon)
                        config = dict(model.cfg)
                    else:
                        predictions, info = run_frame_baseline(
                            model,
                            frame,
                            panel,
                            fold,
                            horizon,
                            validation_fraction=validation_fraction,
                            embargo=5,
                        )
                        config = {"model": name, "train": {"validation_fraction": validation_fraction}}
                    output = checked_predictions(predictions, panel, fold, horizon, name)
                    if not np.isfinite(info["threshold"]) or not 0 <= info["threshold"] <= 1:
                        raise ValueError("invalid validation threshold")
                    info["training"] = training_summary(model, info)
                    info["split"] = next(split for split in splits if split["index"] == fold.index)
                except Exception as error:
                    raise RuntimeError(f"{name}, horizon {horizon}, fold {fold.index}: {error}") from error
                parts.append(output)
                infos.append(info)
                print(f"{name}: horizon {horizon}, fold {fold.index}, {len(output)} predictions", flush=True)
        combined = pd.concat(parts, ignore_index=True)
        validate(combined, "predictions")
        outputs[name] = combined
        records[name] = {"config": {**config, "horizons": list(horizons), "walk_forward": settings}, "folds": infos}
    return outputs, records


def execute_isolated(
    prices,
    labels,
    news,
    names,
    horizons,
    *,
    fold_indices=None,
    walk_forward=None,
    model_configs=None,
    legacy_labels=False,
    reference_hashes=None,
) -> tuple[dict, dict]:
    """Run one fresh interpreter per baseline; publish nothing on worker failure."""
    from src.models.panel import build_panel

    if not names or len(set(names)) != len(names) or set(names) - set(BASELINE_NAMES):
        raise ValueError("select unique names from the six supported baselines")
    settings = walk_forward_config() if walk_forward is None else walk_forward
    panel = build_panel(prices, labels, news, embeddings=False)
    folds = selected_folds(panel, settings, fold_indices)
    splits = fold_manifest(folds)
    include_news = "lstm_news" in names
    hashes = input_hashes(include_news)
    if reference_hashes is not None and hashes != reference_hashes:
        raise ValueError("input artifacts changed while the parent was loading them")
    outputs, records = {}, {}

    with TemporaryDirectory(prefix="ndx-baselines-") as directory:
        temporary = Path(directory)
        configs = {}
        for name in names:
            if name.startswith("lstm_"):
                source = (model_configs or {}).get(name, CONFIGS_DIR / f"{name}.yaml")
                path = temporary / f"{name}_config.json"
                path.write_text(json.dumps(load_config(source), default=str), encoding="utf-8")
                configs[name] = str(path)
        plan = {
            "walk_forward": settings,
            "fold_indices": [f.index for f in folds],
            "folds": splits,
            "horizons": list(horizons),
            "model_configs": configs,
            "legacy_labels": legacy_labels,
            "input_hashes": hashes,
            "include_news": include_news,
        }
        plan_path = temporary / "plan.json"
        plan_path.write_text(json.dumps(plan, default=str), encoding="utf-8")
        environment = {**os.environ, "NDX_DATA_DIR": str(data_dir().resolve())}

        for name in names:
            print(f"Starting isolated baseline: {name}", flush=True)
            command = [
                sys.executable,
                "-B",
                "-m",
                "src.eval.run_baselines",
                "--only",
                name,
                "--worker-plan",
                str(plan_path),
            ]
            result = subprocess.run(command, env=environment, check=False)
            if result.returncode:
                print(f"{name}: worker failed with return code {result.returncode}", file=sys.stderr)
                raise subprocess.CalledProcessError(result.returncode, command)
            output = read_table("predictions", temporary / f"{name}.parquet")
            record = json.loads((temporary / f"{name}_record.json").read_text(encoding="utf-8"))
            if len(record["folds"]) != len(folds) * len(horizons):
                raise ValueError(f"{name}: incomplete worker run record")
            expected_rows = 0
            for horizon in horizons:
                for fold, split in zip(folds, splits, strict=True):
                    part = output[output["horizon"].eq(horizon) & output["fold"].eq(fold.index)]
                    checked_predictions(part, panel, fold, horizon, name)
                    expected_rows += len(part)
                    info = [i for i in record["folds"] if i["horizon"] == horizon and i["fold"] == fold.index]
                    if len(info) != 1 or info[0]["split"] != split:
                        raise ValueError(f"{name}: worker split metadata differs from the parent")
            if len(output) != expected_rows:
                raise ValueError(f"{name}: unexpected horizons or fold IDs")
            outputs[name], records[name] = output, record
        if input_hashes(include_news) != hashes:
            raise ValueError("input artifacts changed during baseline execution")
    return outputs, records


def run_worker(name: str, plan_path: Path) -> None:
    """Internal CLI worker: temporary outputs use the official prediction schema."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if input_hashes(plan["include_news"]) != plan["input_hashes"]:
        raise ValueError("input artifacts differ from the parent's frozen input hashes")
    prices, labels, news = load_inputs(include_news=name == "lstm_news", legacy_labels=plan["legacy_labels"])
    outputs, records = run_baselines(
        prices,
        labels,
        news,
        (name,),
        plan["horizons"],
        fold_indices=plan["fold_indices"],
        walk_forward=plan["walk_forward"],
        model_configs=plan["model_configs"],
        expected_folds=plan["folds"],
    )
    write_table(outputs[name], "predictions", plan_path.parent / f"{name}.parquet")
    (plan_path.parent / f"{name}_record.json").write_text(json.dumps(records[name], default=str), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true")
    which.add_argument("--only", nargs="+", choices=BASELINE_NAMES)
    p.add_argument("--horizon", type=int, choices=[1, 5], default=None, help="default: both")
    p.add_argument("--folds", type=int, nargs="+", help="smoke run on selected folds; writes no files")
    p.add_argument("--legacy-labels", action="store_true", help="verify and rename legacy ret_* labels in memory")
    p.add_argument(
        "--walk-forward-config", type=Path, help="optional fold configuration; defaults to shared production folds"
    )
    p.add_argument("--lstm-price-config", type=Path, help="optional configuration for the existing price LSTM")
    p.add_argument("--lstm-news-config", type=Path, help="optional configuration for the existing news LSTM")
    p.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    names = BASELINE_NAMES if args.all else tuple(args.only)
    horizons = (1, 5) if args.horizon is None else (args.horizon,)

    try:
        if args.worker_plan is not None:
            if args.all or len(names) != 1:
                raise ValueError("an internal worker must select exactly one baseline")
            run_worker(names[0], args.worker_plan)
            return
        settings = load_config(args.walk_forward_config) if args.walk_forward_config else walk_forward_config()
        configs = {
            name: path
            for name, path in (("lstm_price", args.lstm_price_config), ("lstm_news", args.lstm_news_config))
            if path is not None
        }
        hashes = input_hashes("lstm_news" in names)
        prices, labels, news = load_inputs(include_news="lstm_news" in names, legacy_labels=args.legacy_labels)
        outputs, records = execute_isolated(
            prices,
            labels,
            news,
            names,
            horizons,
            fold_indices=args.folds,
            walk_forward=settings,
            model_configs=configs,
            legacy_labels=args.legacy_labels,
            reference_hashes=hashes,
        )
        if args.folds is not None:
            print("Smoke run (--folds): no persistent predictions or run records written.")
            return

        from src.models.train import write_run_record

        # Complete every selected model before writing, so fitting failures cannot publish a partial run.
        for name, predictions in outputs.items():
            path = write_predictions(predictions, name)
            record = records[name]
            sidecar = write_run_record(path, record["config"], (), record["folds"])
            print(f"Wrote {path} and {sidecar.name}")
        if "buy_and_hold" in names:
            print("Buy-and-Hold portfolio evaluation uses buy_and_hold_benchmark (long-only), not ranked long/short.")
    except subprocess.CalledProcessError as error:
        status = error.returncode if error.returncode > 0 else 128 - error.returncode
        p.exit(status, f"run_baselines: worker command failed (exit {status}): {error.cmd}\n")
    except (ValueError, KeyError, OSError, ImportError, RuntimeError) as error:
        p.exit(1, f"run_baselines: {error}\n")


if __name__ == "__main__":
    main()
