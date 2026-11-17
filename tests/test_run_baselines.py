"""Task 4.8: orchestration, official artifacts and lightweight baseline fitting."""

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_config
from src.core.io import read_predictions, read_table
from src.core.paths import CONFIGS_DIR, FIXTURES_DIR
from src.core.schema import validate
from src.eval import run_baselines as runner
from src.models.baselines import BASELINES, BuyAndHold, LSTMNews
from src.models.panel import split_validation

SHORT = {
    # Stop before the fixture's unlabelled final five sessions.
    "period": {"start": "2021-01-01", "end": "2021-03-19"},
    "train_days": 35,
    "test_days": 5,
    "step_days": 5,
    "expanding": False,
    "embargo_days": 5,
}


@pytest.fixture
def tables():
    return {name: read_table(name, FIXTURES_DIR / f"{name}.parquet") for name in ("prices", "labels", "news_features")}


@pytest.fixture
def inputs(tables):
    return tables["prices"], tables["labels"], tables["news_features"]


def call(inputs, names=("buy_and_hold",), horizons=(1, 5), **kwargs):
    return runner.run_baselines(*inputs, names, horizons, walk_forward=SHORT, **kwargs)


@pytest.mark.parametrize("optimization", [(), ("-OO",)])
def test_help_is_lightweight(optimization):
    result = subprocess.run(
        [sys.executable, *optimization, "-m", "src.eval.run_baselines", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--all" in result.stdout and "--folds" in result.stdout and "--legacy-labels" in result.stdout


@pytest.mark.parametrize("documentation", [None, ""])
def test_cli_help_does_not_depend_on_module_documentation(documentation, monkeypatch, capsys):
    monkeypatch.setattr(runner, "__doc__", documentation)
    monkeypatch.setattr(sys, "argv", ["run_baselines", "--help"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 0
    output = capsys.readouterr()
    assert "Run baselines on the shared walk-forward folds" in output.out
    assert "--all" in output.out and not output.err


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--all", "--only", "arima"],
        ["--only", "unknown"],
        ["--all", "--horizon", "2"],
        ["--all", "--folds", "invalid"],
    ],
)
def test_invalid_cli_arguments_are_reported_by_argparse(arguments):
    result = subprocess.run(
        [sys.executable, "-B", "-m", "src.eval.run_baselines", *arguments], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr and "error:" in result.stderr
    assert "Traceback" not in result.stderr


def price_cli_inputs(tables, tmp_path, *, legacy_labels=False):
    from src.data.build_labels import make_labels

    tables["prices"].to_parquet(tmp_path / "prices.parquet", index=False)
    labels = make_labels(tables["prices"])
    if legacy_labels:
        labels = labels.rename(columns={"fwd_ret_1d": "ret_1d", "fwd_ret_5d": "ret_5d"})
    labels.to_parquet(tmp_path / "labels.parquet", index=False)
    config = tmp_path / "walk_forward.json"
    config.write_text(json.dumps(SHORT), encoding="utf-8")
    command = [
        sys.executable,
        "-B",
        "-m",
        "src.eval.run_baselines",
        "--only",
        "buy_and_hold",
        "--walk-forward-config",
        str(config),
        "--horizon",
        "5",
    ]
    if legacy_labels:
        command.append("--legacy-labels")
    return command, {**os.environ, "NDX_DATA_DIR": str(tmp_path)}


@pytest.mark.parametrize("legacy_labels", [False, True])
def test_public_only_horizon_and_fold_selection(tables, tmp_path, legacy_labels):
    command, environment = price_cli_inputs(tables, tmp_path, legacy_labels=legacy_labels)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    result = subprocess.run(
        [*command, "--folds", "1"], env=environment, capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "buy_and_hold: horizon 5, fold 1" in result.stdout
    assert "horizon 1" not in result.stdout and "fold 0" not in result.stdout
    assert result.stdout.count("Starting isolated baseline:") == 1
    assert "no persistent" in result.stdout
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_public_invalid_fold_does_not_start_workers(tables, tmp_path):
    command, environment = price_cli_inputs(tables, tmp_path)
    result = subprocess.run(
        [*command, "--folds", "-1"], env=environment, capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 1
    assert "fold indices" in result.stderr and "Traceback" not in result.stderr
    assert "Starting isolated baseline:" not in result.stdout
    assert not (tmp_path / "predictions").exists()


def test_registry_is_exactly_the_six_expected_baselines():
    assert runner.BASELINE_NAMES == (
        "buy_and_hold",
        "arima",
        "logistic",
        "xgboost",
        "lstm_price",
        "lstm_news",
    )
    assert set(runner.BASELINE_NAMES) == set(BASELINES)


@pytest.mark.parametrize("horizons", [(1,), (5,), (1, 5)])
def test_rows_horizons_membership_missing_labels_and_schema(inputs, horizons):
    prices, labels, news = inputs
    original = [frame.copy(deep=True) for frame in inputs]
    prices = prices.copy()
    labels = labels.copy()
    target = labels["date"].sort_values().unique()[41]
    prices.loc[(prices["date"] == target) & prices["ticker"].eq("BBB"), "in_index"] = False
    labels.loc[(labels["date"] == target) & labels["ticker"].eq("AAA"), "y_1d"] = pd.NA
    outputs, records = call((prices, labels, news), horizons=horizons, fold_indices=[0, 1])
    output = outputs["buy_and_hold"]
    validate(output, "predictions")
    assert not output.duplicated(["model", "horizon", "date", "ticker"]).any()
    assert set(output["horizon"]) == set(horizons)
    assert set(output["fold"]) == {0, 1}
    assert output["p_up"].eq(1).all()
    assert str(output["p_up"].dtype) == "float32"
    assert not ((output["date"] == target) & output["ticker"].eq("BBB")).any()
    if 1 in horizons:
        assert not ((output["date"] == target) & output["ticker"].eq("AAA") & output["horizon"].eq(1)).any()
    expected = prices[prices["in_index"]].merge(labels, on=["date", "ticker"])
    for h in horizons:
        actual = output[output["horizon"] == h]
        available = expected[expected[f"y_{h}d"].notna() & expected["date"].isin(actual["date"])]
        assert len(actual) == len(available)
    assert len(records["buy_and_hold"]["folds"]) == 2 * len(horizons)
    for actual, before in zip(inputs, original, strict=True):
        pd.testing.assert_frame_equal(actual, before)


def test_both_interfaces_receive_identical_folds_and_validation(inputs, monkeypatch):
    calls = []

    class RecordingFrame(BuyAndHold):
        def fit(self, frame, dates, horizon):
            assert not any(c.startswith("fwd_") for c in frame.columns)
            calls.append((self.name, horizon, dates.copy()))
            super().fit(frame, dates, horizon)

    class RecordingPanel:
        name = "lstm_news"
        cfg = LSTMNews().cfg

        def run_fold(self, panel, fold, horizon):
            calls.append((self.name, horizon, fold))
            rows = panel.rows(fold.test, horizon).rename(columns={"y": "y_true"})
            predictions = rows.assign(p_up=0.6, model=self.name, horizon=horizon, fold=fold.index)
            return predictions, {"threshold": 0.5, "fold": fold.index, "horizon": horizon}

    monkeypatch.setitem(BASELINES, "buy_and_hold", RecordingFrame)
    monkeypatch.setitem(BASELINES, "lstm_news", RecordingPanel)
    call(inputs, names=("buy_and_hold", "lstm_news"), fold_indices=[0, 1])
    frame_calls, panel_calls = calls[:4], calls[4:]
    for (_, h, fit), (_, other_h, fold) in zip(frame_calls, panel_calls, strict=True):
        assert h == other_h
        expected, validation = split_validation(fold.train, 0.1, 5)
        pd.testing.assert_index_equal(fit, expected)
        calendar = pd.DatetimeIndex(sorted(inputs[0]["date"].unique()))
        assert calendar.get_loc(fold.test[0]) - calendar.get_loc(fold.train[-1]) == 6
        assert calendar.get_loc(validation[0]) - calendar.get_loc(fit[-1]) == 6
    assert panel_calls[0][2] is panel_calls[2][2]


def test_post_training_changes_do_not_change_fitted_logistic_parameters(inputs, monkeypatch):
    pytest.importorskip("sklearn")
    original_factory = BASELINES["logistic"]
    fitted = []

    def factory():
        model = original_factory()
        fitted.append(model)
        return model

    monkeypatch.setitem(BASELINES, "logistic", factory)
    call(inputs, names=("logistic",), horizons=(1,), fold_indices=[0])
    prices, labels, news = (frame.copy() for frame in inputs)
    cutoff = sorted(prices["date"].unique())[34]
    later = prices["date"] > cutoff
    prices.loc[later, ["open", "high", "low", "close", "volume"]] *= 10
    labels.loc[labels["date"] > cutoff, "y_1d"] = 1
    call((prices, labels, news), names=("logistic",), horizons=(1,), fold_indices=[0])
    for first, second in zip(fitted[0].pipeline.steps, fitted[1].pipeline.steps, strict=True):
        for attr in ("statistics_", "mean_", "scale_", "coef_", "intercept_"):
            if hasattr(first[1], attr):
                np.testing.assert_array_equal(getattr(first[1], attr), getattr(second[1], attr))


@pytest.mark.parametrize(
    "names,horizons,folds",
    [
        ((), (1,), [0]),
        (("unknown",), (1,), [0]),
        (("buy_and_hold", "buy_and_hold"), (1,), [0]),
        (("buy_and_hold",), (2,), [0]),
        (("buy_and_hold",), (True,), [0]),
        (("buy_and_hold",), (1, 1), [0]),
        (("buy_and_hold",), (), [0]),
        (("buy_and_hold",), (1,), [-1]),
        (("buy_and_hold",), (1,), [100]),
        (("buy_and_hold",), (1,), []),
        (("buy_and_hold",), (1,), [0, 0]),
    ],
)
def test_invalid_selection(inputs, names, horizons, folds):
    with pytest.raises(ValueError):
        call(inputs, names, horizons, fold_indices=folds)


def test_production_requires_full_shared_folds(inputs):
    with pytest.raises(ValueError, match="635"):
        runner.run_baselines(*inputs, ("buy_and_hold",))
    wrong = {**SHORT, "embargo_days": 1}
    with pytest.raises(ValueError, match="five-session"):
        runner.run_baselines(*inputs, ("buy_and_hold",), walk_forward=wrong)


def test_missing_news_and_mismatched_label_keys_fail(inputs):
    prices, labels, news = inputs
    with pytest.raises(ValueError, match="news_features omits"):
        call((prices, labels, news.iloc[1:]), names=("lstm_news",))
    with pytest.raises(ValueError, match="identical"):
        call((prices, labels.iloc[1:], news))


def test_no_labelled_test_rows_fail_instead_of_writing_empty_results(inputs):
    prices, labels, news = inputs
    labels = labels.copy()
    days = sorted(prices["date"].unique())
    labels.loc[labels["date"].isin(days[40:45]), "y_1d"] = pd.NA
    with pytest.raises(RuntimeError, match="no eligible labelled test rows"):
        call((prices, labels, news), horizons=(1,), fold_indices=[0])


@pytest.mark.parametrize("problem", ["rows", "truth", "model", "horizon", "fold", "probability", "duplicate"])
def test_bad_model_output_fails_with_context(inputs, monkeypatch, problem):
    class BadPanel:
        name = "lstm_news"
        cfg = LSTMNews().cfg

        def run_fold(self, panel, fold, horizon):
            rows = panel.rows(fold.test, horizon).rename(columns={"y": "y_true"})
            rows = rows.assign(p_up=0.6, model=self.name, horizon=horizon, fold=fold.index)
            if problem == "rows":
                rows = rows.iloc[1:]
            elif problem == "truth":
                rows.loc[0, "y_true"] = 1 - rows.loc[0, "y_true"]
            elif problem == "duplicate":
                rows = pd.concat([rows, rows.iloc[:1]])
            else:
                column = "p_up" if problem == "probability" else problem
                rows[column] = {"model": "wrong", "horizon": 2, "fold": 10, "probability": np.nan}[problem]
            return rows, {"threshold": 0.5}

    monkeypatch.setitem(BASELINES, "lstm_news", BadPanel)
    with pytest.raises(RuntimeError, match="lstm_news, horizon 1, fold 0"):
        call(inputs, names=("lstm_news",), horizons=(1,), fold_indices=[0])


def configure_cli(monkeypatch, inputs, names, *, smoke=False):
    monkeypatch.setattr(runner, "load_inputs", lambda **kwargs: inputs)
    monkeypatch.setattr(runner, "walk_forward_config", lambda: SHORT)
    monkeypatch.setattr(runner, "input_hashes", lambda include_news: {})

    def in_process(prices, labels, news, selected, horizons, **kwargs):
        return runner.run_baselines(
            prices, labels, news, selected, horizons, fold_indices=kwargs["fold_indices"], walk_forward=SHORT
        )

    monkeypatch.setattr(runner, "execute_isolated", in_process)
    arguments = ["run_baselines", "--only", *names]
    if smoke:
        arguments += ["--folds", "0"]
    monkeypatch.setattr(sys, "argv", arguments)


def test_cli_writes_official_files_and_existing_threshold_records(inputs, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    configure_cli(monkeypatch, inputs, ["buy_and_hold"])
    runner.main()
    output = read_predictions("buy_and_hold")
    validate(output, "predictions")
    assert set(output["horizon"]) == {1, 5}
    record = json.loads((tmp_path / "predictions/buy_and_hold_run.json").read_text())
    assert len(record["folds"]) == 6
    assert all(0 <= info["threshold"] <= 1 for info in record["folds"])
    assert "long-only" in capsys.readouterr().out
    assert {p.name for p in tmp_path.rglob("*") if p.is_file()} == {
        "buy_and_hold.parquet",
        "buy_and_hold_run.json",
    }


def test_cli_smoke_writes_nothing(inputs, monkeypatch, tmp_path):
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    configure_cli(monkeypatch, inputs, ["buy_and_hold"], smoke=True)
    runner.main()
    assert list(tmp_path.iterdir()) == []


def test_cli_all_selects_six_and_both_horizons_without_writing(monkeypatch, inputs):
    monkeypatch.setattr(runner, "load_inputs", lambda **kwargs: inputs)
    monkeypatch.setattr(runner, "input_hashes", lambda include_news: {})
    monkeypatch.setattr(sys, "argv", ["run_baselines", "--all", "--folds", "0"])

    def record(*args, **kwargs):
        assert args[3] == runner.BASELINE_NAMES
        assert args[4] == (1, 5) and kwargs["fold_indices"] == [0]
        return {}, {}

    monkeypatch.setattr(runner, "execute_isolated", record)
    runner.main()


def test_cli_failure_publishes_nothing(inputs, monkeypatch, tmp_path, capsys):
    class Broken(BuyAndHold):
        def fit(self, *args):
            raise ImportError("required dependency unavailable")

    monkeypatch.setitem(BASELINES, "arima", Broken)
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    configure_cli(monkeypatch, inputs, ["buy_and_hold", "arima"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 1
    assert "arima, horizon 1, fold 0" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_official_io_and_explicit_legacy_mapping(tables, monkeypatch, tmp_path):
    from src.data.build_labels import make_labels

    # Frozen fixtures include business-day holidays; derive exact-session labels here.
    tables["labels"] = make_labels(tables["prices"])
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    for name, frame in tables.items():
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    official = runner.load_inputs(include_news=True)
    pd.testing.assert_frame_equal(official[1], tables["labels"])
    legacy = tables["labels"].rename(columns={"fwd_ret_1d": "ret_1d", "fwd_ret_5d": "ret_5d"})
    path = tmp_path / "labels.parquet"
    legacy.to_parquet(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="--legacy-labels"):
        runner.load_inputs(include_news=True)
    mapped = runner.load_inputs(include_news=True, legacy_labels=True)
    pd.testing.assert_frame_equal(mapped[1], tables["labels"])
    assert path.read_bytes() == before
    legacy.loc[0, "ret_1d"] = 1000
    legacy.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="exact future-session"):
        runner.load_inputs(include_news=True, legacy_labels=True)


def test_price_models_do_not_require_news(tables, monkeypatch, tmp_path):
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    for name in ("prices", "labels"):
        tables[name].to_parquet(tmp_path / f"{name}.parquet", index=False)
    inputs = runner.load_inputs(include_news=False)
    outputs, _ = call(inputs, fold_indices=[0])
    assert not outputs["buy_and_hold"].empty


@pytest.fixture
def combined_command(tables, tmp_path):
    """One public CLI invocation, with only temporary synthetic artifacts/configs."""
    for package in ("sklearn", "statsmodels", "xgboost", "torch", "torch_geometric"):
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"actual baseline fitting requires {package}")
    from src.data.build_labels import make_labels

    tables["labels"] = make_labels(tables["prices"])
    day = sorted(tables["prices"]["date"].unique())[40]
    nonmember = tables["prices"]["date"].eq(day) & tables["prices"]["ticker"].eq("BBB")
    tables["prices"].loc[nonmember, "in_index"] = False
    missing_label = tables["labels"]["date"].eq(day) & tables["labels"]["ticker"].eq("AAA")
    tables["labels"].loc[missing_label, "y_1d"] = pd.NA
    news = tables["news_features"]
    tables["news_features"] = news[~(news["date"].eq(day) & news["ticker"].eq("BBB"))]
    for name, table in tables.items():
        table.to_parquet(tmp_path / f"{name}.parquet", index=False)
    settings = {**SHORT, "period": {"start": "2021-01-01", "end": "2021-03-05"}}
    path = tmp_path / "walk_forward.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    command = [sys.executable, "-B", "-m", "src.eval.run_baselines", "--all", "--walk-forward-config", str(path)]
    for name in ("lstm_price", "lstm_news"):
        cfg = load_config(CONFIGS_DIR / f"{name}.yaml")
        cfg["features"]["embedding_dim"] = 4
        cfg["encoder"].update(hidden=4, dropout=0.0)
        cfg["train"].update(epochs=1)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        command += [f"--{name.replace('_', '-')}-config", str(path)]
    return command, {**os.environ, "NDX_DATA_DIR": str(tmp_path)}, tables, settings


def run_combined(command, environment):
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False, timeout=180)
    assert result.returncode == 0, f"combined CLI exit {result.returncode}\n{result.stdout}\n{result.stderr}"
    return result


def test_actual_public_all_cli_fits_all_six_and_both_horizons(combined_command, tmp_path):
    command, environment, tables, settings = combined_command
    before = {p: p.read_bytes() for p in tmp_path.glob("*.parquet")}
    result = run_combined(command, environment)
    from src.models.panel import build_panel
    from src.models.train import folds_for

    panel = build_panel(tables["prices"], tables["labels"], tables["news_features"], embeddings=False)
    fold = folds_for(panel, settings)[0]
    assert len(fold.train) == 35 and len(fold.test) == 5
    assert panel.days.get_loc(fold.test[0]) - panel.days.get_loc(fold.train[-1]) == 6
    expected_split = runner.fold_manifest([fold])[0]
    expected_keys = None
    for name in runner.BASELINE_NAMES:
        assert result.stdout.count(f"Starting isolated baseline: {name}\n") == 1
        output = read_table("predictions", tmp_path / "predictions" / f"{name}.parquet")
        validate(output, "predictions")
        assert np.isfinite(output["p_up"]).all()
        assert output["p_up"].between(0, 1).all()
        assert not output.duplicated(["model", "horizon", "date", "ticker"]).any()
        assert set(output["horizon"]) == {1, 5}
        assert set(output["model"]) == {name} and set(output["fold"]) == {0}
        for horizon in (1, 5):
            part = output[output["horizon"].eq(horizon)].reset_index(drop=True)
            expected = panel.rows(fold.test, horizon)
            pd.testing.assert_frame_equal(part[["date", "ticker"]], expected[["date", "ticker"]])
            np.testing.assert_array_equal(part["y_true"], expected["y"])
        assert len(output[output["horizon"].eq(1)]) == 28
        assert len(output[output["horizon"].eq(5)]) == 29
        record = json.loads((tmp_path / "predictions" / f"{name}_run.json").read_text())
        assert len(record["folds"]) == 2
        assert all(info["split"] == expected_split for info in record["folds"])
        rows = output[["date", "ticker", "horizon", "fold", "y_true"]]
        if expected_keys is None:
            expected_keys = rows
        pd.testing.assert_frame_equal(rows, expected_keys)
        if name in ("logistic", "xgboost"):
            assert all(info["training"]["fitted"] for info in record["folds"])
        if name == "arima":
            assert all(info["training"]["fitted_tickers"] > 0 for info in record["folds"])
        if name.startswith("lstm_"):
            assert all(info["epochs"] == 1 and info["edges_per_day"] == 0 for info in record["folds"])
    assert len(list((tmp_path / "predictions").iterdir())) == 12
    assert all(p.read_bytes() == content for p, content in before.items())


def test_combined_cli_future_changes_do_not_change_earlier_predictions(combined_command, tmp_path):
    command, environment, tables, _ = combined_command
    run_combined(command, environment)
    original = {
        name: read_table("predictions", tmp_path / "predictions" / f"{name}.parquet") for name in runner.BASELINE_NAMES
    }
    cutoff = original["buy_and_hold"]["date"].min()
    prices, labels, news = tables["prices"], tables["labels"], tables["news_features"]
    prices.loc[prices["date"] > cutoff, ["open", "high", "low", "close", "volume"]] *= 10
    for horizon in (1, 5):
        labels.loc[labels["date"] >= cutoff, f"y_{horizon}d"] = (
            1 - labels.loc[labels["date"] >= cutoff, f"y_{horizon}d"]
        )
    later_news = news["date"].gt(cutoff) & news["has_news"]
    embeddings = [c for c in news if c.startswith("emb_")]
    news.loc[later_news, embeddings] += 100
    for name, table in tables.items():
        table.to_parquet(tmp_path / f"{name}.parquet", index=False)
    run_combined(command, environment)
    for name in runner.BASELINE_NAMES:
        changed = read_table("predictions", tmp_path / "predictions" / f"{name}.parquet")
        old = original[name]
        np.testing.assert_array_equal(
            old.loc[old["date"].eq(cutoff), "p_up"], changed.loc[changed["date"].eq(cutoff), "p_up"]
        )


def test_combined_cli_preserves_multiple_folds(combined_command, tmp_path):
    command, environment, tables, settings = combined_command
    settings["period"]["end"] = "2021-03-12"
    (tmp_path / "walk_forward.json").write_text(json.dumps(settings), encoding="utf-8")
    run_combined(command, environment)
    from src.graph.comention_report import shared_folds

    splits = runner.fold_manifest(shared_folds(tables["prices"], settings))
    assert len(splits) == 2
    reference = None
    for name in runner.BASELINE_NAMES:
        output = read_table("predictions", tmp_path / "predictions" / f"{name}.parquet")
        assert set(output["fold"]) == {0, 1}
        record = json.loads((tmp_path / "predictions" / f"{name}_run.json").read_text())
        assert len(record["folds"]) == 4
        for info in record["folds"]:
            assert info["split"] == splits[info["fold"]]
        rows = output[["date", "ticker", "horizon", "fold", "y_true"]]
        if reference is None:
            reference = rows
        pd.testing.assert_frame_equal(rows, reference)


def test_combined_smoke_has_no_persistent_outputs(combined_command, tmp_path):
    command, environment, _, _ = combined_command
    result = run_combined([*command, "--folds", "0"], environment)
    assert "no persistent" in result.stdout
    assert not (tmp_path / "predictions").exists()


def test_combined_cli_runs_when_python_strips_docstrings(combined_command, tmp_path):
    command, environment, _, _ = combined_command
    result = run_combined([*command, "--folds", "0"], {**environment, "PYTHONOPTIMIZE": "2"})
    for name in runner.BASELINE_NAMES:
        assert f"{name}: horizon 1, fold 0" in result.stdout
        assert f"{name}: horizon 5, fold 0" in result.stdout
    assert not (tmp_path / "predictions").exists()


def test_combined_failure_does_not_publish_partial_results(combined_command, tmp_path):
    command, environment, _, _ = combined_command
    path = tmp_path / "lstm_news.json"
    cfg = json.loads(path.read_text())
    cfg["gat"]["edge_types"] = ["sector"]
    path.write_text(json.dumps(cfg), encoding="utf-8")
    directory = tmp_path / "predictions"
    directory.mkdir()
    existing = directory / "buy_and_hold.parquet"
    existing.write_bytes(b"existing output must remain unchanged")
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False, timeout=180)
    assert result.returncode == 1
    assert "lstm_news: worker failed" in result.stderr
    assert "not a lstm_news config" in result.stderr
    assert existing.read_bytes() == b"existing output must remain unchanged"
    assert list(directory.iterdir()) == [existing]


@pytest.mark.parametrize("worker_status,cli_status", [(7, 7), (-11, 139)])
def test_worker_failure_status_is_propagated(inputs, monkeypatch, worker_status, cli_status, capsys):
    monkeypatch.setattr(runner, "load_inputs", lambda **kwargs: inputs)
    monkeypatch.setattr(runner, "input_hashes", lambda include_news: {})
    monkeypatch.setattr(runner, "walk_forward_config", lambda: SHORT)
    monkeypatch.setattr(sys, "argv", ["run_baselines", "--only", "buy_and_hold"])
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, worker_status)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == cli_status
    assert len(calls) == 1 and "buy_and_hold" in capsys.readouterr().err


def test_worker_rejects_changed_fold_plan(inputs):
    with pytest.raises(ValueError, match="frozen fold definitions"):
        call(inputs, fold_indices=[0], expected_folds=[])


def test_parent_rejects_changed_inputs(inputs, monkeypatch):
    monkeypatch.setattr(runner, "input_hashes", lambda include_news: {"prices": "changed"})
    with pytest.raises(ValueError, match="changed while"):
        runner.execute_isolated(*inputs, ("buy_and_hold",), (1, 5), walk_forward=SHORT, reference_hashes={})
