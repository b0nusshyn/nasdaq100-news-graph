"""Task 4.10: validated comparisons, shared portfolio calculations, safe outputs."""

import copy
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.core.io import write_table
from src.core.schema import conform
from src.eval import compare
from src.eval.backtest import backtest
from src.eval.splitter import walk_forward


@pytest.fixture
def market():
    from src.data.build_index_history import trading_days
    from src.data.build_labels import make_labels

    dates = trading_days(pd.Timestamp("2021-01-04"), pd.Timestamp("2021-04-01"))[:50]
    rows = []
    for ticker, rate in zip(("AAA", "BBB", "CCC", "DDD"), (0.01, 0.02, -0.01, -0.02), strict=True):
        for index, date in enumerate(dates):
            close = 100 * (1 + rate) ** index
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 100.0,
                    "in_index": True,
                }
            )
    prices = conform(pd.DataFrame(rows), "prices")
    labels = make_labels(prices)
    folds = walk_forward(dates, train_days=10, test_days=6, horizon=5, embargo_days=5)
    assert len(folds) == 5
    return prices, labels, folds


def predictions(market, model="gat_lstm"):
    _, labels, folds = market
    parts, entries = [], []
    for horizon in (1, 5):
        for fold in folds:
            rows = labels.loc[labels.date.isin(fold.test)].copy()
            rows["model"] = model
            rows["horizon"] = horizon
            rows["fold"] = fold.index
            rows["y_true"] = rows[f"y_{horizon}d"]
            # Synthetic oracle scores: AAA is the top long, DDD the bottom short.
            rows["p_up"] = rows.ticker.map({"AAA": 0.9, "BBB": 0.8, "CCC": 0.2, "DDD": 0.1})
            parts.append(conform(rows, "predictions"))
            entries.append({"horizon": horizon, "fold": fold.index, "threshold": 0.5, "test_rows": len(rows)})
    record = {
        "model": model,
        "config": {"model": model, "seed": 42, "horizons": [1, 5]},
        "folds": entries,
        "git_commit": "synthetic",
        "ablate": [],
    }
    return pd.concat(parts, ignore_index=True), record


def saved_inputs(tmp_path, market, names=("gat_lstm", "lstm_news")):
    for name in names:
        frame, record = predictions(market, name)
        write_table(frame, "predictions", tmp_path / f"{name}.parquet")
        (tmp_path / f"{name}_run.json").write_text(json.dumps(record))
    return compare.load_predictions(tmp_path, market[0], market[1], market[2])


def tables(market, names=("gat_lstm",)):
    pairs = {name: predictions(market, name) for name in names}
    return compare.build_tables(
        {n: pair[0] for n, pair in pairs.items()},
        {n: pair[1] for n, pair in pairs.items()},
        market[1],
        market[0],
        market[2],
    )


def test_known_metrics_groups_and_fold_means(market):
    detail, summary, *_ = tables(market)
    assert len(detail) == 10 and len(summary) == 2
    assert set(detail.horizon) == {1, 5} and set(detail.fold) == set(range(5))
    assert detail.n_observations.eq(24).all()
    assert detail.within_day_auc_days.eq(6).all()
    for metric in compare.METRICS:
        assert detail[metric].eq(1).all()
        assert summary[f"mean_fold_{metric}"].eq(1).all()
    assert summary.n_observations.eq(120).all()
    assert summary.n_folds.eq(5).all()


def test_thresholds_are_recorded_not_test_selected(market):
    frame, record = predictions(market)
    for entry in record["folds"]:
        entry["threshold"] = 0.85
    detail, summary, *_ = compare.build_tables(
        {"gat_lstm": frame}, {"gat_lstm": record}, market[1], market[0], market[2]
    )
    # TP=1, FN=1, TN=2, FP=0 per day: MCC = 1/sqrt(3), accuracy = 3/4.
    np.testing.assert_allclose(detail.mcc, 1 / np.sqrt(3))
    assert detail.accuracy.eq(0.75).all()
    assert detail.threshold.eq(0.85).all()
    assert summary.mean_fold_within_day_auc.eq(1).all()


def test_portfolio_costs_and_continuous_five_day_phase(market):
    detail, summary, equity, *_ = tables(market)
    frame, _ = predictions(market)
    daily = summary.loc[summary.horizon.eq(1)].iloc[0]
    # Half capital long AAA (+1%) and half short DDD (-2%), less 10 bps round trip.
    assert daily.backtest_return == pytest.approx(1.014**30 - 1)
    assert daily.turnover == 60
    for horizon in (1, 5):
        expected = backtest(frame, market[1], horizon)
        row = summary.loc[summary.horizon.eq(horizon)].iloc[0]
        assert row.backtest_return == expected.cumulative_return
        assert row.sharpe == expected.sharpe
        assert row.max_drawdown == expected.max_drawdown
        assert row.turnover == expected.turnover
        curve = equity.loc[(equity.model == "gat_lstm") & (equity.horizon == horizon)]
        np.testing.assert_array_equal(curve.equity, expected.equity)
        assert curve.date.tolist() == expected.equity.index.tolist()
    assert summary.loc[summary.horizon.eq(5), "turnover"].item() == 12
    # Independent six-session fold diagnostics have two five-day trades each.
    assert detail.loc[detail.horizon.eq(5), "turnover"].sum() == 20
    free = compare.build_tables(
        {"gat_lstm": frame}, {"gat_lstm": predictions(market)[1]}, market[1], market[0], market[2], 0
    )[1]
    assert (free.backtest_return > summary.backtest_return).all()


def test_buy_and_hold_is_separate_long_only(market):
    with pytest.warns(UserWarning, match="MCC is undefined"):
        frame, record = predictions(market, "buy_and_hold")
        frame["p_up"] = 1.0
        detail, summary, equity, benchmark, _ = compare.build_tables(
            {"buy_and_hold": frame}, {"buy_and_hold": record}, market[1], market[0], market[2]
        )
    assert detail.backtest_return.isna().all() and summary.backtest_return.isna().all()
    assert summary.mean_fold_mcc.isna().all()
    assert summary.defined_folds_mcc.eq(0).all()
    assert set(equity.strategy) == {"long_only"}
    assert benchmark.status.eq("available").all()
    assert len(benchmark) == 12


def test_missing_long_only_holding_does_not_invent_equity(market):
    prices, labels, folds = market
    broken = prices.loc[~((prices.ticker == "AAA") & (prices.date == folds[0].test[1]))]
    frame, record = predictions(market)
    _, _, equity, benchmark, warnings = compare.build_tables(
        {"gat_lstm": frame}, {"gat_lstm": record}, labels, broken, folds
    )
    assert not equity.model.eq("buy_and_hold_long_only").any()
    failures = benchmark[benchmark.status.eq("unavailable")]
    assert failures.backtest_return.isna().all() and len(failures) >= 2
    assert any("AAA" in warning for warning in warnings)


def test_load_exact_rows_and_metadata(tmp_path, market):
    frames, records, sources = saved_inputs(tmp_path, market)
    assert len(sources) == 4
    assert set(frames) == set(records) == {"gat_lstm", "lstm_news"}
    for source in sources:
        assert len(source["sha256"]) == 64
    pd.testing.assert_frame_equal(frames["gat_lstm"][compare.KEYS], frames["lstm_news"][compare.KEYS])


@pytest.mark.parametrize(
    "problem",
    [
        "missing_row",
        "duplicate",
        "wrong_label",
        "bad_model",
        "bad_horizon",
        "nan",
        "infinity",
        "extra_column",
        "wrong_dtype",
        "missing_fold",
    ],
)
def test_bad_prediction_artifacts_fail(tmp_path, market, problem):
    saved_inputs(tmp_path, market)
    path = tmp_path / "gat_lstm.parquet"
    frame, _ = predictions(market)
    if problem == "missing_row":
        frame = frame.iloc[1:]
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "wrong_label":
        frame.loc[0, "y_true"] = 1 - frame.loc[0, "y_true"]
    elif problem == "bad_model":
        frame["model"] = "other"
    elif problem == "bad_horizon":
        frame.loc[0, "horizon"] = 2
    elif problem in ("nan", "infinity"):
        frame.loc[0, "p_up"] = np.nan if problem == "nan" else np.inf
    elif problem == "extra_column":
        frame["future_return"] = 1.0
    elif problem == "wrong_dtype":
        frame["y_true"] = frame.y_true.astype(float) + 0.1
    else:
        frame = frame[frame.fold != 4]
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        compare.load_predictions(tmp_path, market[0], market[1], market[2])


@pytest.mark.parametrize(
    "problem", ["absent", "duplicate", "threshold", "missing_threshold", "coverage", "stale", "wrong_model"]
)
def test_bad_run_records_fail(tmp_path, market, problem):
    saved_inputs(tmp_path, market)
    path = tmp_path / "gat_lstm_run.json"
    record = json.loads(path.read_text())
    if problem == "absent":
        path.unlink()
    elif problem == "duplicate":
        record["folds"].append(record["folds"][0])
    elif problem == "threshold":
        record["folds"][0]["threshold"] = float("nan")
    elif problem == "missing_threshold":
        del record["folds"][0]["threshold"]
    elif problem == "coverage":
        record["folds"].pop()
    elif problem == "stale":
        record["folds"][0]["test_rows"] = 1
    else:
        record["model"] = "other"
    if problem != "absent":
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        compare.load_predictions(tmp_path, market[0], market[1], market[2])


def test_historical_membership_and_missing_labels(tmp_path, market):
    saved_inputs(tmp_path, market)
    prices, labels, folds = market
    prices.loc[(prices.date == folds[0].test[0]) & (prices.ticker == "AAA"), "in_index"] = False
    with pytest.raises(ValueError, match="historical-member"):
        compare.load_predictions(tmp_path, prices, labels, folds)
    prices["in_index"] = True
    labels.loc[(labels.date == folds[0].test[0]) & (labels.ticker == "AAA"), "y_1d"] = pd.NA
    with pytest.raises(ValueError, match="historical-member"):
        compare.load_predictions(tmp_path, prices, labels, folds)


def test_no_artifacts_is_explicit(tmp_path, market):
    with pytest.raises(ValueError, match="no prediction artifacts"):
        compare.load_predictions(tmp_path, *market)


def test_row_order_invariance_and_input_preservation(market):
    frame, record = predictions(market)
    before = copy.deepcopy(record)
    expected = compare.build_tables({"gat_lstm": frame}, {"gat_lstm": record}, market[1], market[0], market[2])
    shuffled = compare.build_tables(
        {"gat_lstm": frame.sample(frac=1, random_state=3)}, {"gat_lstm": record}, market[1], market[0], market[2]
    )
    for left, right in zip(expected[:4], shuffled[:4], strict=True):
        pd.testing.assert_frame_equal(left, right)
    assert record == before


def test_exclusive_publication_and_undefined_values(tmp_path, market):
    result = tables(market)
    manifest = {"cost_bps_per_side": 5, "missing_models": ["arima"]}
    outputs = compare.publish_outputs(tmp_path, result, manifest, figures=False)
    assert len(outputs) == 5
    before = {p: p.read_bytes() for p in outputs}
    with pytest.raises(ValueError, match="refusing to overwrite"):
        compare.publish_outputs(tmp_path, result, manifest, figures=False)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert json.loads((tmp_path / compare.TABLE_FILES[-1]).read_text())["missing_models"] == ["arima"]


def test_deterministic_csv_files(tmp_path, market):
    result = tables(market)
    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        compare.publish_outputs(directory, result, {"cost_bps_per_side": 5}, figures=False)
    for name in compare.TABLE_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_figure_failure_does_not_publish_tables(tmp_path, market, monkeypatch):
    monkeypatch.setattr(compare.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="Rscript"):
        compare.publish_outputs(tmp_path, tables(market), {"cost_bps_per_side": 5})
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(
    shutil.which("Rscript") is None, reason="figure integration requires installed Rscript PNG graphics"
)
def test_actual_figures_are_valid_and_deterministic(tmp_path, market):
    result = tables(market, ("gat_lstm", "lstm_news"))
    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        outputs = compare.publish_outputs(directory, result, {"cost_bps_per_side": 5})
        assert len(outputs) == 9
        for name in compare.FIGURE_FILES:
            content = (directory / name).read_bytes()
            assert content[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack(">II", content[16:24]) == (3600, 2250)
            assert len(content) > 10000
    for name in compare.FIGURE_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.parametrize("flag", [[], ["-OO"]])
def test_help_with_light_dependencies_and_no_docstrings(flag):
    result = subprocess.run(
        [sys.executable, *flag, "-m", "src.eval.compare", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "--legacy-labels" in result.stdout and "--tables-only" in result.stdout


@pytest.mark.parametrize("cost", [-1, np.nan, np.inf, 10000])
def test_invalid_cost(market, cost):
    frame, record = predictions(market)
    with pytest.raises(ValueError, match="cost_bps_per_side"):
        compare.build_tables({"gat_lstm": frame}, {"gat_lstm": record}, market[1], market[0], market[2], cost)


def test_summary_is_unweighted_and_does_not_skip_undefined_folds(market):
    frame, record = predictions(market)
    # Fewer rows in fold 0 and MCC=0 there; the other four folds still have MCC=1.
    first_date = frame.loc[frame.fold == 0, "date"].min()
    frame = frame.loc[~((frame.fold == 0) & (frame.date == first_date))].copy()
    selected = frame.fold == 0
    frame.loc[selected, "p_up"] = (
        frame.loc[selected, "ticker"].map({"AAA": 0.9, "BBB": 0.1, "CCC": 0.8, "DDD": 0.2}).astype("float32")
    )
    result = compare.build_tables({"gat_lstm": frame}, {"gat_lstm": record}, market[1], market[0], market[2])
    np.testing.assert_allclose(result[1].mean_fold_mcc, 0.8)
    assert result[0].loc[result[0].fold.eq(0), "n_observations"].eq(20).all()
    frame.loc[selected, "p_up"] = 1.0
    with pytest.warns(UserWarning, match="MCC is undefined"):
        result = compare.build_tables({"gat_lstm": frame}, {"gat_lstm": record}, market[1], market[0], market[2])
    assert result[1].mean_fold_mcc.isna().all()
    assert result[1].defined_folds_mcc.eq(4).all()


def test_verified_legacy_mapping_is_in_memory_only(tmp_path, market, monkeypatch):
    prices, labels, _ = market
    write_table(prices, "prices", tmp_path / "prices.parquet")
    path = tmp_path / "labels.parquet"
    labels.rename(columns={"fwd_ret_1d": "ret_1d", "fwd_ret_5d": "ret_5d"}).to_parquet(path, index=False)
    before = path.read_bytes()
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="legacy-labels"):
        compare.load_market_data()
    actual_prices, actual_labels = compare.load_market_data(legacy_labels=True)
    pd.testing.assert_frame_equal(actual_prices, prices)
    pd.testing.assert_frame_equal(actual_labels, labels)
    assert path.read_bytes() == before
    invalid = pd.read_parquet(path)
    invalid.loc[0, "ret_1d"] = 0.5
    invalid.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="exact future-session closes"):
        compare.load_market_data(legacy_labels=True)


def test_zero_exit_graphics_without_files_is_not_success(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(compare.shutil, "which", lambda _: "Rscript")
    monkeypatch.setattr(
        compare.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr="failed device")
    )
    with pytest.raises(ValueError, match="did not produce"):
        compare.draw_figures(tmp_path, 5)


def test_copy_failure_rolls_back_only_new_outputs(tmp_path, market, monkeypatch):
    from pathlib import Path

    real_open = Path.open

    def fail_third_file(path, *args, **kwargs):
        if path == tmp_path / compare.TABLE_FILES[2]:
            raise OSError("simulated copy failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_third_file)
    with pytest.raises(OSError, match="copy failure"):
        compare.publish_outputs(tmp_path, tables(market), {"cost_bps_per_side": 5}, figures=False)
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_run_groups_disclose_configuration_differences(market):
    _, record = predictions(market)
    from src.core.config import load_config

    record["config"] = load_config("gat_lstm.yaml")
    assert compare.run_group("gat_lstm", record) == "current_T1_seed42_main"
    record["config"]["seed"] = 43
    assert compare.run_group("gat_lstm", record) == "other_recorded_configuration"
    record["ablate"] = ["corr"]
    assert compare.run_group("gat_lstm_no_corr", record) == "ablation_separate_provenance"


def test_conflicting_recorded_thresholds_fail(tmp_path, market):
    saved_inputs(tmp_path, market)
    path = tmp_path / "gat_lstm_run.json"
    record = json.loads(path.read_text())
    record["folds"][0]["test"] = {"threshold": 0.9}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="inconsistent validation threshold"):
        compare.load_predictions(tmp_path, market[0], market[1], market[2])


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="graphics failure regression requires Rscript")
def test_failed_png_device_cannot_create_default_pdf(tmp_path, market, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Reproduce R's warning-only PNG failure: no device opens, exit is otherwise zero.
    monkeypatch.setattr(
        compare,
        "PLOT_SCRIPT",
        'png <- function(...) warning("simulated unavailable PNG device")\n' + compare.PLOT_SCRIPT,
    )
    with pytest.raises(ValueError, match="PNG device did not open"):
        compare.publish_outputs(tmp_path / "outputs", tables(market), {"cost_bps_per_side": 5})
    assert not (tmp_path / "Rplots.pdf").exists()
    assert not (tmp_path / "outputs").exists()
