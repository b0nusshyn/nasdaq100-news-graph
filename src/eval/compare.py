"""Validated tables and figures from existing predictions. Evaluation, Task 4.10 (#45).

    python -m src.eval.compare --output results --legacy-labels

No training or threshold selection occurs. Missing model files are disclosed;
malformed available files fail. Main runs and ablations retain separate provenance.
Figures use installed Rscript/base graphics (no R packages); --tables-only needs
only the project's light Python dependencies. Existing outputs are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.core.config import load_config, walk_forward_config
from src.core.io import read_table
from src.core.paths import RESULTS_DIR, data_dir, table_path
from src.core.schema import PREDICTIONS, validate
from src.eval.backtest import backtest
from src.eval.buy_and_hold import buy_and_hold_benchmark
from src.eval.metrics import classification_metrics
from src.models.ladder_report import within_day_auc

EXPECTED_MODELS = ("buy_and_hold", "arima", "logistic", "xgboost", "lstm_price", "lstm_news", "gat_lstm")
KEYS = ["horizon", "fold", "date", "ticker"]
METRICS = ("accuracy", "f1", "mcc", "auc", "within_day_auc")
TABLE_FILES = (
    "tables/metrics_by_fold.csv",
    "tables/model_comparison.csv",
    "tables/equity_curves.csv",
    "tables/buy_and_hold_benchmark.csv",
    "tables/comparison_manifest.json",
)
FIGURE_FILES = tuple(
    f"figures/{kind}_{horizon}d.png" for horizon in (1, 5) for kind in ("equity_curves", "metrics_by_fold")
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_market_data(legacy_labels: bool = False) -> tuple:
    """Reuse Task 4.8's exact-session verification for opt-in legacy labels."""
    from src.eval.run_baselines import load_inputs

    prices, labels, _ = load_inputs(include_news=False, legacy_labels=legacy_labels)
    return prices, labels


def load_predictions(directory: Path, prices: pd.DataFrame, labels: pd.DataFrame, folds: list) -> tuple:
    """Require exact shared eligible rows and thresholds for every available model."""
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise ValueError(f"no prediction artifacts in {directory}")
    frames, records, sources = {}, {}, []
    fold_ids = {fold.index for fold in folds}
    if len(fold_ids) != len(folds) or not folds:
        raise ValueError("fold definitions must be nonempty with unique identifiers")
    coverage = {(h, f) for h in (1, 5) for f in fold_ids}
    eligible = prices.loc[prices.in_index, ["date", "ticker"]].merge(
        labels, on=["date", "ticker"], validate="one_to_one"
    )

    for path in paths:
        name = path.stem
        record_path = path.with_name(f"{name}_run.json")
        if not record_path.is_file():
            raise ValueError(f"{name}: missing run record {record_path}")
        if set(pq.read_schema(path).names) != set(PREDICTIONS.column_names):
            raise ValueError(f"{name}: columns differ from official prediction schema")
        frame = read_table("predictions", path, columns=PREDICTIONS.column_names)
        validate(frame, PREDICTIONS)
        if frame.empty or set(frame.model) != {name} or not frame.y_true.isin([0, 1]).all():
            raise ValueError(f"{name}: invalid model name or realised labels")
        if not np.isfinite(frame.p_up).all():
            raise ValueError(f"{name}: probabilities must be finite")
        if set(frame[["horizon", "fold"]].itertuples(index=False, name=None)) != coverage:
            raise ValueError(f"{name}: require both horizons and all shared folds")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        _validate_record(record, name, frame, coverage)

        for fold in folds:
            for horizon in (1, 5):
                observed = frame.loc[(frame.fold == fold.index) & (frame.horizon == horizon)]
                expected = eligible.loc[eligible.date.isin(fold.test) & eligible[f"y_{horizon}d"].notna()]
                observed = observed.sort_values(["date", "ticker"]).reset_index(drop=True)
                expected = expected.sort_values(["date", "ticker"]).reset_index(drop=True)
                if not observed[["date", "ticker"]].equals(expected[["date", "ticker"]]):
                    raise ValueError(
                        f"{name}: keys differ from shared historical-member rows, h={horizon} fold={fold.index}"
                    )
                if not np.array_equal(observed.y_true, expected[f"y_{horizon}d"]):
                    raise ValueError(f"{name}: conflicting realised labels")
        frames[name] = frame.sort_values(KEYS).reset_index(drop=True)
        records[name] = record
        sources.extend({"path": str(p), "sha256": sha256(p)} for p in (path, record_path))

    reference = next(iter(frames.values()))
    for name, frame in frames.items():
        if not frame[KEYS + ["y_true"]].equals(reference[KEYS + ["y_true"]]):
            raise ValueError(f"{name}: unmatched model evaluation population")
    return frames, records, sources


def _validate_record(record: dict, name: str, frame: pd.DataFrame, coverage: set) -> None:
    if not isinstance(record, dict) or record.get("model") != name or not isinstance(record.get("config"), dict):
        raise ValueError(f"{name}: missing model/config run provenance")
    entries = record.get("folds")
    if not isinstance(entries, list):
        raise ValueError(f"{name}: missing run folds")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{name}: invalid run fold")
        horizon, fold = entry.get("horizon"), entry.get("fold")
        if type(horizon) is not int or type(fold) is not int or (horizon, fold) in seen:
            raise ValueError(f"{name}: invalid or duplicate run key")
        threshold = entry.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f"{name}: missing or invalid validation-selected threshold")
        rows = frame.loc[(frame.horizon == horizon) & (frame.fold == fold)]
        if "test_rows" in entry and entry["test_rows"] != len(rows):
            raise ValueError(f"{name}: stale run test_rows")
        if "test" in entry and entry["test"].get("threshold", threshold) != threshold:
            raise ValueError(f"{name}: inconsistent validation threshold in run record")
        seen.add((horizon, fold))
    if seen != coverage:
        raise ValueError(f"{name}: incomplete run coverage")


def run_group(name: str, record: dict) -> str:
    """Label recorded configuration matches, rather than infer tuning from scores."""
    if name in ("gat_lstm", "lstm_news", "lstm_price"):
        if record["config"] == load_config(f"{name}.yaml"):
            return "current_T1_seed42_main"
    if record.get("ablate"):
        return "ablation_separate_provenance"
    return "other_recorded_configuration"


def _classification(rows: pd.DataFrame, threshold: float) -> dict:
    scores = classification_metrics(rows.y_true.to_numpy(), rows.p_up.to_numpy(), threshold)
    valid_dates = rows.groupby("date").y_true.nunique().eq(2)
    # Within-day AUC is undefined on single-class days, as in the model report.
    scores["within_day_auc_days"] = int(valid_dates.sum())
    scores["within_day_auc"] = within_day_auc(rows) if valid_dates.any() else float("nan")
    return scores


def _portfolio_fields(result) -> dict:
    return {
        "backtest_return": result.cumulative_return,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "turnover": result.turnover,
    }


def _equity_rows(result, model: str, horizon: int, strategy: str) -> list:
    return [
        {"model": model, "horizon": horizon, "strategy": strategy, "date": date, "equity": float(value)}
        for date, value in result.equity.items()
    ]


def build_tables(
    frames: dict, records: dict, labels: pd.DataFrame, prices: pd.DataFrame, folds: list, cost_bps_per_side: float = 5.0
) -> tuple:
    """Score folds separately; compute aggregate portfolio statistics continuously.

    Per-fold portfolios restart independently and can exit after the fold's last
    signal. Never concatenate them: continuous 5d results use one global phase.
    Buy-and-Hold is a separate initial-member basket, not ranked predictions.
    """
    if not np.isfinite(cost_bps_per_side) or not 0 <= cost_bps_per_side < 10_000:
        raise ValueError("cost_bps_per_side must be finite and in [0, 10000)")
    fold_rows, summary_rows, equity_rows, benchmark_rows = [], [], [], []
    warnings = []
    calendar = pd.DatetimeIndex(labels.date.unique()).sort_values()
    for name, frame in frames.items():
        thresholds = {(e["horizon"], e["fold"]): e["threshold"] for e in records[name]["folds"]}
        group = run_group(name, records[name])
        for (horizon, fold), rows in frame.groupby(["horizon", "fold"], sort=True):
            threshold = thresholds[horizon, fold]
            row = {
                "model": name,
                "run_group": group,
                "horizon": int(horizon),
                "fold": int(fold),
                "n_observations": len(rows),
                "threshold": threshold,
                **_classification(rows, threshold),
            }
            if name == "buy_and_hold":
                row.update(dict.fromkeys(("backtest_return", "sharpe", "max_drawdown", "turnover"), np.nan))
            else:
                row.update(_portfolio_fields(backtest(rows, labels, int(horizon), cost_bps_per_side=cost_bps_per_side)))
            fold_rows.append(row)

        for horizon, rows in frame.groupby("horizon", sort=True):
            details = [r for r in fold_rows if r["model"] == name and r["horizon"] == horizon]
            summary = {
                "model": name,
                "run_group": group,
                "horizon": int(horizon),
                "n_folds": len(details),
                "n_observations": len(rows),
            }
            for metric in METRICS:
                values = np.array([r[metric] for r in details])
                summary[f"defined_folds_{metric}"] = int(np.isfinite(values).sum())
                # Do not silently average a smaller set of defined folds.
                summary[f"mean_fold_{metric}"] = float(values.mean())
            if name == "buy_and_hold":
                summary.update(dict.fromkeys(("backtest_return", "sharpe", "max_drawdown", "turnover"), np.nan))
            else:
                result = backtest(rows, labels, int(horizon), cost_bps_per_side=cost_bps_per_side)
                summary.update(_portfolio_fields(result))
                equity_rows.extend(_equity_rows(result, name, int(horizon), "long_short"))
            summary_rows.append(summary)

    reference = next(iter(frames.values()))
    for horizon in (1, 5):
        rows = reference.loc[reference.horizon == horizon]
        first, last = calendar.get_loc(rows.date.min()), calendar.get_loc(rows.date.max())
        # Follow the existing backtest's global entry/exit calendar exactly.
        final_exit = first + ((last - first) // horizon + 1) * horizon
        if final_exit >= len(calendar):
            raise ValueError("labels calendar does not include the final portfolio exit")
        for fold_id, start, end in [(-1, calendar[first], calendar[final_exit])] + [
            (fold.index, fold.test[0], fold.test[-1]) for fold in folds
        ]:
            days = calendar[(calendar >= start) & (calendar <= end)]
            row = {
                "model": "buy_and_hold_long_only",
                "horizon": horizon,
                "fold": fold_id,
                "strategy": "long_only",
                "status": "available",
                "reason": "",
            }
            try:
                benchmark = buy_and_hold_benchmark(prices, days, cost_bps_per_side=cost_bps_per_side)
            except ValueError as error:
                row.update(status="unavailable", reason=str(error))
                row.update(dict.fromkeys(("backtest_return", "sharpe", "max_drawdown", "turnover"), np.nan))
                warnings.append(f"Buy-and-Hold h={horizon}, fold={fold_id}: {error}")
            else:
                row.update(_portfolio_fields(benchmark))
                if fold_id == -1:
                    equity_rows.extend(_equity_rows(benchmark, "buy_and_hold_long_only", horizon, "long_only"))
            benchmark_rows.append(row)
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(equity_rows),
        pd.DataFrame(benchmark_rows),
        warnings,
    )


# Standard R graphics are available on the local evaluation machine. No R packages
# are needed. Input CSVs are produced by Python; R only draws the recorded values.
PLOT_SCRIPT = r"""
args <- commandArgs(trailingOnly=TRUE)
folder <- args[1]
d <- read.csv(file.path(folder, "tables/metrics_by_fold.csv"), na.strings="NA")
e <- read.csv(file.path(folder, "tables/equity_curves.csv"), na.strings="NA")
cost <- as.numeric(args[2])
models <- sort(unique(c(as.character(d$model), as.character(e$model))))
colors <- setNames(hcl.colors(length(models), "Dark 3"), models)
styles <- setNames(ifelse(grepl("_no_", models), 2, 1), models)
styles["buy_and_hold_long_only"] <- 3
open_plot <- function(path) {
  png(path, width=3600, height=2250, res=300,
      type=if (capabilities("aqua")) "quartz" else "cairo")
  if (dev.cur()==1L) stop("PNG device did not open; no default graphics output allowed")
  par(mar=c(4, 4.5, 3, 1), oma=c(1.5, 0, 1.5, 0), family="sans", las=1)
}
legend_models <- function(names) {
  par(mar=c(0, 0, 0, 0))
  plot.new()
  legend("center", legend=gsub("_", " ", names),
         col=colors[names], lty=styles[names], ncol=3, cex=0.80, bty="n")
  mtext("Available artifacts only; see manifest for missing models, benchmark failures and run provenance.",
        side=1, outer=TRUE, cex=0.70)
}
for (h in c(1, 5)) {
  eq <- e[e$horizon == h, ]; eq$date <- as.Date(eq$date)
  eq_models <- sort(unique(eq$model))
  open_plot(file.path(folder, sprintf("figures/equity_curves_%dd.png", h)))
  layout(matrix(c(1, 2), nrow=2), heights=c(5, 1.1))
  limits <- range(c(1, 1 + eq$equity), finite=TRUE)
  if (diff(limits)==0) limits <- limits + c(-0.01, 0.01)
  plot(range(eq$date), limits, type="n", xlab="Settlement date", ylab="Wealth (initial capital = 1)",
       main=sprintf("%d-day horizon: continuous portfolio equity", h))
  abline(h=1, col="grey75", lty=3)
  for (model in eq_models) {
    rows <- eq[eq$model==model, ]; rows <- rows[order(rows$date), ]
    lines(rows$date, 1 + rows$equity, col=colors[model], lty=styles[model], lwd=1.5)
  }
  legend_models(eq_models)
  mtext(sprintf("%.1f bps per side; %dd non-overlapping holds; idealized execution; ablations dashed", cost, h),
        side=3, outer=TRUE, cex=0.75)
  dev.off()
  scores <- d[d$horizon==h, ]
  score_models <- sort(unique(scores$model))
  open_plot(file.path(folder, sprintf("figures/metrics_by_fold_%dd.png", h)))
  layout(matrix(c(1, 2, 3, 3), nrow=2, byrow=TRUE), heights=c(5, 1.1))
  par(oma=c(1.5, 0, 2, 0))
  for (metric in c("within_day_auc", "mcc")) {
    limits <- range(c(if (metric=="mcc") 0 else 0.5, scores[[metric]]), finite=TRUE)
    if (diff(limits)==0) limits <- limits + c(-0.01, 0.01)
    padding <- max(diff(limits)*0.10, 0.005)
    plot(range(scores$fold), limits+c(-padding, padding), type="n", xaxt="n", xlab="Walk-forward fold",
         ylab=if (metric=="mcc") "MCC (supporting metric)" else "Within-day AUC (primary metric)",
         main=if (metric=="mcc") "Validation-threshold MCC" else "Stock selection")
    axis(1, at=sort(unique(scores$fold)))
    abline(h=if (metric=="mcc") 0 else 0.5, col="grey65", lty=3)
    for (model in score_models) {
      rows <- scores[scores$model==model, ]; rows <- rows[order(rows$fold), ]
      lines(rows$fold, rows[[metric]], type="o", pch=16, cex=0.5,
            col=colors[model], lty=styles[model], lwd=1.3)
    }
  }
  legend_models(score_models)
  mtext(sprintf("%d-day horizon; observed axis range; ablations dashed; no significance inferred from lines", h),
        side=3, outer=TRUE, cex=0.8)
  dev.off()
}
"""


def draw_figures(directory: Path, cost: float) -> str:
    executable = shutil.which("Rscript")
    if executable is None:
        raise ValueError("figures require Rscript with PNG graphics; use --tables-only for tables")
    completed = subprocess.run(
        [executable, "--vanilla", "-e", PLOT_SCRIPT, str(directory), str(cost)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise ValueError(f"figure generation failed (exit {completed.returncode}): {completed.stderr}")
    for name in FIGURE_FILES:
        path = directory / name
        if not path.is_file() or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"graphics device did not produce {name}: {completed.stderr}")
    if completed.stderr:
        raise ValueError(f"graphics runtime reported warnings; outputs withheld: {completed.stderr}")
    version = subprocess.run([executable, "--version"], capture_output=True, text=True, check=True)
    return (version.stdout + version.stderr).strip()


def publish_outputs(output: Path, tables: tuple, manifest: dict, *, figures: bool = True) -> list[Path]:
    """Stage every artifact before exclusive creation; roll back only our new files."""
    names = TABLE_FILES + (FIGURE_FILES if figures else ())
    existing = [str(output / name) for name in names if (output / name).exists()]
    if existing:
        raise ValueError("refusing to overwrite existing results: " + ", ".join(existing))
    if figures and shutil.which("Rscript") is None:
        raise ValueError("figures require Rscript with PNG graphics; use --tables-only for tables")
    with tempfile.TemporaryDirectory(prefix="task410-") as temporary:
        stage = Path(temporary)
        (stage / "tables").mkdir()
        (stage / "figures").mkdir()
        for frame, name in zip(tables[:4], TABLE_FILES[:4], strict=True):
            frame.to_csv(stage / name, index=False, na_rep="NA", float_format="%.12g")
        metadata = dict(manifest)
        metadata["graphics_backend"] = (
            draw_figures(stage, manifest["cost_bps_per_side"]) if figures else "not requested"
        )
        (stage / TABLE_FILES[-1]).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        created = []
        try:
            for name in names:
                destination = output / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    created.append(destination)
                    with (stage / name).open("rb") as source:
                        shutil.copyfileobj(source, handle)
        except OSError:
            for path in created:
                path.unlink()
            raise
    return created


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validated production comparison tables and figures (no training)")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--legacy-labels", action="store_true", help="verify legacy forward returns before in-memory mapping"
    )
    parser.add_argument("--tables-only", action="store_true", help="omit figures; no graphics runtime needed")
    parser.add_argument("--cost-bps-per-side", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        prices, labels = load_market_data(args.legacy_labels)
        from src.graph.comention_report import shared_folds

        folds = shared_folds(prices)
        if len(folds) != 5:
            raise ValueError("production comparison requires all five official folds")
        frames, records, sources = load_predictions(data_dir() / "predictions", prices, labels, folds)
        tables = build_tables(frames, records, labels, prices, folds, args.cost_bps_per_side)
        missing = sorted(set(EXPECTED_MODELS) - set(frames))
        significance = RESULTS_DIR / "tables/significance_mcc.json"
        settings = walk_forward_config()
        settings["period"] = {key: str(value) for key, value in settings["period"].items()}
        manifest = {
            "task": "4.10",
            "analysis_source": {"path": "src/eval/compare.py", "sha256": sha256(Path(__file__))},
            "complete_baseline_comparison": not missing and not tables[4],
            "included_models": sorted(frames),
            "missing_models": missing,
            "primary_metric": "within_day_auc",
            "supporting_metric": "mcc",
            "cost_bps_per_side": args.cost_bps_per_side,
            "long_short_quantile": 0.2,
            "walk_forward": settings,
            "sources": sources
            + [{"path": str(table_path(name)), "sha256": sha256(table_path(name))} for name in ("prices", "labels")],
            "runs": {
                name: {
                    key: r.get(key)
                    for key in ("model", "config", "ablate", "git_commit", "git_dirty", "written_at_utc")
                }
                for name, r in records.items()
            },
            "significance_artifact": {"path": str(significance), "sha256": sha256(significance)}
            if significance.is_file()
            else None,
            "legacy_label_mapping_verified": args.legacy_labels,
            "unavailable_benchmarks": tables[4],
            "notes": [
                "Missing predictions are not generated. Available malformed or unmatched artifacts fail validation.",
                (
                    "Main T1 runs and ablations with different commits are labelled separately; no "
                    "matched-run claim across commits."
                ),
                (
                    "Classification summary is unweighted fold means, not pooled scores; any "
                    "undefined fold makes its mean NA."
                ),
                (
                    "Within-day AUC excludes single-class dates only for that undefined daily "
                    "metric; eligible day counts are recorded."
                ),
                (
                    "Portfolio summary and equity use one continuous backtest per model/horizon, "
                    "not averages of fold portfolios."
                ),
                "Per-fold long/short diagnostics restart independently; exits may extend beyond the last fold signal.",
                (
                    "5d holds are non-overlapping globally; top/bottom 20%, half capital each "
                    "side, round-trip cost on each trade."
                ),
                "Turnover follows shared backtest units: sum of round-trip gross exposure, not a daily average.",
                "Buy-and-Hold is a separate long-only initial-member basket; removals become cash, no later entrants.",
                (
                    "Long-only fold windows end on the final fold session; the continuous window "
                    "ends at the strategy's final exit."
                ),
                (
                    "Idealized same-close execution; no slippage, borrow fees, liquidity "
                    "constraints or delisting recovery assumptions."
                ),
                (
                    "NA denotes unavailable/undefined, never zero. Graphics show observed axis "
                    "ranges and do not establish significance."
                ),
                (
                    "Training-input hashes are not available in the supplied modelling records; "
                    "evaluation input hashes are recorded here."
                ),
            ],
        }
        outputs = publish_outputs(args.output, tables, manifest, figures=not args.tables_only)
        print("Included: " + ", ".join(sorted(frames)))
        print("Missing predictions: " + (", ".join(missing) or "none"))
        for warning in tables[4]:
            print("Unavailable benchmark: " + warning)
        for path in outputs:
            print(f"Wrote {path}")
    except (ValueError, OSError) as error:
        parser.exit(1, f"compare: {error}\n")


if __name__ == "__main__":
    main()
