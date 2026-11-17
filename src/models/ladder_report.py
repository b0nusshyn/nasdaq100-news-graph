"""Walk-forward ladder report: does each rung add signal, at each horizon? Owner: Modelling. Issue #36 (3.8).

Reads `predictions/<model>.parquet` for the LSTM ladder (5 `lstm_price` -> 6 `lstm_news` ->
proposed `gat_lstm`) and any classical baselines present, and reports AUC per fold and horizon,
the two steps that answer the research question (news: 5 -> 6; graph: 6 -> proposed), and an
explicit **horizon-pattern** flag: propagation takes time, so the graph should add nothing at
1 day and something at 5 days. Writes `results/tables/ladder_3_8.md`.

    python -m src.models.ladder_report --output results/tables
    python -m src.models.ladder_report --seeds 43 44 --classical   # seed spread; baselines 1-4 in memory

AUC is the headline (no threshold). Fold numbers come from one seed unless `--seeds` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LADDER = ("lstm_price", "lstm_news", "gat_lstm")  # Baseline 5, Baseline 6, proposed
CLASSICAL = ("buy_and_hold", "arima", "logistic", "xgboost")
LABEL = {"lstm_price": "B5 LSTM price", "lstm_news": "B6 LSTM + news", "gat_lstm": "proposed GAT + LSTM",
         "buy_and_hold": "B1 buy-and-hold", "arima": "B2 ARIMA", "logistic": "B3 logistic", "xgboost": "B4 XGBoost"}


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC with average ranks for ties (as `src.eval.metrics`), NaN for one class."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if not n_pos or not n_neg:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """AUC per (horizon, fold) and model, plus `mean` (of folds) and `pooled` (all test rows) rows."""
    rows = {}
    for name, df in preds.items():
        for (h, f), g in df.groupby(["horizon", "fold"]):
            rows.setdefault((int(h), str(f)), {})[name] = auc(g["y_true"], g["p_up"])
        for h, g in df.groupby("horizon"):
            rows.setdefault((int(h), "pooled"), {})[name] = auc(g["y_true"], g["p_up"])
    t = pd.DataFrame.from_dict(rows, orient="index").rename_axis(["horizon", "fold"])
    folds = t[~t.index.get_level_values("fold").isin(["pooled"])]
    mean = folds.groupby(level="horizon").mean()
    mean.index = pd.MultiIndex.from_product([mean.index, ["mean"]], names=["horizon", "fold"])
    order = [c for c in (*CLASSICAL, *LADDER) if c in t.columns]
    t = pd.concat([folds, mean, t[t.index.get_level_values("fold") == "pooled"]])[order]
    return t.sort_index(level="horizon", sort_remaining=False)


def steps(t: pd.DataFrame) -> pd.DataFrame:
    """The two research-question steps per row of `auc_table`: news (6 - 5) and graph (proposed - 6)."""
    out = pd.DataFrame({"news (B6 - B5)": t["lstm_news"] - t["lstm_price"],
                        "graph (proposed - B6)": t["gat_lstm"] - t["lstm_news"]})
    out["monotone"] = (out > 0).all(axis=1)
    return out


def horizon_pattern(s: pd.DataFrame) -> list[str]:
    """The flag: does the graph step stay near zero at 1 day and turn positive at 5 days?

    Criteria, stated so the verdict can be checked by eye: at 5 days the ladder is monotone
    (both steps > 0) in at least 4 of the folds, and the mean graph step at 5 days exceeds
    the mean graph step at 1 day.
    """
    folds = s[~s.index.get_level_values("fold").isin(["mean", "pooled"])]
    g = folds["graph (proposed - B6)"]
    lines = []
    for h in sorted(folds.index.get_level_values("horizon").unique()):
        gh, mh = g.xs(h, level="horizon"), folds["monotone"].xs(h, level="horizon")
        lines.append(f"{h}d: graph step per fold {', '.join(f'{v:+.3f}' for v in gh)} "
                     f"(mean {gh.mean():+.3f}, positive in {(gh > 0).sum()}/{len(gh)}); "
                     f"ladder monotone in {mh.sum()}/{len(mh)} folds")
    hs = sorted(folds.index.get_level_values("horizon").unique())
    if hs == [1, 5]:
        mono5 = folds["monotone"].xs(5, level="horizon")
        g1, g5 = g.xs(1, level="horizon").mean(), g.xs(5, level="horizon").mean()
        holds = mono5.sum() >= len(mono5) - 1 and g5 > g1
        lines.append(f"VERDICT: horizon pattern {'HOLDS' if holds else 'DOES NOT HOLD'} "
                     f"(5d monotone in {mono5.sum()}/{len(mono5)} folds, need {len(mono5) - 1}; "
                     f"mean graph step 5d {g5:+.3f} vs 1d {g1:+.3f})")
    return lines


def _md(df: pd.DataFrame) -> str:
    from src.graph.comention_report import md

    return md(df, digits=3)


def retrain_ladder(seed: int, panel, edges, folds, horizons=(1, 5)) -> dict[str, pd.DataFrame]:
    """The three ladder models under another seed, in memory (nothing written)."""
    from src.core.config import load_config
    from src.core.paths import CONFIGS_DIR
    from src.models.train import train_fold

    out = {}
    for name in LADDER:
        cfg = load_config(CONFIGS_DIR / f"{name}.yaml") | {"seed": seed}
        parts = [train_fold(panel, edges, f, h, cfg, log=lambda *_: None)[0] for h in horizons for f in folds]
        out[name] = pd.concat(parts, ignore_index=True)
    return out


def classical_in_memory(panel, folds, horizons=(1, 5)) -> dict[str, pd.DataFrame]:
    """Baselines 1-4 through `run_frame_baseline`, for when 4.8 has not written their files yet."""
    import warnings

    from src.core.io import read_table
    from src.data.indicators import compute_indicators
    from src.models.baselines import BASELINES
    from src.models.baselines.base import run_frame_baseline

    prices, labels = read_table("prices"), read_table("labels")
    frame = (prices.merge(labels, on=["date", "ticker"]).merge(compute_indicators(prices), on=["date", "ticker"])
             .set_index(["date", "ticker"]).sort_index())
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in CLASSICAL:
            parts = [run_frame_baseline(BASELINES[name](), frame, panel, f, h)[0] for h in horizons for f in folds]
            out[name] = pd.concat(parts, ignore_index=True)
    return out


def build(preds: dict[str, pd.DataFrame], seed_runs: dict[int, dict] | None = None) -> str:
    from src.core.schema import conform

    missing = [m for m in LADDER if m not in preds]
    if missing:
        raise ValueError(f"missing ladder predictions: {missing}; run src.models.train with their configs")
    # Files come back with SCHEMA.md dtypes (int8 horizon, int16 fold); in-memory runs do not.
    # Conform first, so the check compares rows rather than dtypes.
    preds = {m: conform(d, "predictions") for m, d in preds.items()}
    rows = {m: preds[m][["horizon", "fold", "date", "ticker"]].sort_values(["horizon", "fold", "date", "ticker"])
            .reset_index(drop=True) for m in preds}
    ref = rows["gat_lstm"]
    differ = [m for m, r in rows.items() if not r.equals(ref)]
    if differ:
        raise ValueError(f"predictions cover different rows from gat_lstm: {differ}")

    t = auc_table(preds)
    s = steps(t)
    base = {int(h): float(g["y_true"].mean()) for h, g in preds["gat_lstm"].groupby("horizon")}
    out = [
        "# Walk-forward ladder (task 3.8, #36)",
        "",
        "Generated by `python -m src.models.ladder_report`. Test-window AUC per fold; `mean` averages "
        "the folds, `pooled` scores every test row at once. Every model covers the same "
        f"{len(ref):,} (horizon, date, ticker) rows. Base rate (share up): "
        + ", ".join(f"{h}d {b:.3f}" for h, b in sorted(base.items())) + ".",
        "",
        "## Horizon pattern",
        "",
        "Propagation takes time: the graph step should be near zero at 1 day and positive at 5 days.",
        "",
        *[f"- {line}" for line in horizon_pattern(s)],
    ]
    if seed_runs:
        out += ["", "### Across seeds", ""]
        for seed, sp in sorted(seed_runs.items()):
            out += [f"- seed {seed}: {horizon_pattern(steps(auc_table(sp)))[-1]}"]
    out += ["", "## AUC", "", _md(t.rename(columns=LABEL)), "", "## Steps", "", _md(s)]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=None, help="directory for ladder_3_8.md (default results/tables)")
    p.add_argument("--seeds", type=int, nargs="*", default=[], help="extra seeds: retrain the ladder in memory")
    p.add_argument("--classical", action="store_true", help="run baselines 1-4 in memory if their files are absent")
    args = p.parse_args()

    from src.core.io import read_predictions
    from src.core.paths import RESULTS_DIR, predictions_path

    preds = {m: read_predictions(m) for m in (*CLASSICAL, *LADDER) if predictions_path(m).exists()}
    seed_runs = {}
    if args.seeds or (args.classical and not all(m in preds for m in CLASSICAL)):
        from src.core.io import read_table
        from src.models.panel import load_panel
        from src.models.train import folds_for

        panel = load_panel()
        folds = folds_for(panel)
        if args.classical:
            preds |= {m: d for m, d in classical_in_memory(panel, folds).items() if m not in preds}
        if args.seeds:
            from src.core.config import load_config
            from src.core.paths import CONFIGS_DIR

            edges = read_table("edges")
            file_seed = load_config(CONFIGS_DIR / "gat_lstm.yaml")["seed"]  # the seed behind the files
            seed_runs = {file_seed: {m: preds[m] for m in LADDER}}
            seed_runs |= {s: retrain_ladder(s, panel, edges, folds) for s in args.seeds if s != file_seed}
    text = build(preds, seed_runs)
    out = (args.output or RESULTS_DIR / "tables") / "ladder_3_8.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
