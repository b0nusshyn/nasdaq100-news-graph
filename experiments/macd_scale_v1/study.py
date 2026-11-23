"""Matched test: does making MACD scale-free change the original-attention GAT-LSTM?

Hypothesis (fixed before fitting): `macd` and `macd_signal` (src/data/indicators.py) are price
differences on *adjusted* closes, z-scored with one global mean/std across tickers
(src/models/panel.py). Their cross-sectional scale therefore tracks each ticker's adjusted
price level, which depends on splits made after the sample (look-ahead in the level) and is
otherwise noise for direction. Dividing both by the same-day adjusted close makes them
scale-free (the adjustment factor cancels) and uses only data up to that close.

Arms, identical in everything else to experiments/gat_edge_weight_v1 (same windows, fit/val
partitions, graph, epochs, seeds, selection rule):
- original:  re-train of the archived arm (also checks training determinism vs the archive)
- macd_norm: macd/close, macd_signal/close

Both 2022 and 2023 windows are development data already inspected by earlier studies; this
is one pre-registered comparison, not a search, and is reported as such.
"""
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.gat_lstm_tuning.feature_comparison import clean
from experiments.gat_lstm_tuning.run import digest
from src.core.config import load_config
from src.core.paths import table_path
from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold
from src.models.ladder_report import auc
from src.models.panel import build_panel
from src.models.train import GraphSources, fold_report, train_fold, training_partitions

ROOT = Path(__file__).parent
ARCHIVE = Path("experiments/gat_edge_weight_v1/runs")
ARMS = ("original", "macd_norm")
RANGES = (("2022-07-18", "2022-12-30"), ("2023-07-24", "2023-12-20"))


def normalise_macd(panel, prices):
    close = (prices.pivot(index="date", columns="ticker", values="close")
             .reindex(index=panel.days, columns=list(panel.tickers)).to_numpy(float))
    out = deepcopy(panel)
    for name in ("macd", "macd_signal"):
        i = panel.price_names.index(name)
        out.price[:, :, i] = panel.price[:, :, i] / close
    return out


def diagnostics(panel, prices):
    """Share of cross-sectional |macd| variation explained by log adjusted price level."""
    close = (prices.pivot(index="date", columns="ticker", values="close")
             .reindex(index=panel.days, columns=list(panel.tickers)).to_numpy(float))
    i = panel.price_names.index("macd")
    m = np.abs(panel.price[:, :, i])
    sel = panel.days.year.isin([2021, 2022, 2023])
    corrs = []
    for d in np.where(sel)[0]:
        ok = panel.in_index[d] & np.isfinite(m[d]) & np.isfinite(close[d]) & (m[d] > 0)
        if ok.sum() > 20:
            corrs.append(pd.Series(np.log(m[d][ok])).corr(pd.Series(np.log(close[d][ok])), method="spearman"))
    return {"mean_daily_spearman_log_abs_macd_vs_log_close": float(np.mean(corrs)), "days": len(corrs)}


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    panel = build_panel(prices, labels, empty, embeddings=False)
    panels = {"original": panel, "macd_norm": normalise_macd(panel, prices)}
    base = load_config(Path("configs/gat_lstm.yaml"))
    base["embargo_days"] = 5
    base["features"].update(news=[], embeddings=False)
    base["train"].update(epochs=12, validate_every_steps=None, validation_fraction=50 / 307,
                         preprocessing_scope="weight_fit", record_history=True)
    sources = GraphSources(prices=prices)
    output = ROOT / "runs"
    output.mkdir()
    paths = [table_path(n) for n in ("prices", "labels", "membership")]
    paths += [Path("configs/gat_lstm.yaml"), Path(__file__), *sorted(Path("src/models").glob("*.py"))]
    manifest = {"config": base, "arms": ARMS, "ranges": RANGES, "torch": str(torch.__version__),
                "diagnostics": diagnostics(panel, prices), "max_fits": 8,
                "source_hashes": {str(p): digest(p) for p in paths}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(manifest["diagnostics"], flush=True)
    folds = []
    for w, (first, last) in enumerate(RANGES):
        days = panel.days[(panel.days >= first) & (panel.days <= last)]
        start = panel.days.get_loc(days[0])
        fold = Fold(w, panel.days[start - 312:start - 5], days)
        fit, val = training_partitions(fold, base)
        assert (len(fit), len(val)) == (252, 50)
        folds.append((fold, fit, val))
    with (output / "ledger.jsonl").open("x") as ledger:
        for seed in (42, 43):
            for fold, _fit, _val in folds:
                for arm in ARMS:
                    if shutil.disk_usage(ROOT).free < 2.1 * 1024**3:
                        raise OSError("Disk space below experimental margin")
                    cfg = deepcopy(base)
                    cfg.update(seed=seed, model=f"edge_v1_{arm}")
                    cfg["gat"].update(edge_types=["corr"], signed_messages=False, center_graph_inputs=False)
                    name = f"w{fold.index}_h5_s{seed}_{arm}"
                    p = panels[arm]
                    preds, info, state = train_fold(p, sources, fold, 5, cfg, return_model=True, log=lambda *_: None)
                    rows = p.rows(fold.test, 5)
                    np.testing.assert_array_equal(preds.y_true, rows.y)
                    m = fold_report(preds, info)
                    m["majority"] = max(m["base_rate"], 1 - m["base_rate"])
                    m["daily_auc"] = {str(d.date()): auc(g.y_true, g.p_up) for d, g in preds.groupby("date")}
                    archived = ARCHIVE / f"w{fold.index}_h5_s{seed}_original.parquet"
                    if arm == "original" and archived.exists():
                        a = pd.read_parquet(archived)
                        m["max_delta_vs_archive"] = float(np.abs(a.p_up.to_numpy() - preds.p_up.to_numpy()).max())
                    preds.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
                    torch.save(state.model.state_dict(), output / f"{name}.pt")
                    print(json.dumps(clean(dict(id=name, arm=arm, seed=seed, config=cfg, rows=len(preds),
                                                metrics=m, **info)), allow_nan=False), file=ledger, flush=True)
                    print(f"done {name}: wdAUC {m['within_day_auc']:.4f} acc {m['accuracy']:.4f} "
                          f"maj {m['majority']:.4f} mcc {m['mcc']:.4f} "
                          f"delta {m.get('max_delta_vs_archive', 'n/a')}", flush=True)


if __name__ == "__main__":
    main()
