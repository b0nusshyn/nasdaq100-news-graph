"""Task 3.12 (#58): the tuning grid and how the shared config is chosen."""

import pandas as pd

from src.models.tune import GRID, config_for, report, summarize

BASE = {"horizons": [1, 5], "encoder": {"hidden": 64, "layers": 1, "dropout": 0.1}, "gat": {"heads": 4},
        "train": {"epochs": 30, "batch_days": 32, "lr": 0.001, "early_stopping_patience": 5}}


def test_overrides_land_in_their_block_and_base_is_untouched():
    cfg = config_for(BASE, {"batch_days": 8, "dropout": 0.3, "heads": 2})
    assert cfg["train"]["batch_days"] == 8 and cfg["encoder"]["dropout"] == 0.3 and cfg["gat"]["heads"] == 2
    assert cfg["train"]["early_stopping_metric"] == "within_day_auc"
    assert BASE["train"]["batch_days"] == 32 and "early_stopping_metric" not in BASE["train"]


def test_the_grid_has_at_least_six_runs_and_sweeps_batch_days_down_to_8():
    assert len(GRID) >= 6 and {g["batch_days"] for g in GRID.values()} >= {8, 32}


def runs(scores: dict[str, tuple[float, float, float]], folds: int = 5) -> pd.DataFrame:
    """Fake log rows: config -> (val within-day AUC 1d, 5d, val MCC)."""
    rows = []
    for cid, (a1, a5, mcc) in scores.items():
        for h, a in ((1, a1), (5, a5)):
            for f in range(folds):
                rows.append({"config": cid, "horizon": h, "fold": f, "batch_days": 8, "lr": 1e-3, "hidden": 64,
                             "heads": 4, "dropout": 0.1, "best_val_within_day_auc": a, "val_mcc": mcc,
                             "val_within_day_auc_at_loss_optimum": a - 0.01, "best_step": 30,
                             "loss_optimal_step": 10, "steps_per_epoch": 14, "seconds": 60.0})
    return pd.DataFrame(rows)


def test_the_shared_winner_is_chosen_on_validation_within_day_auc_not_mcc():
    _, winner = summarize(runs({"A": (0.52, 0.51, 0.01), "B": (0.50, 0.50, 0.20)}))
    assert winner == "A"


def test_an_unfinished_config_cannot_win():
    r = runs({"A": (0.52, 0.51, 0.0), "B": (0.60, 0.60, 0.0)})
    r = r[~((r["config"] == "B") & (r["fold"] == 4))]
    assert summarize(r)[1] == "A"


def test_report_names_each_horizons_own_best_and_the_stopping_gap():
    text = report(runs({"A": (0.52, 0.50, 0.0), "B": (0.50, 0.53, 0.0)}))
    assert "1d: own best A" in text and "5d: own best B" in text
    assert "median gap +20 steps" in text and "Chosen shared config" in text
