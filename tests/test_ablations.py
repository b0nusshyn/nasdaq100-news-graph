"""Task 3.9 (#41): the ablation and seed report."""

import pandas as pd

from src.models.ablations import NEWS_GRID, VARIANTS, delta_cell, monotone_folds, per_seed, report

SEEDS = (42, 43, 44)


def fake_log(level: dict[str, float], seed_shift=(0.0, 0.001, -0.001)) -> pd.DataFrame:
    """Every variant at every seed, horizon and fold, with within-day AUC `level[variant] + shift`."""
    rows = [{"variant": v, "seed": sd, "horizon": h, "fold": f, "within_day_auc": level.get(v, 0.5) + sh,
             "auc": 0.5, "timing_auc": 0.5}
            for v in VARIANTS for sd, sh in zip(SEEDS, seed_shift, strict=True) for h in (1, 5) for f in range(5)]
    return pd.DataFrame(rows)


LEVELS = {"B5 LSTM price": 0.500, "B6 LSTM + news": 0.505, "proposed": 0.509, "drop comention": 0.508,
          "drop news features": 0.502, "drop news features and comention": 0.501, "drop sector": 0.509,
          "drop corr": 0.507, "drop deal-arb corr pairs": 0.509}


def test_every_cell_of_the_news_grid_is_a_distinct_variant_with_sector_and_corr_kept():
    assert len(set(NEWS_GRID.values())) == 4
    for v in NEWS_GRID.values():
        assert "sector" not in VARIANTS[v][1] and "corr" not in VARIANTS[v][1]
    assert VARIANTS[NEWS_GRID[True, False]][1] == ("comention",)
    assert set(VARIANTS[NEWS_GRID[False, False]][1]) == {"news", "comention"}


def test_deltas_are_paired_by_seed():
    a, b = pd.Series([0.51, 0.52, 0.50], index=SEEDS), pd.Series([0.50, 0.53, 0.49], index=SEEDS)
    assert delta_cell(a, b) == "+0.0033 (-0.0100 to +0.0100; > 0 in 2/3 seeds)"


def test_fold_seed_count_exposes_a_step_carried_by_one_fold():
    a, b = pd.Series([0.51, 0.51, 0.51], index=SEEDS), pd.Series([0.50, 0.50, 0.50], index=SEEDS)
    fold_d = pd.Series([0.05, -0.0025, -0.0025, -0.0025, -0.0025] * 3)
    assert delta_cell(a, b, fold_d).endswith("> 0 in 3/3 seeds, 3/15 fold-seeds)")


def test_report_shows_the_grid_the_ladder_with_seed_range_and_the_noise_floor():
    text = report(fake_log(LEVELS))
    assert "### News: signal vs structure" in text and "| **news features** | 0.509" in text
    assert "seed noise floor: the proposed model's seed range is 0.0020" in text
    assert "graph step (proposed - B6): +0.0040" in text and "3/3 seeds, 15/15 fold-seeds" in text
    assert "Each step by fold, mean over seeds:" in text and "| step | fold 0 |" in text
    assert "| drop corr | 0.507" in text and "Incomplete" not in text


def test_monotone_counts_each_seeds_folds():
    assert monotone_folds(fake_log(LEVELS), 1) == {42: 5, 43: 5, 44: 5}


def test_an_unfinished_variant_is_flagged():
    log = fake_log(LEVELS)
    log = log[~((log["variant"] == "drop corr") & (log["seed"] == 44) & (log["fold"] == 4))]
    assert "**Incomplete:** drop corr seed 44" in report(log)
    assert per_seed(log).loc[("drop corr", 1), 44] == 0.507 - 0.001
