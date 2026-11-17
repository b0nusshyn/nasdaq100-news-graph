"""Task 3.8 (#36): the ladder report and its horizon-pattern flag."""

import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import classification_metrics
from src.models.ladder_report import auc, auc_table, build, horizon_pattern, steps

FOLDS, ROWS = 5, 400


def synthetic(signal: dict[tuple[str, int], float], seed: int = 0) -> dict[str, pd.DataFrame]:
    """Predictions for every ladder model whose P(up) carries `signal[(model, horizon)]` of the label."""
    rng = np.random.default_rng(seed)
    base = []
    for h in (1, 5):
        for f in range(FOLDS):
            y = rng.integers(0, 2, ROWS)
            base.append(pd.DataFrame({"horizon": h, "fold": f, "date": pd.Timestamp("2021-01-04") + pd.to_timedelta(
                np.arange(ROWS), "D"), "ticker": "AAA", "y_true": y}))
    base = pd.concat(base, ignore_index=True)
    out = {}
    for m in ("lstm_price", "lstm_news", "gat_lstm"):
        k = base["horizon"].map(lambda h, m=m: signal[(m, h)])
        p = 1 / (1 + np.exp(-(k * (2 * base["y_true"] - 1) + rng.normal(0, 1, len(base)))))
        out[m] = base.assign(p_up=p, model=m)
    return out


# The pattern the hypothesis predicts: at 1 day nothing adds, at 5 days each rung adds.
HOLDS = {("lstm_price", 1): 0.2, ("lstm_news", 1): 0.2, ("gat_lstm", 1): 0.2,
         ("lstm_price", 5): 0.1, ("lstm_news", 5): 0.4, ("gat_lstm", 5): 0.8}
FLAT = {k: 0.3 for k in HOLDS}


def test_auc_matches_the_shared_metrics_module():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 500), rng.random(500)
    assert auc(y, p) == pytest.approx(classification_metrics(y, p)["auc"])


def test_table_has_every_fold_plus_mean_and_pooled_rows_per_horizon():
    t = auc_table(synthetic(HOLDS))
    assert list(t.columns) == ["lstm_price", "lstm_news", "gat_lstm"]
    for h in (1, 5):
        assert list(t.xs(h, level="horizon").index) == [*map(str, range(FOLDS)), "mean", "pooled"]


def test_the_flag_reports_the_pattern_when_it_is_there():
    lines = horizon_pattern(steps(auc_table(synthetic(HOLDS))))
    assert lines[-1].startswith("VERDICT: horizon pattern HOLDS")
    assert lines[1].startswith("5d:") and "monotone in 5/5" in lines[1]


def test_the_flag_reports_its_absence():
    assert "DOES NOT HOLD" in horizon_pattern(steps(auc_table(synthetic(FLAT))))[-1]


def test_report_refuses_models_scored_on_different_rows():
    preds = synthetic(HOLDS)
    preds["lstm_news"] = preds["lstm_news"].iloc[1:]
    with pytest.raises(ValueError, match="different rows"):
        build(preds)


def test_report_states_the_rows_base_rate_and_the_verdict():
    text = build(synthetic(HOLDS))
    assert "## Horizon pattern" in text and "VERDICT" in text and "Base rate" in text
    assert f"{2 * FOLDS * ROWS:,}" in text
