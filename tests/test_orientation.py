"""Label and score orientation: a below-chance AUC must mean the model is wrong, never a sign flip.

Added 18 Nov 2026 after Baseline 5 scored below 0.5 within-day at 5 days in all five folds (#92).
Three invariants, each from first principles rather than through the code that built the files:

- stored labels equal labels recomputed from prices (`fwd_ret_hd = close[t+h] / close[t] - 1`
  over each ticker's own trading days, `y_hd = fwd_ret_hd > 0`, as SCHEMA.md states);
- every model's `y_true` equals the stored label for its (date, ticker, horizon);
- the AUC functions score a perfect predictor (the true forward return) at 1.0 and its
  negative at 0.0.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.paths import FIXTURES_DIR, table_path
from src.models.ladder_report import auc, within_day_auc

HORIZONS = (1, 5)


def recomputed(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices[["date", "ticker", "close"]].sort_values(["ticker", "date"]).copy()
    for h in HORIZONS:
        p[f"fwd_{h}"] = p.groupby("ticker")["close"].shift(-h) / p["close"] - 1
    return p


def assert_labels_match_prices(prices: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    m = labels.merge(recomputed(prices), on=["date", "ticker"], how="inner", validate="1:1")
    assert len(m) == len(labels), "every label row has a price row"
    for h in HORIZONS:
        stored, fresh = m[f"fwd_ret_{h}d"], m[f"fwd_{h}"]
        assert (stored.isna() == fresh.isna()).all(), f"{h}d: null pattern differs (last {h} days per ticker)"
        ok = stored.notna()
        np.testing.assert_allclose(stored[ok], fresh[ok], rtol=0, atol=1e-12)
        assert (m.loc[ok, f"y_{h}d"].astype(int) == (fresh[ok] > 0).astype(int)).all(), f"{h}d: y != (fwd_ret > 0)"
    return m


def assert_oracle_scores_one(m: pd.DataFrame) -> None:
    for h in HORIZONS:
        d = m.dropna(subset=[f"fwd_{h}"]).assign(y_true=lambda x, h=h: x[f"y_{h}d"].astype(int))
        assert auc(d["y_true"], d[f"fwd_{h}"]) == 1.0 and auc(d["y_true"], -d[f"fwd_{h}"]) == 0.0
        assert within_day_auc(d.assign(p_up=d[f"fwd_{h}"])) == 1.0
        assert within_day_auc(d.assign(p_up=-d[f"fwd_{h}"])) == 0.0


def test_auc_scores_a_perfect_predictor_at_one_and_its_negative_at_zero():
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.02, 1000)
    y = (ret > 0).astype(int)
    assert auc(y, ret) == 1.0 and auc(y, -ret) == 0.0
    df = pd.DataFrame({"date": np.repeat(pd.bdate_range("2021-01-04", periods=50), 20), "y_true": y, "p_up": ret})
    assert within_day_auc(df) == 1.0 and within_day_auc(df.assign(p_up=-ret)) == 0.0


def test_fixture_labels_match_labels_recomputed_from_fixture_prices():
    from src.core.io import read_table

    m = assert_labels_match_prices(read_table("prices", FIXTURES_DIR / "prices.parquet"),
                                   read_table("labels", FIXTURES_DIR / "labels.parquet"))
    assert_oracle_scores_one(m)


@pytest.mark.real_data
def test_real_labels_and_predictions_are_oriented():
    if not (table_path("prices").exists() and table_path("labels").exists()):
        pytest.skip("prices/labels not in the data directory (gitignored; on the team Drive)")
    from src.core.io import read_predictions, read_table
    from src.core.paths import predictions_path
    from src.models.ladder_report import CLASSICAL, LADDER

    labels = read_table("labels")
    m = assert_labels_match_prices(read_table("prices"), labels)
    assert_oracle_scores_one(m)
    for name in (*CLASSICAL, *LADDER):
        if not predictions_path(name).exists():
            continue
        p = read_predictions(name)
        for h in HORIZONS:
            ph = p[p["horizon"] == h].merge(labels[["date", "ticker", f"y_{h}d"]], on=["date", "ticker"],
                                            how="left", validate="1:1")
            assert (ph["y_true"].astype(int) == ph[f"y_{h}d"].astype(int)).all(), f"{name} {h}d: y_true != label"
