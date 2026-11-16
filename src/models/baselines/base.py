"""Common interface for every baseline. Owner: Evaluation (panel path: Modelling, #35).

`src.eval.run_baselines` drives each baseline through the same folds as the proposed model.
Two kinds of baseline, one fold procedure:

- **`Baseline`** (1-4: buy-and-hold, ARIMA, logistic, XGBoost) are different model families and
  read a pandas `(date, ticker)` panel with their own preprocessing, fitted on the training rows.
  `run_frame_baseline` runs one through a fold.
- **`PanelBaseline`** (5-6: the LSTMs) are the proposed model with a component removed. They run
  through `src.models.train.train_fold`, so 5 -> 6 -> proposed differ by one component each.

What every model shares, whichever kind: the folds and embargo, the row set
(`src.models.panel.Panel.rows`), the validation split (`split_validation`: fit on the first part
of the training window, never the validation days) and the threshold procedure
(`choose_threshold` on the validation days, never the test window).
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Baseline(Protocol):
    name: str

    def fit(self, panel: pd.DataFrame, train_dates: pd.DatetimeIndex, horizon: int) -> None:
        """`panel` is indexed by (date, ticker) and holds prices, indicators, news features and labels.
        Fit only on rows whose date is in `train_dates`."""

    def predict_proba(self, panel: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        """P(up) indexed by (date, ticker) for rows in `test_dates`. Earlier rows of `panel` may be
        read as history (e.g. LSTM windows) but never labels from the test window."""


class PanelBaseline(Protocol):
    name: str

    def run_fold(self, panel, fold, horizon: int) -> tuple[pd.DataFrame, dict]:
        """Train on `fold.train` only; return predictions for `panel.rows(fold.test, horizon)` in
        SCHEMA.md `predictions` columns, and an info dict holding the validation `threshold`."""


def run_frame_baseline(
    model: Baseline, frame: pd.DataFrame, panel, fold, horizon: int, *, validation_fraction: float = 0.1,
    embargo: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """One fold of a `Baseline`, with the same procedure as `train_fold`.

    Fits on the training days before the validation split, picks the threshold on the validation
    days, predicts the test days, and returns exactly the shared row set: a row the baseline did
    not predict is an error, and rows outside the set (unlabelled in-index rows) are dropped.
    """
    from src.models.panel import choose_threshold, split_validation

    fit_days, val_days = split_validation(fold.train, validation_fraction, embargo)
    model.fit(frame, fit_days, horizon)

    def on_rows(dates) -> pd.DataFrame:
        rows = panel.rows(dates, horizon)
        p = model.predict_proba(frame, pd.DatetimeIndex(dates))
        p.index = pd.MultiIndex.from_arrays(
            [p.index.get_level_values("date"), p.index.get_level_values("ticker").astype("string")],
            names=["date", "ticker"])
        key = pd.MultiIndex.from_frame(rows[["date", "ticker"]])
        missing = key.difference(p.index)
        if len(missing):
            raise ValueError(f"{model.name}: no prediction for {len(missing)} rows of the shared row set, "
                             f"e.g. {list(missing[:3])}")
        return rows.assign(p_up=p.reindex(key).to_numpy("float32")).rename(columns={"y": "y_true"})

    val = on_rows(val_days)
    threshold = choose_threshold(val["y_true"].to_numpy(), val["p_up"].to_numpy())
    preds = on_rows(fold.test).assign(model=model.name, horizon=horizon, fold=fold.index)
    info = {"fold": fold.index, "horizon": horizon, "threshold": threshold,
            "val_base_rate": float(val["y_true"].mean()), "test_rows": len(preds)}
    return preds, info
