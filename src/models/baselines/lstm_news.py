"""Baseline 6: LSTM + news, no graph. Owner: Evaluation (runs, 4.13 tuning); wrapper: Modelling, #35.

The decisive comparison: the proposed model with the GAT removed (README, Baselines).

This wrapper holds no training logic on purpose: it runs `src.models.train.train_fold`, the
loop that trains the proposed model, with `configs/lstm_news.yaml`. That makes the comparison
structurally exact: same panel, features, normalisation, encoder, optimiser, early stopping,
validation split and threshold procedure, and under one seed the same initial encoder weights.

    python -m src.models.train --config configs/lstm_news.yaml
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import pandas as pd

from src.core.paths import CONFIGS_DIR


def check_no_graph_config(cfg: dict, name: str, config: Path) -> None:
    if cfg.get("model") != name or cfg["gat"]["edge_types"]:
        raise ValueError(
            f"{config} is not a {name} config: model {cfg.get('model')!r}, edge types {cfg['gat']['edge_types']}"
        )


class LSTMNews:
    name = "lstm_news"

    def __init__(self, config: Path = CONFIGS_DIR / "lstm_news.yaml"):
        from src.core.config import load_config

        self.cfg = load_config(config)
        check_no_graph_config(self.cfg, self.name, config)

    def run_fold(self, panel, fold, horizon: int) -> tuple[pd.DataFrame, dict]:
        """Run the shared training loop after checking the evaluation boundaries.

        This is the project's PanelBaseline interface. No separate encoder,
        projection, or training implementation is maintained by this wrapper.
        The panel must contain the complete trading calendar for the embargo.
        """
        from src.models.train import train_fold

        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon not in (1, 5):
            raise ValueError("horizon must be 1 or 5")
        if horizon not in panel.labels:
            raise ValueError(f"panel has no labels for horizon {horizon}")

        # Validate again because callers may change settings after construction.
        check_no_graph_config(self.cfg, self.name, Path("run_fold settings"))
        for name, dates in (("train", fold.train), ("test", fold.test)):
            if not isinstance(dates, pd.DatetimeIndex) or dates.empty:
                raise ValueError(f"{name} dates must be a nonempty DatetimeIndex")
            if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
                raise ValueError(f"{name} dates must be sorted, unique, and nonmissing")
            if dates.tz is not None or not dates.equals(dates.normalize()):
                raise ValueError(f"{name} dates must be timezone-naive dates at midnight")
            positions = panel.day_index(dates)
            if len(positions) > 1 and ((positions[1:] - positions[:-1]) != 1).any():
                raise ValueError(f"{name} dates must be contiguous trading sessions")

        train_end = panel.day_index(fold.train[-1:])[0]
        test_start = panel.day_index(fold.test[:1])[0]
        if test_start - train_end - 1 < 5:
            raise ValueError("test dates must follow training with at least five embargo sessions")
        if self.cfg.get("embargo_days", 5) != 5:
            raise ValueError("Baseline 6 requires the shared five-session embargo")

        return train_fold(panel, None, fold, horizon, self.cfg, log=lambda *_: None)
