"""Baseline 6: LSTM + news, no graph. Owner: Evaluation (runs, 4.13 tuning); wrapper: Modelling, #35.

The decisive comparison: the proposed model with the GAT removed (README, Baselines).

This wrapper holds no training logic on purpose: it runs `src.models.train.train_fold`, the
loop that trains the proposed model, with `configs/lstm_news.yaml`. That makes the comparison
structurally exact: same panel, features, normalisation, encoder, optimiser, early stopping,
validation split and threshold procedure, and under one seed the same initial encoder weights.

    python -m src.models.train --config configs/lstm_news.yaml
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.paths import CONFIGS_DIR


def check_no_graph_config(cfg: dict, name: str, config: Path) -> None:
    if cfg.get("model") != name or cfg["gat"]["edge_types"]:
        raise ValueError(f"{config} is not a {name} config: model {cfg.get('model')!r}, "
                         f"edge types {cfg['gat']['edge_types']}")


class LSTMNews:
    name = "lstm_news"

    def __init__(self, config: Path = CONFIGS_DIR / "lstm_news.yaml"):
        from src.core.config import load_config

        self.cfg = load_config(config)
        check_no_graph_config(self.cfg, self.name, config)

    def run_fold(self, panel, fold, horizon: int) -> tuple[pd.DataFrame, dict]:
        from src.models.train import train_fold

        return train_fold(panel, None, fold, horizon, self.cfg, log=lambda *_: None)
