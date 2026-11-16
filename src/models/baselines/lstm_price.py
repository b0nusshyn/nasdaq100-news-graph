"""Baseline 5: LSTM, price only. Owner: Evaluation (runs, 4.13 tuning); wrapper: Modelling, #35.

Baseline 6 with the news removed, so the 5 -> 6 step isolates the contribution of news.

This wrapper holds no training logic on purpose: it runs `src.models.train.train_fold`, the
loop that trains the proposed model, with `configs/lstm_price.yaml`. The ladder 5 -> 6 ->
proposed then differs by one component per step, in the same code. (The first, standalone
implementation from 16 Nov is in git history at f64bddf.)

    python -m src.models.train --config configs/lstm_price.yaml
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.paths import CONFIGS_DIR
from src.models.baselines.lstm_news import check_no_graph_config


class LSTMPrice:
    name = "lstm_price"

    def __init__(self, config: Path = CONFIGS_DIR / "lstm_price.yaml"):
        from src.core.config import load_config

        self.cfg = load_config(config)
        check_no_graph_config(self.cfg, self.name, config)
        if self.cfg["features"]["news"] or self.cfg["features"]["embeddings"]:
            raise ValueError(f"{config}: Baseline 5 uses no news features")

    def run_fold(self, panel, fold, horizon: int) -> tuple[pd.DataFrame, dict]:
        from src.models.train import train_fold

        return train_fold(panel, None, fold, horizon, self.cfg, log=lambda *_: None)
