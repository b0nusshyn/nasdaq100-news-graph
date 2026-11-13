"""YAML config loading. `configs/walk_forward.yaml` is the single source of fold settings:
the proposed model and every baseline must be evaluated on the same folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.core.paths import CONFIGS_DIR


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS_DIR / path.name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def walk_forward_config() -> dict[str, Any]:
    return load_config(CONFIGS_DIR / "walk_forward.yaml")
