"""Where things live. The data directory is configurable; everything else is repo-relative.

Set `NDX_DATA_DIR` to point at the shared Google Drive folder (see data/README.md).
Unset, it defaults to `<repo>/data`. Point it at `tests/fixtures` to run any stage on
synthetic data.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

DATA_DIR_ENV = "NDX_DATA_DIR"


def data_dir() -> Path:
    return Path(os.environ.get(DATA_DIR_ENV, REPO_ROOT / "data")).expanduser()


def raw_dir() -> Path:
    """Unprocessed downloads (FNSPID extract, scraped tables). Not covered by SCHEMA.md."""
    return data_dir() / "raw"


def table_path(name: str) -> Path:
    """Path of a SCHEMA.md artifact. Predictions are one file per model."""
    if name == "predictions":
        raise ValueError("use predictions_path(model) for predictions")
    return data_dir() / f"{name}.parquet"


def predictions_path(model: str) -> Path:
    return data_dir() / "predictions" / f"{model}.parquet"
