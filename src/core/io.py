"""Read and write SCHEMA.md artifacts. Both directions validate, so bad data fails loudly."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core import paths
from src.core.schema import conform, get_table, validate


def write_table(df: pd.DataFrame, name: str, path: Path | None = None) -> Path:
    """Conform `df` to the contract, validate it, and write it as Parquet."""
    table = get_table(name)
    out = conform(df, table)
    validate(out, table)
    path = path or paths.table_path(table.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def read_table(name: str, path: Path | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Read an artifact and validate it. With `columns`, only those columns are read and checked."""
    table = get_table(name)
    path = path or paths.table_path(table.name)
    df = pd.read_parquet(path, columns=columns)
    df = conform(df, table) if columns is None else df
    if columns is None:
        validate(df, table)
    return df


def write_predictions(df: pd.DataFrame, model: str) -> Path:
    return write_table(df, "predictions", paths.predictions_path(model))


def read_predictions(model: str) -> pd.DataFrame:
    return read_table("predictions", paths.predictions_path(model))
