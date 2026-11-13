"""Machine-readable data contract. Mirrors SCHEMA.md; tests/test_schema.py keeps the two in sync.

Every Parquet artifact under the data directory is described by one `Table`. Writers call
`src.core.io.write_table`, which conforms and validates before writing, so a file that
violates the contract never reaches a teammate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

EMB_DIM = 768
EMB_COLS = tuple(f"emb_{i}" for i in range(EMB_DIM))
EDGE_TYPES = ("sector", "comention", "corr")
HORIZONS = (1, 5)
ALL_FOLDS = -1  # `fold` value for edges valid in every fold (e.g. sector)

# Logical dtypes used in SCHEMA.md. `conform` casts to the pandas dtype on the right.
PANDAS_DTYPE = {
    "date": "datetime64[ns]",
    "string": "string",
    "bool": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "float32": "float32",
    "float64": "float64",
    "Int8": "Int8",  # nullable int
}


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # key of PANDAS_DTYPE
    nullable: bool = False


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    key: tuple[str, ...]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


def _cols(*specs: tuple) -> tuple[Column, ...]:
    return tuple(Column(*s) for s in specs)


MEMBERSHIP = Table(
    "membership",
    _cols(("ticker", "string"), ("start", "date"), ("end", "date", True)),
    key=("ticker", "start"),
)

PRICES = Table(
    "prices",
    _cols(
        ("date", "date"), ("ticker", "string"),
        ("open", "float64"), ("high", "float64"), ("low", "float64"), ("close", "float64"),
        ("volume", "float64"), ("in_index", "bool"),
    ),
    key=("date", "ticker"),
)

LABELS = Table(
    "labels",
    _cols(
        ("date", "date"), ("ticker", "string"),
        ("ret_1d", "float64", True), ("ret_5d", "float64", True),
        ("y_1d", "Int8", True), ("y_5d", "Int8", True),
    ),
    key=("date", "ticker"),
)

ARTICLE_TICKERS = Table(
    "article_tickers",
    _cols(("article_id", "string"), ("date", "date"), ("ticker", "string")),
    key=("article_id", "ticker"),
)

NEWS_FEATURES = Table(
    "news_features",
    _cols(
        ("date", "date"), ("ticker", "string"),
        ("has_news", "bool"), ("n_articles", "int32"),
        ("sentiment", "float32"), ("confidence", "float32"),
        *((c, "float32") for c in EMB_COLS),
    ),
    key=("date", "ticker"),
)

EDGES = Table(
    "edges",
    _cols(
        ("fold", "int16"), ("src", "string"), ("dst", "string"),
        ("edge_type", "string"), ("weight", "float32"),
    ),
    key=("fold", "src", "dst", "edge_type"),
)

PREDICTIONS = Table(
    "predictions",
    _cols(
        ("model", "string"), ("horizon", "int8"), ("fold", "int16"),
        ("date", "date"), ("ticker", "string"),
        ("p_up", "float32"), ("y_true", "int8"),
    ),
    key=("model", "horizon", "date", "ticker"),
)

TABLES: dict[str, Table] = {
    t.name: t
    for t in (MEMBERSHIP, PRICES, LABELS, ARTICLE_TICKERS, NEWS_FEATURES, EDGES, PREDICTIONS)
}


class SchemaError(ValueError):
    """A DataFrame does not match its table in the data contract."""


def get_table(table: str | Table) -> Table:
    if isinstance(table, Table):
        return table
    try:
        return TABLES[table]
    except KeyError:
        raise SchemaError(f"unknown table {table!r}; known: {sorted(TABLES)}") from None


def conform(df: pd.DataFrame, table: str | Table) -> pd.DataFrame:
    """Return `df` with the table's column order and dtypes. Missing columns raise."""
    t = get_table(table)
    missing = [c for c in t.column_names if c not in df.columns]
    if missing:
        raise SchemaError(f"{t.name}: missing columns {_short(missing)}")
    out = df.loc[:, t.column_names].copy()
    for col in t.columns:
        if col.dtype == "date":
            out[col.name] = pd.to_datetime(out[col.name]).astype("datetime64[ns]")
        else:
            out[col.name] = out[col.name].astype(PANDAS_DTYPE[col.dtype])
    return out.reset_index(drop=True)


def validate(df: pd.DataFrame, table: str | Table) -> None:
    """Raise SchemaError listing every way `df` breaks the contract for `table`."""
    t = get_table(table)
    errors: list[str] = []

    missing = [c for c in t.column_names if c not in df.columns]
    extra = [c for c in df.columns if c not in t.column_names]
    if missing:
        errors.append(f"missing columns {_short(missing)}")
    if extra:
        errors.append(f"unexpected columns {_short(extra)}")

    for col in t.columns:
        if col.name not in df.columns:
            continue
        s = df[col.name]
        if not _dtype_ok(s, col.dtype):
            errors.append(f"{col.name}: dtype {s.dtype}, expected {col.dtype}")
        if not col.nullable and s.isna().any():
            errors.append(f"{col.name}: {int(s.isna().sum())} nulls in non-nullable column")
        if col.dtype == "date" and _dtype_ok(s, "date"):
            if (s.dropna() != s.dropna().dt.normalize()).any():
                errors.append(f"{col.name}: has a time component; dates must be midnight")

    if not missing and df.duplicated(list(t.key)).any():
        errors.append(f"duplicate keys on {t.key}: {int(df.duplicated(list(t.key)).sum())} rows")

    errors += _table_checks(df, t)
    if errors:
        raise SchemaError(f"{t.name}: " + "; ".join(errors))


def _table_checks(df: pd.DataFrame, t: Table) -> list[str]:
    errs: list[str] = []
    cols = set(df.columns)
    if t.name == "labels":
        for h in HORIZONS:
            y = df.get(f"y_{h}d")
            if y is not None and not y.dropna().isin([0, 1]).all():
                errs.append(f"y_{h}d: values outside {{0, 1}}")
    if t.name == "news_features" and {"has_news", "n_articles"} <= cols:
        if ((df["n_articles"] > 0) != df["has_news"]).any():
            errs.append("has_news must equal n_articles > 0")
    if t.name == "edges" and "edge_type" in cols:
        bad = set(df["edge_type"].dropna()) - set(EDGE_TYPES)
        if bad:
            errs.append(f"edge_type: unknown values {sorted(bad)}")
    if t.name == "predictions" and "p_up" in cols:
        if ((df["p_up"] < 0) | (df["p_up"] > 1)).any():
            errs.append("p_up: values outside [0, 1]")
    return errs


def _dtype_ok(s: pd.Series, dtype: str) -> bool:
    if dtype == "date":
        return pd.api.types.is_datetime64_dtype(s) and getattr(s.dt, "tz", None) is None
    if dtype == "string":
        return pd.api.types.is_string_dtype(s)
    return str(s.dtype) == PANDAS_DTYPE[dtype]


def _short(items: list[str], n: int = 5) -> str:
    return str(items[:n]) + (f" (+{len(items) - n} more)" if len(items) > n else "")
