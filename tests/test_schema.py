import re

import pandas as pd
import pytest

from src.core.fixtures import make_fixtures
from src.core.io import read_table, write_table
from src.core.paths import FIXTURES_DIR, REPO_ROOT
from src.core.schema import EMB_DIM, TABLES, SchemaError, conform, validate


@pytest.fixture(scope="module")
def tables():
    return {name: conform(df, name) for name, df in make_fixtures().items()}


@pytest.mark.parametrize("name", sorted(TABLES))
def test_fixtures_satisfy_contract(tables, name):
    validate(tables[name], name)


@pytest.mark.parametrize("name", sorted(TABLES))
def test_committed_fixtures_satisfy_contract(name):
    path = FIXTURES_DIR / ("predictions/random.parquet" if name == "predictions" else f"{name}.parquet")
    read_table(name, path)


def test_roundtrip(tables, tmp_path):
    path = write_table(tables["labels"], "labels", tmp_path / "labels.parquet")
    pd.testing.assert_frame_equal(read_table("labels", path), tables["labels"])


def test_fixtures_are_consistent(tables):
    prices, news, edges = tables["prices"], tables["news_features"], tables["edges"]
    assert set(tables["labels"][["date", "ticker"]].itertuples(index=False)) == set(
        prices[["date", "ticker"]].itertuples(index=False)
    )
    in_index = set(prices.loc[prices["in_index"], ["date", "ticker"]].itertuples(index=False))
    assert set(news[["date", "ticker"]].itertuples(index=False)) == in_index
    assert set(edges["src"]) | set(edges["dst"]) <= set(prices["ticker"])
    assert not prices["in_index"].all(), "fixtures must include a late index joiner"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda df: df.drop(columns="close"), "missing columns"),
        (lambda df: df.assign(extra=1), "unexpected columns"),
        (lambda df: pd.concat([df, df.head(1)]), "duplicate keys"),
        (lambda df: df.assign(close=df["close"].where(df.index > 0)), "nulls"),
        (lambda df: df.assign(volume=df["volume"].astype("int64")), "dtype"),
        (lambda df: df.assign(date=df["date"] + pd.Timedelta(hours=16)), "time component"),
    ],
)
def test_validate_rejects(tables, mutate, message):
    with pytest.raises(SchemaError, match=message):
        validate(mutate(tables["prices"]), "prices")


def test_validate_rejects_unknown_edge_type(tables):
    edges = tables["edges"].copy()
    edges.loc[0, "edge_type"] = "supplier"
    with pytest.raises(SchemaError, match="edge_type"):
        validate(edges, "edges")


def _schema_md_tables() -> dict[str, dict]:
    """Parse SCHEMA.md: '## `name`' sections, '**Key:**' lines and '| `col` | dtype | nullable |' rows."""
    text = (REPO_ROOT / "SCHEMA.md").read_text(encoding="utf-8")
    out = {}
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        m = re.match(r"`(\w+)`", section)
        if not m:
            continue
        key_line = re.search(r"^\*\*Key:\*\*(.*)$", section, flags=re.M)
        cols = []
        for row in re.findall(r"^\|(.+)\|$", section, flags=re.M):
            cells = [c.strip() for c in row.split("|")]
            names = re.findall(r"`(\w+)`", cells[0])
            if not names or cells[1] in ("dtype", "---"):
                continue
            if len(names) == 2 and names[0] == "emb_0":
                names = [f"emb_{i}" for i in range(int(names[1].split("_")[1]) + 1)]
            cols += [(n, cells[1], cells[2] == "yes") for n in names]
        out[m.group(1)] = {"columns": cols, "key": tuple(re.findall(r"`(\w+)`", key_line.group(1)))}
    return out


def test_schema_md_matches_code():
    md = _schema_md_tables()
    assert sorted(md) == sorted(TABLES)
    for name, table in TABLES.items():
        assert md[name]["columns"] == [(c.name, c.dtype, c.nullable) for c in table.columns], name
        assert md[name]["key"] == table.key, name
    assert len([c for c in md["news_features"]["columns"] if c[0].startswith("emb_")]) == EMB_DIM


def test_forward_looking_columns_are_marked_and_never_reused_as_inputs():
    # labels hold the future; a feature sharing a name with one would leak it through a merge.
    from src.data.indicators import FEATURES

    future = [c for c in TABLES["labels"].column_names if c not in ("date", "ticker")]
    assert all(c.startswith(("fwd_", "y_")) for c in future), future
    assert not set(FEATURES) & set(future)
