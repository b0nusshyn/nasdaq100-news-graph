"""Task 3.5 (#28): `build_edges` merges all three edge types into `edges.parquet`.

A full build replaces every fold; `--fold N` replaces one fold and warns, because the file then
mixes builds. `edges_manifest.json` records what built each (fold, edge type).
"""

import json
import shutil

import pandas as pd
import pytest

from src.core.io import read_table, write_table
from src.core.paths import DATA_DIR_ENV, FIXTURES_DIR
from src.core.schema import conform
from src.graph import build_edges as be
from src.graph.build_edges import MANIFEST_NAME, build_edges_file, check_edges, summary_text

# The committed fixtures cover 60 trading days: 30 training days, a 5-day embargo and 10 test
# days give folds 0 and 1.
SHORT_FOLDS = {"period": {"start": "2021-01-01", "end": "2021-12-31"}, "train_days": 30, "test_days": 10,
               "step_days": 10, "expanding": False, "embargo_days": 5}
SHORT_PARAMS = {"corr": {"min_overlap": 20}, "comention": {"min_count": 1}}
GICS = "ticker,gics_sector,gics_industry,gics_sub_industry\n" + "".join(
    f"{t},{s},{s} industry,{s} sub\n" for t, s in
    [("AAA", "tech"), ("BBB", "tech"), ("CCC", "tech"), ("DDD", "health"), ("EEE", "health"), ("FFF", "health")]
)


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A data directory with the fixture inputs and no `edges.parquet` yet."""
    for name in ("prices", "membership", "article_tickers"):
        shutil.copy(FIXTURES_DIR / f"{name}.parquet", tmp_path / f"{name}.parquet")
    (tmp_path / "gics.csv").write_text(GICS, encoding="utf-8")
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(be, "_code_version", lambda: {"git_commit": "abc", "git_dirty": False})
    return tmp_path


def build(data, types=be.EDGE_TYPES, fold=None, params=SHORT_PARAMS):
    return build_edges_file(types, fold, cfg=SHORT_FOLDS, params=params, gics_path=data / "gics.csv")


def manifest(data) -> dict:
    m = json.loads((data / MANIFEST_NAME).read_text(encoding="utf-8"))
    return {(e["fold"], e["edge_type"]): e for e in m["entries"]}


def rows(edges: pd.DataFrame, fold: int, edge_type: str) -> pd.DataFrame:
    e = edges[(edges["fold"] == fold) & (edges["edge_type"] == edge_type)]
    return e.reset_index(drop=True)


# --------------------------------------------------------------------------- full build


def test_full_build_writes_every_fold_of_every_type_and_matches_the_contract(data):
    result = build(data)
    edges = read_table("edges")  # validates against SCHEMA.md
    check_edges(edges)
    pd.testing.assert_frame_equal(edges, result.edges)
    assert result.n_folds == 2
    assert set(zip(edges["fold"], edges["edge_type"], strict=True)) == {
        (-1, "sector"), (0, "comention"), (0, "corr"), (1, "comention"), (1, "corr"),
    }
    assert result.warnings == []


def test_manifest_records_parameters_inputs_and_window_per_fold_and_type(data):
    result = build(data)
    entries = manifest(data)
    assert set(entries) == {(-1, "sector"), (0, "comention"), (0, "corr"), (1, "comention"), (1, "corr")}
    corr = entries[(1, "corr")]
    assert corr["params"] == be.default_params()["corr"] | SHORT_PARAMS["corr"]
    assert corr["train_window"]["days"] == 30
    assert corr["walk_forward"]["train_days"] == 30
    assert set(corr["inputs"]) == {"prices"}
    assert set(entries[(0, "comention")]["inputs"]) == {"article_tickers", "prices"}
    assert entries[(-1, "sector")]["train_window"] is None
    assert corr["rows"] == len(rows(result.edges, 1, "corr"))
    assert json.loads((data / MANIFEST_NAME).read_text())["edges_sha256"] == be._sha256(data / "edges.parquet")


def test_summary_lists_every_fold_and_the_checks(data):
    text = summary_text(build(data).edges, 2)
    assert "Folds present: -1, 0, 1 (walk_forward.yaml gives folds 0–1: all present)" in text
    assert "Checks passed" in text


# --------------------------------------------------------------------------- one fold


def test_one_fold_run_replaces_only_that_folds_rows(data):
    before = build(data).edges
    after = build(data, types=["corr"], fold=1, params={"corr": {"min_overlap": 20, "top_k": 1}}).edges
    for fold, t in [(0, "corr"), (0, "comention"), (1, "comention"), (-1, "sector")]:
        pd.testing.assert_frame_equal(rows(after, fold, t), rows(before, fold, t))
    assert len(rows(after, 1, "corr")) < len(rows(before, 1, "corr"))


def test_one_fold_run_warns_that_the_file_mixes_builds(data):
    build(data)
    warnings = build(data, fold=0).warnings
    assert warnings[0].startswith("--fold 0:")
    assert "sector rows (fold -1) that every fold shares were rewritten" in warnings[0]
    assert "Rebuild in full" in warnings[0]


def test_one_fold_run_without_sector_leaves_the_shared_rows_alone(data):
    build(data)
    sector_entry = manifest(data)[(-1, "sector")]
    warnings = build(data, types=["corr"], fold=0).warnings
    assert manifest(data)[(-1, "sector")] == sector_entry
    assert "sector" not in warnings[0]


def test_folds_built_with_different_parameters_are_reported(data):
    build(data)
    warnings = build(data, types=["corr"], fold=1, params={"corr": {"min_overlap": 20, "top_k": 1}}).warnings
    [conflict] = [w for w in warnings if w.startswith("corr parameters differ")]
    assert "fold 0 corr" in conflict and "fold 1 corr" in conflict


def test_full_rebuild_clears_a_mixed_build(data):
    build(data)
    build(data, types=["corr"], fold=1, params={"corr": {"min_overlap": 20, "top_k": 1}})
    assert build(data).warnings == []


def test_rows_with_no_manifest_entry_are_reported(data):
    build(data)
    (data / MANIFEST_NAME).unlink()
    warnings = build(data, types=["corr"], fold=0).warnings
    [unknown] = [w for w in warnings if w.startswith("no manifest entry")]
    assert "fold -1 sector" in unknown and "fold 1 comention" in unknown and "fold 0 corr" not in unknown


def test_an_edges_file_changed_since_its_manifest_loses_its_old_entries(data):
    edges = build(data).edges
    write_table(edges[edges["edge_type"] != "comention"], "edges")
    warnings = build(data, types=["corr"]).warnings
    assert any("changed since" in w for w in warnings)
    assert set(manifest(data)) == {(0, "corr"), (1, "corr")}


def test_a_fold_outside_the_walk_forward_settings_is_rejected(data):
    with pytest.raises(ValueError, match="fold 2 does not exist"):
        build(data, fold=2)


def test_a_failed_check_writes_nothing(data):
    build(data)
    edges = read_table("edges")
    one_way = edges[~((edges["edge_type"] == "sector") & (edges["src"] == "AAA") & (edges["dst"] == "BBB"))]
    write_table(one_way, "edges")
    stamp = (data / "edges.parquet").stat().st_mtime_ns
    with pytest.raises(ValueError, match="not stored in both directions"):
        build(data, types=["corr"])
    assert (data / "edges.parquet").stat().st_mtime_ns == stamp


# --------------------------------------------------------------------------- check_edges


def edge_table(*edges) -> pd.DataFrame:
    df = pd.DataFrame(edges, columns=["fold", "src", "dst", "edge_type", "weight"])
    return conform(df, "edges")


GOOD = [(-1, "AAA", "BBB", "sector", 1.0), (-1, "BBB", "AAA", "sector", 1.0),
        (0, "AAA", "CCC", "corr", 0.3), (0, "CCC", "AAA", "corr", 0.3)]


def test_check_edges_accepts_a_symmetric_graph():
    check_edges(edge_table(*GOOD))


@pytest.mark.parametrize("bad, message", [
    ([(0, "AAA", "AAA", "corr", 0.3)], "self-loops"),
    ([(0, "AAA", "DDD", "corr", 0.3)], "not stored in both directions"),
    ([(0, "AAA", "DDD", "corr", 0.3), (0, "DDD", "AAA", "corr", 0.4)], "different weights"),
    ([(0, "GOOG", "GOOGL", "corr", 0.9), (0, "GOOGL", "GOOG", "corr", 0.9)], "share classes"),
    ([(0, "AAA", "DDD", "sector", 1.0), (0, "DDD", "AAA", "sector", 1.0)], "sector rows outside fold -1"),
    ([(-1, "AAA", "DDD", "corr", 0.3), (-1, "DDD", "AAA", "corr", 0.3)], "negative fold"),
    ([(0, "AAA", "DDD", "comention", 3.0), (0, "DDD", "AAA", "comention", 3.0)], "comention weights outside"),
    ([(-1, "AAA", "BBB", "sector", 1.0)], "duplicate"),
])
def test_check_edges_rejects_graphs_that_break_the_schema(bad, message):
    with pytest.raises(ValueError, match=message):
        check_edges(edge_table(*GOOD, *bad))
