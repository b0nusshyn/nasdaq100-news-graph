"""Task 3.1 (#8): static `sector` edges from the GICS classification."""

import pandas as pd
import pytest

from src.core.schema import validate
from src.graph.build_edges import check_gics_coverage, load_gics, sector_edges

SEMIS = "Semiconductors & Semiconductor Equipment"

# NVDA and AMAT share a GICS industry but not a sub-industry: the chip-designer /
# equipment-supplier link the project wants the graph to carry.
GICS = pd.DataFrame(
    [
        ("NVDA", "Information Technology", SEMIS, "Semiconductors"),
        ("INTC", "Information Technology", SEMIS, "Semiconductors"),
        ("AMAT", "Information Technology", SEMIS, "Semiconductor Materials & Equipment"),
        ("MSFT", "Information Technology", "Software", "Systems Software"),
        ("ADBE", "Information Technology", "Software", "Application Software"),
        ("AMGN", "Health Care", "Biotechnology", "Biotechnology"),
    ],
    columns=["ticker", "gics_sector", "gics_industry", "gics_sub_industry"],
)


HEADER = "ticker,gics_sector,gics_industry,gics_sub_industry\n"


def write_csv(tmp_path, rows: str):
    path = tmp_path / "gics.csv"
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def pairs(edges: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(edges["src"], edges["dst"], strict=True))


def test_firms_in_same_industry_are_linked_both_ways():
    assert pairs(sector_edges(GICS)) == {
        ("NVDA", "INTC"), ("INTC", "NVDA"),
        ("NVDA", "AMAT"), ("AMAT", "NVDA"),
        ("INTC", "AMAT"), ("AMAT", "INTC"),
        ("MSFT", "ADBE"), ("ADBE", "MSFT"),
    }


def test_sub_industry_level_splits_designers_from_equipment_makers():
    assert pairs(sector_edges(GICS, level="gics_sub_industry")) == {
        ("NVDA", "INTC"), ("INTC", "NVDA"),
    }


def test_load_gics_maps_old_symbols_to_current_ticker(tmp_path):
    path = write_csv(tmp_path, "fb,Communication Services,Interactive Media & Services,Interactive Media & Services\n")
    assert list(load_gics(path)["ticker"]) == ["META"]


def test_load_gics_rejects_a_company_listed_twice(tmp_path):
    row = "Communication Services,Interactive Media & Services,Interactive Media & Services\n"
    path = write_csv(tmp_path, f"FB,{row}META,{row}")
    with pytest.raises(ValueError, match="META"):
        load_gics(path)


def test_load_gics_rejects_unclassified_companies(tmp_path):
    # Left in, two blank industries would merge as equal and link unrelated firms.
    path = write_csv(tmp_path, "AAA,Health Care,,Biotechnology\nBBB,Industrials,,\n")
    with pytest.raises(ValueError, match="AAA.*BBB"):
        load_gics(path)


def test_every_former_member_needs_a_classification():
    membership = pd.DataFrame({
        "ticker": ["NVDA", "XLNX", "SGEN"],  # XLNX, SGEN: left the index in 2022 and 2023
        "start": pd.to_datetime(["2015-01-02", "2015-01-02", "2015-01-02"]),
        "end": pd.to_datetime([None, "2022-02-11", "2023-12-14"]),
    })
    with pytest.raises(ValueError, match=r"\['SGEN', 'XLNX'\]"):
        check_gics_coverage(GICS, membership)


def test_full_coverage_passes():
    membership = pd.DataFrame({"ticker": ["NVDA", "AMGN"], "start": pd.to_datetime(["2015-01-02"] * 2)})
    check_gics_coverage(GICS, membership)


def test_edges_are_static_unweighted_and_match_the_contract():
    edges = sector_edges(GICS)
    validate(edges, "edges")
    assert set(edges["fold"]) == {-1}
    assert set(edges["edge_type"]) == {"sector"}
    assert set(edges["weight"]) == {1.0}
