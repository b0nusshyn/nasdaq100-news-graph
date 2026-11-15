"""Task 3.1 (#8): the committed GICS table behind the static `sector` edges.

Classification is the pre-March-2023 GICS assignment (in force for most of 2019-2023): firms the
2023 revision moved are entered with their old code. Codes that were only renamed keep their
current name, since a rename regroups nothing.
"""

import pandas as pd
import pytest

from src.core.paths import table_path
from src.graph.build_edges import GICS_CSV, check_gics_coverage, load_gics

INDUSTRIES_CSV = GICS_CSV.with_name("gics_industries.csv")
SOURCES = {"wikipedia_sp500", "wikipedia_sp500_pre2023", "manual", "manual_out_of_window"}
SEMIS = "Semiconductors & Semiconductor Equipment"


@pytest.fixture(scope="module")
def gics():
    return load_gics()


@pytest.fixture(scope="module")
def industries():
    return pd.read_csv(INDUSTRIES_CSV, dtype="string").set_index("gics_sub_industry")["gics_industry"]


def members(gics, industry):
    return set(gics.loc[gics["gics_industry"] == industry, "ticker"])


def test_chip_designers_and_equipment_makers_share_an_industry(industries):
    assert industries["Semiconductors"] == SEMIS
    assert industries["Semiconductor Materials & Equipment"] == SEMIS


def test_every_industry_follows_from_its_sub_industry(gics, industries):
    unmapped = sorted(set(gics["gics_sub_industry"]) - set(industries.index))
    assert not unmapped, f"sub-industries missing from gics_industries.csv: {unmapped}"
    wrong = gics[gics["gics_industry"] != gics["gics_sub_industry"].map(industries)]
    assert wrong.empty, wrong[["ticker", "gics_sub_industry", "gics_industry"]].to_string()


def test_semiconductor_industry_holds_the_obvious_members(gics):
    assert {"NVDA", "AMD", "AMAT", "LRCX"} <= members(gics, SEMIS)


def test_firms_moved_in_march_2023_keep_their_earlier_code(gics):
    by = gics.set_index("ticker")
    for t in ("PYPL", "ADP", "PAYX", "FISV"):
        assert by.loc[t, "gics_sub_industry"] == "Data Processing & Outsourced Services", t
        assert by.loc[t, "source"] == "wikipedia_sp500_pre2023", t
    # Online retail and online travel were one industry before 2023.
    assert {"AMZN", "EBAY", "BKNG", "EXPE"} <= members(gics, "Internet & Direct Marketing Retail")
    assert by.loc["DLTR", "gics_industry"] != by.loc["COST", "gics_industry"]


def test_provenance_is_recorded_for_every_row(gics):
    assert set(gics["source"]) <= SOURCES
    manual = gics["source"] == "manual"
    assert set(gics.loc[manual, "crosscheck"]) <= {"agree", "disagree", "unavailable"}
    assert gics.loc[~manual, "crosscheck"].isna().all()


def test_every_member_is_classified(gics):
    path = table_path("membership")
    if not path.exists():
        pytest.skip("membership.parquet not in the data directory (gitignored; on the team Drive)")
    from src.core.io import read_table

    check_gics_coverage(gics, read_table("membership"))
