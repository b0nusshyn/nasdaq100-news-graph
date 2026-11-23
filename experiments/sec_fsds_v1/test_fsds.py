"""Offline tests for the FSDS point-in-time pipeline on a synthetic quarterly ZIP."""
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from experiments.sec_fsds_v1 import extract as ex
from experiments.sec_fsds_v1 import features as ft
from experiments.sec_fsds_v1.fetch import quarters

SUB_HEAD = "\t".join(ex.SUB_COLS)
NUM_HEAD = "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tsegments\tvalue\tfootnote"
SESSIONS = mcal.get_calendar("NASDAQ").schedule("2021-06-01", "2023-06-30").index


def sub_row(adsh, cik, form, period, filed, accepted, instance, name="X"):
    vals = dict(adsh=adsh, cik=cik, name=name, sic="", countryinc="US", former="", changed="",
                fye="1231", form=form, period=period, fy="", fp="", filed=filed,
                accepted=accepted, prevrpt="0", instance=instance, nciks="1", aciks="")
    return "\t".join(vals[c] for c in ex.SUB_COLS)


def num_row(adsh, tag, ddate, qtrs, uom, value, coreg="", segments="", version="us-gaap/2021"):
    return "\t".join([adsh, tag, version, coreg, ddate, str(qtrs), uom, segments, str(value), ""])


A1, A2, A3, AM, B1 = (f"0000000001-22-00000{i}" for i in (1, 2, 3, 4, 5))


@pytest.fixture()
def archive(tmp_path) -> Path:
    subs = [
        # FB-era 10-Q with full facts, accepted Fri 2022-07-29 17:00 ET
        sub_row(A1, "1", "10-Q", "20220630", "20220729", "2022-07-29 17:00:00.0", "fb-20220630_htm.xml"),
        # later META 10-Q with NO usable facts
        sub_row(A2, "1", "10-Q", "20220930", "20221026", "2022-10-26 16:30:00.0", "meta-20220930_htm.xml"),
        # amendment with different (restated) facts: must never be used
        sub_row(AM, "1", "10-Q/A", "20220630", "20220815", "2022-08-15 09:00:00.0", "meta-20220630_htm.xml"),
        # a different issuer that later re-uses the prefix "zz"
        sub_row(B1, "2", "10-K", "20211231", "20220210", "2022-02-10 08:00:00.0", "zz-20211231_htm.xml"),
        sub_row(A3, "3", "10-K", "20221231", "20230215", "2023-02-15 08:00:00.0", "zz-20221231_htm.xml"),
        # cik 2 files again after cik 3 started: concurrent use of "zz"
        sub_row("0000000002-23-000007", "2", "10-Q", "20230331", "20230510", "2023-05-10 08:00:00.0",
                "zz-20230331_htm.xml"),
        sub_row("0000000009-22-000009", "9", "8-K", "20220630", "20220701", "2022-07-01 08:00:00.0", "x-1_htm.xml"),
    ]
    nums = [
        num_row(A1, "Revenues", "20220630", 1, "USD", 120), num_row(A1, "Revenues", "20210630", 1, "USD", 100),
        num_row(A1, "OperatingIncomeLoss", "20220630", 1, "USD", 30),
        num_row(A1, "OperatingIncomeLoss", "20210630", 1, "USD", 20),
        num_row(A1, "NetIncomeLoss", "20220630", 1, "USD", 24), num_row(A1, "NetIncomeLoss", "20210630", 1, "USD", 10),
        num_row(A1, "EarningsPerShareDiluted", "20220630", 1, "USD", 3),  # FSDS: currency only
        num_row(A1, "EarningsPerShareDiluted", "20210630", 1, "USD", 1),
        num_row(A1, "Revenues", "20220630", 2, "USD", 999),                   # YTD: ignored
        num_row(A1, "Revenues", "20220630", 1, "USD", 5, segments="Seg=A"),   # segment: dropped
        num_row(A1, "Revenues", "20220630", 1, "USD", 7, coreg="Sub"),        # co-registrant: dropped
        num_row(A1, "Revenues", "20220630", 1, "USD", 9, version=A1),         # custom tag: dropped
        num_row(AM, "Revenues", "20220630", 1, "USD", 50), num_row(AM, "Revenues", "20210630", 1, "USD", 100),
        num_row(B1, "Revenues", "20211231", 4, "EUR", 10), num_row(B1, "Revenues", "20201231", 4, "EUR", 8),
        num_row(B1, "OperatingIncomeLoss", "20211231", 4, "USD", 1),          # currency mismatch
    ]
    path = tmp_path / "2022q3.zip"
    with ZipFile(path, "w") as z:
        z.writestr("sub.txt", "\n".join([SUB_HEAD, *subs]) + "\n")
        z.writestr("num.txt", "\n".join([NUM_HEAD, *nums]) + "\n")
        z.writestr("tag.txt", "")
    return path


@pytest.fixture()
def built(archive, tmp_path):
    ex.extract(archive, tmp_path / "extract")
    sub, num, _ = ft.load_extracts(tmp_path / "extract")
    filings = ft.filings_ledger(sub, SESSIONS)
    facts = ft.filing_facts(filings, num)
    return filings, facts


def test_extract_filters(archive):
    sub, num = ex.filter_zip(archive)
    assert set(sub.form) == {"10-Q", "10-Q/A", "10-K"}
    a1 = num.loc[num.adsh.eq(A1) & num.tag.eq("Revenues")]
    assert sorted(a1.value) == [100, 120]  # YTD, segment, coreg, custom excluded


def test_lag_two_complete_sessions(built):
    filings, _ = built
    f = filings.set_index("adsh")
    # filed Fri 29 Jul -> Mon 1 Aug, Tue 2 Aug complete -> eligible Wed 3 Aug
    assert f.loc[A1, "eligible_date"] == pd.Timestamp("2022-08-03")
    assert not f.loc[AM, "original"]


def test_facts_from_same_filing(built):
    _, facts = built
    a = facts.set_index("adsh").loc[A1]
    assert a.sec_rev_growth == pytest.approx(0.2)
    assert a.sec_op_margin == pytest.approx(0.25)
    assert a.sec_op_margin_chg == pytest.approx(0.25 - 0.2)
    assert a.sec_net_margin_chg == pytest.approx(0.2 - 0.1)
    assert a.sec_eps_chg == pytest.approx(0.5)
    b = facts.set_index("adsh").loc[B1]
    assert np.isnan(b.sec_op_margin) and b.sec_rev_growth == pytest.approx(0.25)
    assert AM not in set(facts.adsh)


def test_store_states_and_point_in_time(built):
    filings, facts = built
    keys = pd.DataFrame({"ticker": "META", "date": pd.to_datetime(
        ["2022-07-28", "2022-08-02", "2022-08-03", "2022-08-16", "2022-10-31", "2022-11-01"])})
    s = ft.build_store(keys, filings, facts, SESSIONS).set_index("date")
    # before first filing: no identity (no future filing used)
    assert not s.loc["2022-07-28", "sec_has_issuer"]
    # identity known from filing date, features only once eligible
    assert s.loc["2022-08-02", "sec_has_issuer"] and not s.loc["2022-08-02", "sec_has_event"]
    assert s.loc["2022-08-03", "sec_sessions_since_filing"] == 0
    assert s.loc["2022-08-03", "sec_rev_growth"] == pytest.approx(0.2)
    # amendment (restated revenue 50) never replaces the original
    assert s.loc["2022-08-16", "sec_rev_growth"] == pytest.approx(0.2)
    # A2 (no facts) eligible 31 Oct: event state moves, fundamentals persist
    assert s.loc["2022-10-31", "event_accession"] == A2
    assert s.loc["2022-10-31", "fund_accession"] == A1
    assert s.loc["2022-10-31", "sec_rev_growth"] == pytest.approx(0.2)
    ft.check_point_in_time(s.reset_index(), filings)


def test_fundamentals_expire(built):
    filings, facts = built
    late = SESSIONS[SESSIONS.get_loc(pd.Timestamp("2022-08-03")) + ft.MAX_FUND_AGE + 1]
    s = ft.build_store(pd.DataFrame({"ticker": ["META"], "date": [late]}), filings, facts, SESSIONS)
    assert not s.sec_has_fund.iloc[0] and np.isnan(s.sec_rev_growth.iloc[0])


def test_identity_conflict_and_reuse(built):
    filings, facts = built
    keys = pd.DataFrame({"ticker": "ZZ", "date": pd.to_datetime(["2022-06-01", "2023-03-01", "2023-06-01"])})
    s = ft.map_identity(keys, filings).set_index("date")
    assert s.loc["2022-06-01", "cik"] == "0000000002"
    # clean succession (2 stopped before 3 started): newest filer, no conflict
    assert s.loc["2023-03-01", "cik"] == "0000000003" and not s.loc["2023-03-01", "identity_conflict"]
    # 2 files again after 3 started: concurrent filers -> conflict, unmapped
    assert s.loc["2023-06-01", "identity_conflict"] and pd.isna(s.loc["2023-06-01", "cik"])
    # a reviewed block removes the reused-symbol filer entirely
    b = ft.map_identity(keys, filings, ({("ZZ", "0000000002")}, {})).set_index("date")
    assert pd.isna(b.loc["2022-06-01", "cik"]) and b.loc["2023-06-01", "cik"] == "0000000003"


def test_review_file_and_alias():
    blocked, aliases = ft.load_review()
    assert ("GEN", "0001351051") in blocked and "symc" in ft.ticker_prefixes("GEN", aliases)


def test_prefixes():
    assert ft.ticker_prefixes("META") == {"meta", "fb"}
    assert ft.ticker_prefixes("GOOG") == {"goog", "googl"}
    assert ft.instance_prefix("aapl-20230701_htm.xml") == "aapl"
    assert ft.instance_prefix("0001-x.xml") is None
    assert ft.prefix_matches("csx03312010", {"csx"}) and ft.prefix_matches("goog10", {"goog"})
    assert not ft.prefix_matches("a10", {"a"}) and not ft.prefix_matches("costco", {"cost"})


def test_quarters():
    q = quarters()
    assert q[0] == "2018q3" and q[-1] == "2023q4" and len(q) == 22
