import pandas as pd
import pandas_market_calendars as mcal
import pytest

from experiments.sec_pit_v1.features import available_session, build_features, quarterly_margin


def filing(**changes):
    return {
        "cik": "1", "accession": "0000000001-23-000001", "form": "10-Q",
        "period_end": "2023-09-30", "accepted_at": "2023-11-20T16:05:00-05:00",
        "public_by": "2023-11-21T06:00:00-05:00",
        "acceptance_source": "fixture", "public_evidence": "fixture", **changes,
    }


def observations(f):
    row = {"cik": "1", "accession": f["accession"], "unit": "USD",
           "start": "2023-07-01", "end": f["period_end"], "consolidated": True}
    return {"RevenueFromContractWithCustomerExcludingAssessedTax": [{**row, "value": 100}],
            "OperatingIncomeLoss": [{**row, "value": 20}]}


def test_next_session_and_timezone():
    calendar = mcal.get_calendar("NASDAQ").schedule("2023-11-20", "2023-11-30")
    early_close = pd.Timestamp("2023-11-24")
    assert calendar.loc[early_close, "market_close"].hour == 18
    assert available_session("2023-11-23T09:00:00-05:00", "2023-11-23T10:00:00-05:00",
                             calendar.index) == early_close
    assert available_session("2023-11-24T13:01:00-05:00", "2023-11-25T12:00:00-05:00",
                             calendar.index) == pd.Timestamp("2023-11-27")
    with pytest.raises(ValueError):
        available_session("2023-11-23", "2023-11-23T10:00:00-05:00", calendar.index)


def test_accession_missing_coverage_and_amendment():
    original = filing()
    amendment = filing(accession="0000000001-23-000002", form="10-Q/A",
                       accepted_at="2023-11-24T13:01:00-05:00",
                       public_by="2023-11-24T13:02:00-05:00")
    facts = observations(original)
    correction = observations(amendment)
    correction["OperatingIncomeLoss"][0]["value"] = 30
    for tag, values in correction.items():
        facts[tag].extend(values)
    keys = pd.DataFrame({
        "date": pd.to_datetime(["2023-11-21", "2023-11-22", "2023-11-24", "2023-11-27",
                                "2023-11-27"]),
        "ticker": ["A", "A", "A", "A", "B"], "cik": ["1", "1", "1", "1", None],
    }, index=[0, 0, 1, 2, 2])
    result = build_features(keys, [amendment, original], facts)
    pd.testing.assert_frame_equal(result[keys.columns], keys)
    assert result.quarterly_operating_margin.iloc[1:4].tolist() == [0.2, 0.2, 0.3]
    assert result.filing_accession.iloc[[0, 4]].isna().all()
    assert build_features(keys, [filing(public_evidence=None)], facts).filing_accession.isna().all()


def test_fact_context_and_accession_validation():
    f = filing()
    facts = observations(f)
    assert quarterly_margin(facts, f) == 0.2
    facts["OperatingIncomeLoss"][0]["start"] = "2023-01-01"
    assert quarterly_margin(facts, f) is None
    facts["OperatingIncomeLoss"][0]["start"] = "2023-07-01"
    facts["OperatingIncomeLoss"][0]["unit"] = "EUR"
    assert quarterly_margin(facts, f) is None
    facts["OperatingIncomeLoss"][0]["unit"] = "USD"
    assert quarterly_margin(facts, filing(accession="0000000001-23-000002")) is None
    facts["OperatingIncomeLoss"].append({**facts["OperatingIncomeLoss"][0], "value": 999})
    assert quarterly_margin(facts, f) is None
