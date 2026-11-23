from zipfile import ZipFile

import pandas as pd
import pandas_market_calendars as mcal
import pytest

from experiments.sec_lagged_proxy.build import HERE, assemble, eligible_session, financial_features
from experiments.sec_lagged_proxy.index_quarter import index_submissions


def test_full_session_lag_holiday_and_early_close():
    schedule = mcal.get_calendar("NASDAQ").schedule("2023-11-20", "2023-12-08")
    assert schedule.loc["2023-11-24", "market_close"].hour == 18
    args = ("2023-11-21", "2023-11-22T19:00:00-05:00", None, schedule.index)
    assert eligible_session(*args, 2) == pd.Timestamp("2023-11-28")
    assert eligible_session(*args, 5) == pd.Timestamp("2023-12-01")
    assert eligible_session("2023-11-22", None, None, schedule.index, 2) == pd.Timestamp("2023-11-28")
    with pytest.raises(ValueError):
        eligible_session("2023-11-22", "2023-11-22T19:00:00", None, schedule.index, 2)


def test_real_facts_preserve_keys_and_amendment_timing():
    dtypes = {"cik": str}
    filings = pd.read_csv(HERE / "filings.csv", dtype=dtypes, parse_dates=["filed_date", "report_period"])
    filings = filings.loc[filings.ticker.eq("MSFT")].copy()
    facts = pd.read_csv(HERE / "facts.csv", dtype=dtypes, parse_dates=["start", "end"])
    ids = pd.read_csv(HERE / "identities.csv", dtype=dtypes, parse_dates=["start", "end"])
    f = next(filings.itertuples(index=False))
    summary = financial_features(facts, f)
    assert summary["operating_margin"] == pytest.approx(26895 / 56517)
    assert summary["revenue_growth_yoy"] == pytest.approx(56517 / 50122 - 1)
    assert summary["net_margin_change_yoy"] == pytest.approx(22291 / 56517 - 17556 / 50122)
    assert financial_features(facts.assign(unit="EUR_millions"), f)["operating_margin"] is None
    assert financial_features(pd.concat([facts, facts]), f)["operating_margin"] is None

    amend = filings.assign(accession="0000950170-23-999999", form="10-Q/A",
                           filed_date=pd.Timestamp("2023-11-01"), accepted_local=None)
    sessions = mcal.get_calendar("NASDAQ").schedule("2023-10-20", "2023-11-10").index
    dates = pd.to_datetime(["2023-10-26", "2023-10-27", "2023-11-03", "2023-11-06", "2023-11-06"])
    keys = pd.DataFrame({"date": dates, "ticker": ["MSFT"] * 4 + ["UNKNOWN"]})
    output, _ = assemble(keys, ids, pd.concat([amend, filings]), facts, sessions, 2)
    pd.testing.assert_frame_equal(output[keys.columns], keys)
    assert output.accession.iloc[1:4].tolist() == [f.accession, f.accession, amend.accession.iloc[0]]
    assert output.accession.iloc[[0, 4]].isna().all()
    assert output.operating_margin.iloc[1:3].eq(summary["operating_margin"]).all()
    assert pd.isna(output.operating_margin.iloc[3])
    assert not output.strict_public_verified.any()


def test_stream_local_zip(tmp_path):
    archive = tmp_path / "2023q4.zip"
    with ZipFile(archive, "w") as destination:
        destination.writestr("sub.txt", "cik\tadsh\tform\tfiled\taccepted\tperiod\tname\n"
                             "789019\t0000950170-23-054855\t10-Q\t20231024\t20231024160740\t"
                             "20230930\tMicrosoft\n")
    entries = list(index_submissions(archive, {"0000789019"}))
    assert len(entries) == 1 and entries[0]["review_state"] == "verify_original_accession"
    assert not list(index_submissions(archive, {"0000320193"}))
