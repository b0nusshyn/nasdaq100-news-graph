import pandas as pd

from src.data.audit import missing_price_rows
from src.data.fetch_prices import add_in_index


def test_missing_requests_include_entire_ticker_and_holiday_boundary():
    members = pd.DataFrame({"ticker": ["AAA", "BBB"], "start": pd.to_datetime(["2021-01-15"] * 2),
                            "end": pd.to_datetime(["2021-01-20"] * 2)})
    raw = pd.DataFrame({"date": pd.to_datetime(["2021-01-15", "2021-01-20"]), "ticker": ["AAA"] * 2,
                        "open": [10.] * 2, "high": [11.] * 2, "low": [9.] * 2,
                        "close": [10.] * 2, "volume": [100.] * 2})
    prices = add_in_index(raw, members)
    missing = missing_price_rows(prices, members, "2021-01-15", "2021-01-20")
    # Jan 18 (MLK Day) is not a missing trading session.
    assert missing.groupby("ticker").size().to_dict() == {"AAA": 1, "BBB": 3}
    assert missing.loc[missing.ticker.eq("AAA"), "date"].iloc[0] == pd.Timestamp("2021-01-19")


def test_audit_refuses_stale_membership_flags(monkeypatch):
    import pytest

    from src.data import audit
    from src.data.build_labels import make_labels

    members = pd.DataFrame({"ticker": ["AAA"], "start": [pd.Timestamp("2020-01-01")], "end": [pd.NaT]})
    raw = pd.DataFrame({"date": [pd.Timestamp("2021-01-04")], "ticker": ["AAA"],
                        "open": [10.], "high": [11.], "low": [9.], "close": [10.], "volume": [100.]})
    prices = add_in_index(raw, members).assign(in_index=False)
    tables = {"prices": prices, "membership": members, "labels": make_labels(prices)}
    monkeypatch.setattr(audit, "read_table", lambda name: tables[name])
    monkeypatch.setattr("sys.argv", ["audit"])
    with pytest.raises(ValueError, match="stale"):
        audit.main()


def test_nontrading_classification_preserves_missing_keys_and_requires_evidence():
    import pytest

    from src.data.audit import classify_missing_rows

    missing = pd.DataFrame({"date": pd.to_datetime(["2022-02-14", "2022-02-18", "2022-02-18"]),
                            "ticker": ["XLNX", "XLNX", "AAA"]})
    events = pd.DataFrame({"ticker": ["XLNX"], "start": ["2022-02-14"], "end": ["2022-02-18"],
                           "event": ["Acquisition halt"], "source_url": ["https://example.org/halt"],
                           "membership_source_url": ["https://example.org/index-removal"]})
    result = classify_missing_rows(missing, events)
    pd.testing.assert_frame_equal(result[["date", "ticker"]], missing)
    assert result.classification.tolist() == ["documented_nontrading_member_day"] * 2 + ["unexplained_missing_quote"]
    events.loc[0, "membership_source_url"] = ""
    with pytest.raises(ValueError, match="evidence"):
        classify_missing_rows(missing, events)
