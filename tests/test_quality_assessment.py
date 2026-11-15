import json
import sys

import pandas as pd
import pytest

from src.core.io import write_table
from src.data import quality
from src.data.audit import classify_missing_rows
from src.data.build_labels import make_labels


def sample():
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-05"]),
                           "ticker": "AAA", "open": [10., 20., 21.], "high": [11., 21., 22.],
                           "low": [9., 19., 20.], "close": [10., 20., 21.], "volume": [0., 100., 100.],
                           "in_index": True})
    members = pd.DataFrame({"ticker": ["AAA"], "start": [pd.Timestamp("2020-12-31")],
                            "end": [pd.Timestamp("2021-01-05")]})
    return prices, make_labels(prices), members


def test_bounds_filter_anomalies_and_balance_but_keep_prior_close():
    prices, labels, members = sample()
    labels.loc[0, "ret_1d"] = -0.9  # outside the requested scope
    report = quality.quality_report(prices, labels, members, start="2021-01-04", end="2021-01-05")
    assert "zero_volume" not in set(report["check"])
    assert "label_return_mismatch" not in set(report["check"])
    jump = report.loc[report["check"].eq("adjustment_review")]
    assert jump.date.tolist() == [pd.Timestamp("2021-01-04")]
    assert jump.value.tolist() == [1.]
    assert report.loc[report["check"].eq("class_balance"), "year"].eq(2021).all()


def test_documented_halt_does_not_hide_an_unexplained_gap():
    missing = pd.DataFrame({"ticker": ["AAA", "BBB"], "date": pd.to_datetime(["2021-01-04"] * 2)})
    events = pd.DataFrame({"ticker": ["AAA"], "start": ["2021-01-04"], "end": ["2021-01-04"],
                           "event": ["halt"], "source_url": ["https://example.org/halt"],
                           "membership_source_url": ["https://example.org/removal"]})
    classified = classify_missing_rows(missing, events)
    report = pd.DataFrame(columns=quality.REPORT_COLUMNS)
    result = quality.assess_quality(report, classified)
    assert result["status"] == "FAIL"
    assert result["unexplained_missing_member_sessions"] == 1
    assert quality.assess_quality(report, classified.iloc[:1])["status"] == "PASS_WITH_FLAGS"


def test_strict_cli_writes_failure_evidence_and_reports_recovered_download(monkeypatch, tmp_path):
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    prices, labels, members = sample()
    # A missing terminal member session must fail even if internal continuity looks fine.
    write_table(prices.iloc[:2], "prices")
    write_table(make_labels(prices.iloc[:2]), "labels")
    write_table(members, "membership")
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"ticker": ["AAA"], "reason": ["original Yahoo failure"]}).to_csv(
        raw / "failed_tickers.csv", index=False)
    # Explicit upper bound exposes the absent trailing session.
    monkeypatch.setattr(sys, "argv", ["quality", "--end", "2021-01-05", "--strict"])
    with pytest.raises(SystemExit) as error:
        quality.main()
    assert error.value.code == 1
    summary = json.loads((raw / "quality_summary.json").read_text())
    assert summary["unexplained_missing_member_sessions"] == 1
    failed = pd.read_csv(raw / "quality_download_status.csv")
    assert failed.combined_price_rows.iloc[0] == 2
    assert not failed.currently_absent_in_audit.iloc[0]
    balance = pd.read_csv(raw / "quality_class_balance.csv")
    assert (balance.labelled + balance["null"]).eq(balance.rows).all()
    assert (balance.up + balance.down).eq(balance.labelled).all()


def test_stale_zero_volume_review_and_new_jump_are_not_silently_accepted():
    report = pd.DataFrame([{"check": "zero_volume", "ticker": "FB", "date": None, "value": 2},
                           {"check": "adjustment_review", "ticker": "AAA", "date": "2021-01-04", "value": .8}])
    reviewed = pd.DataFrame({"check": ["zero_volume"], "ticker": ["META"], "date": [None],
                             "flagged_rows": [1], "status": ["retained_flag"], "notes": ["Reviewed one day"]})
    result = quality.review_findings(report, reviewed)
    assert result.status.eq("unreviewed").all()
    reviewed.loc[0, "flagged_rows"] = 2
    assert quality.review_findings(report, reviewed).status.tolist() == ["retained_flag", "unreviewed"]
