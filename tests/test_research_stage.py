import pandas as pd
import pytest

from src.data.stage_research_prices import compare_candidates, prepare_daily_candidates


def test_retired_2018_source_ids_preserve_membership_identity():
    members = pd.DataFrame({"ticker": ["CA", "ESRX"],
                            "start": pd.to_datetime(["2018-01-01"] * 2),
                            "end": pd.to_datetime(["2018-11-02", "2018-12-21"])})
    frame = pd.DataFrame({"date": pd.to_datetime(["2018-11-02", "2018-12-20"]),
                          "ticker": ["CA~201811", "ESRX~201812"], "aopen": [40., 90.],
                          "ahigh": [41., 91.], "alow": [39., 89.], "aclose": [40., 90.],
                          "volume": [100., 200.], "source": "alpaca"})
    result = prepare_daily_candidates(frame, members)
    assert result.ticker.tolist() == ["CA", "ESRX"]
    assert result.in_index.all()
    assert result.source_id.tolist() == ["CA~201811", "ESRX~201812"]


def test_source_adjusted_fields_and_old_issuer_mapping_are_explicit():
    members = pd.DataFrame({"ticker": ["TFCFA", "FOXA"],
                            "start": pd.to_datetime(["2018-01-01", "2019-03-19"]),
                            "end": pd.to_datetime(["2019-03-18", None])})
    frame = pd.DataFrame({"date": pd.to_datetime(["2019-03-18", "2019-03-19"]),
                          "ticker": ["FOXA~201903", "FOXA"], "aopen": [50., 30.],
                          "ahigh": [51., 31.], "alow": [49., 29.], "aclose": [50.5, 30.5],
                          "volume": [100., 200.], "source": ["alpaca", "yahoo"]})
    result = prepare_daily_candidates(frame, members)
    assert result.ticker.tolist() == ["TFCFA", "FOXA"]
    assert result.close.tolist() == [50.5, 30.5]
    assert result.in_index.all()
    assert result.source_id.tolist() == ["FOXA~201903", "FOXA"]
    with pytest.raises(ValueError, match="adjusted|input needs"):
        prepare_daily_candidates(frame.drop(columns="aclose"), members)


def test_overlap_comparison_reports_return_disagreement_without_changing_data():
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    old = pd.DataFrame({"date": dates, "ticker": "AAA", "close": [100., 101., 102.], "volume": 100.})
    candidate = old.assign(close=[50., 50.5, 55.], source="alpaca")
    before = old.copy()
    report = compare_candidates(old, candidate)
    assert report.max_overlap_return_difference.iloc[0] > 0.05
    assert report.decision.iloc[0] == "review_required_not_applied"
    pd.testing.assert_frame_equal(old, before)
