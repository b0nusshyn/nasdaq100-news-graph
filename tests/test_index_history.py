from io import StringIO
from unittest.mock import Mock

import pandas as pd
import pytest

from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.core.schema import validate
from src.data import build_index_history as history


@pytest.fixture
def parsed():
    root = FIXTURES_DIR / "wikipedia"
    changes = history.parse_changes(pd.read_html(StringIO((root / "history.html").read_text())))
    current = history.parse_current(pd.read_html(StringIO((root / "current.html").read_text())))
    return changes, current


def test_parse_and_reconstruct_rejoin_rename_and_holiday(parsed):
    changes, current = parsed
    assert "gics sector" in current
    assert set(current.ticker) >= {"GOOG", "GOOGL"}
    result = history.membership_intervals(changes, current)
    validate(result, "membership")
    aaa = result.loc[result.ticker == "AAA"].reset_index(drop=True)
    assert list(aaa.start) == [pd.Timestamp("2018-01-01"), pd.Timestamp("2020-01-21")]
    # July 4 is a market holiday; do not use a generic business-day offset.
    assert aaa.end.iloc[0] == pd.Timestamp("2019-07-03")
    assert pd.isna(aaa.end.iloc[1])
    bbb = result.loc[result.ticker == "BBB"].iloc[0]
    assert bbb.start == pd.Timestamp("2019-07-05")
    assert bbb.end == pd.Timestamp("2020-01-17")  # MLK holiday on Monday
    meta = result.loc[result.ticker == "META"]
    assert len(meta) == 1 and meta.start.iloc[0] == pd.Timestamp("2018-01-01")


@pytest.mark.parametrize("added, removed, message", [
    ("ZZZ", None, "not a member afterwards"),
    (None, "AAA", "still a member afterwards"),
])
def test_inconsistent_history_rejected(parsed, added, removed, message):
    _, current = parsed
    changes = pd.DataFrame({"date": ["2020-01-02"], "added": [added], "removed": [removed]})
    with pytest.raises(ValueError, match=message):
        history.membership_intervals(changes, current)


def test_duplicate_current_and_events_rejected(parsed):
    changes, current = parsed
    with pytest.raises(ValueError, match="unique"):
        history.membership_intervals(changes, pd.concat([current, current.head(1)]))
    with pytest.raises(ValueError, match="Duplicate event"):
        history.membership_intervals(pd.concat([changes, changes.head(1)]), current)


def test_counts_and_main_write(monkeypatch, tmp_path):
    current = pd.DataFrame({"ticker": [f"T{i}" for i in range(101)]})
    changes = pd.DataFrame(columns=["date", "added", "removed"])
    monkeypatch.setattr(history, "scrape_current", lambda: current)
    monkeypatch.setattr(history, "scrape_changes", lambda: changes)
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["build_index_history"])
    history.main()
    membership = read_table("membership")
    assert history.validate_member_counts(membership).eq(101).all()
    with pytest.raises(ValueError, match="outside 100"):
        history.validate_member_counts(membership.head(99))


def test_http_header_and_moved_pages(monkeypatch):
    root = FIXTURES_DIR / "wikipedia"
    requested = []

    def get(url, **kwargs):
        assert kwargs["headers"]["User-Agent"]
        assert kwargs["timeout"] == 30
        requested.append(url)
        filename = "history.html" if url == history.CHANGES_URL else "current.html"
        return Mock(text=(root / filename).read_text())

    monkeypatch.setattr("requests.get", get)
    assert len(history.scrape_changes()) == 3
    assert requested == [history.WIKI_URL, history.CHANGES_URL]
    assert len(history.scrape_current()) == 4


def test_unknown_layout_rejected():
    with pytest.raises(ValueError, match="Expected one changes"):
        history.parse_changes([pd.DataFrame({"unexpected": [1]})])


def test_blank_separator_is_ignored_but_undated_event_is_rejected():
    table = pd.DataFrame({"Date": ["2020-01-02", None], "Added": ["AAA", None],
                          "Removed": [None, None]})
    parsed = history.parse_changes([table])
    assert len(parsed) == 1
    table.loc[1, "Added"] = "BBB"
    parsed = history.parse_changes([table])
    with pytest.raises(ValueError, match="Event dates"):
        history.membership_intervals(parsed, pd.DataFrame({"ticker": ["AAA", "BBB"]}))


def test_one_sided_events_weekend_and_window_clipping():
    changes = pd.DataFrame({
        "date": ["2017-12-18", "2019-06-01", "2020-01-04"],
        "added": ["OLD", "AAA", None], "removed": [None, None, "BBB"],
    })
    result = history.membership_intervals(changes, pd.DataFrame({"ticker": ["AAA"]}))
    aaa = result.loc[result.ticker == "AAA"].iloc[0]
    bbb = result.loc[result.ticker == "BBB"].iloc[0]
    assert aaa.start == pd.Timestamp("2019-06-03")
    assert bbb.start == pd.Timestamp("2018-01-01")
    assert bbb.end == pd.Timestamp("2020-01-03")
    assert "OLD" not in set(result.ticker)


def test_single_session_count_problem_is_not_hidden_by_annual_average():
    membership = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(100)],
        "start": [pd.Timestamp("2018-01-01")] * 100, "end": [pd.NaT] * 100,
    })
    missing_day = pd.Timestamp("2021-07-06")
    membership.loc[0, "end"] = missing_day - pd.Timedelta(days=1)
    membership = pd.concat([membership, pd.DataFrame({
        "ticker": ["T0"], "start": [missing_day + pd.Timedelta(days=1)], "end": [pd.NaT],
    })], ignore_index=True)
    with pytest.raises(ValueError, match="2021-07-06"):
        history.validate_member_counts(membership)


def test_current_canonical_collision_rejected():
    with pytest.raises(ValueError, match="unique"):
        history.membership_intervals(
            pd.DataFrame(columns=["date", "added", "removed"]),
            pd.DataFrame({"ticker": ["FB", "META"]}),
        )


def test_offline_snapshot_cli_does_not_contact_wikipedia(monkeypatch, tmp_path):
    current = pd.DataFrame({"ticker": [f"T{i}" for i in range(101)]})
    changes = pd.DataFrame(columns=["date", "added", "removed"])
    current.to_csv(tmp_path / "current.csv", index=False)
    changes.to_csv(tmp_path / "changes.csv", index=False)
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(history, "scrape_current", lambda: pytest.fail("Unexpected network"))
    monkeypatch.setattr(history, "scrape_changes", lambda: pytest.fail("Unexpected network"))
    monkeypatch.setattr("sys.argv", ["build_index_history", "--changes-csv", str(tmp_path / "changes.csv"),
                                     "--current-csv", str(tmp_path / "current.csv")])
    history.main()
    assert len(read_table("membership")) == 101


def test_old_and_new_fox_are_distinct_issuers_with_equal_daily_counts():
    current = pd.DataFrame({"ticker": ["FOX", "FOXA"]})
    changes = pd.DataFrame(columns=["date", "added", "removed"])
    corrections = pd.DataFrame({
        "date": ["2019-03-19"] * 2, "added": ["FOX", "FOXA"], "removed": ["TFCF", "TFCFA"],
        "source_url": ["https://example.org/membership"] * 2,
        "identity_url": ["https://nasdaqtrader.com/TraderNews.aspx?id=ECA2019-26"] * 2,
        "reason": ["Distinct pre-spin and post-spin issuers"] * 2,
    })
    corrected = history.apply_reviewed_events(changes, current, corrections)
    result = history.membership_intervals(corrected, current).set_index("ticker")
    assert result.loc["TFCF", "end"] == pd.Timestamp("2019-03-18")
    assert result.loc["TFCFA", "end"] == pd.Timestamp("2019-03-18")
    assert result.loc["FOX", "start"] == pd.Timestamp("2019-03-19")
    assert result.loc["FOXA", "start"] == pd.Timestamp("2019-03-19")
    assert len(history.apply_reviewed_events(corrected, current, corrections)) == 2
    conflicts = corrected.copy()
    conflicts.loc[0, "removed"] = "FOX"
    with pytest.raises(ValueError, match="conflicts"):
        history.apply_reviewed_events(conflicts, current, corrections)
