import pandas as pd
import pytest

from src.data.intraday_daily import regular_session_daily


def bars_for(opening="2021-01-04 14:30Z", closing="2021-01-04 21:00Z"):
    dates = pd.date_range(opening, closing, freq="30min", inclusive="left")
    return pd.DataFrame({"ts": dates, "open": 100., "high": 102., "low": 99., "close": 101., "volume": 100.})


def test_regular_session_excludes_after_close_and_pre_market():
    bars = bars_for()
    outside = pd.DataFrame({"ts": pd.to_datetime(["2021-01-04 14:00Z", "2021-01-04 21:00Z"]),
                            "open": 900., "high": 999., "low": 800., "close": 950., "volume": 10000.})
    daily, review = regular_session_daily(pd.concat([bars, outside]), "FB", bar_minutes=30,
                                          start="2021-01-01", end="2021-01-05")
    assert daily.ticker.iloc[0] == "META"
    assert daily.close.iloc[0] == 101 and daily.high.iloc[0] == 102
    assert daily.volume.iloc[0] == 1300
    assert review.complete.all()


def test_missing_bar_not_filled_and_early_close_respected():
    bars = bars_for("2021-11-26 14:30Z", "2021-11-26 18:00Z")
    daily, review = regular_session_daily(bars, "AAA", bar_minutes=30,
                                          start="2021-11-26", end="2021-11-27")
    assert len(daily) == 1 and review.expected_bars.iloc[0] == 7
    daily, review = regular_session_daily(bars.drop(index=2), "AAA", bar_minutes=30,
                                          start="2021-11-26", end="2021-11-27")
    assert daily.empty and not review.complete.any()


def test_timezone_and_duplicates_rejected():
    bars = bars_for()
    with pytest.raises(ValueError, match="Duplicate"):
        regular_session_daily(pd.concat([bars, bars.head(1)]), "AAA", bar_minutes=30,
                              start="2021-01-01", end="2021-01-05")
    bars["ts"] = bars.ts.dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        regular_session_daily(bars, "AAA", bar_minutes=30, start="2021-01-01", end="2021-01-05")
