import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.core.io import read_table, write_table
from src.core.paths import FIXTURES_DIR
from src.data import fetch_prices as prices


def provider_frame():
    return pd.DataFrame({
        "Open": [10.0, 11.0], "High": [12.0, 13.0], "Low": [9.0, 10.0],
        "Close": [11.0, 12.0], "Volume": [100, 200],
    }, index=pd.DatetimeIndex(["2021-01-04", "2021-01-05"], tz="America/New_York"))


@pytest.mark.parametrize("layout", ["flat", "field_first", "ticker_first"])
def test_normalize_multiindex_timezone_and_types(layout):
    frame = provider_frame()
    if layout != "flat":
        columns = [(c, "FB") if layout == "field_first" else ("FB", c) for c in frame.columns]
        frame.columns = pd.MultiIndex.from_tuples(columns)
    out = prices.normalize_prices(frame, " fb ", "2021-01-01", "2021-01-05")
    assert list(out.columns) == prices.RAW_COLUMNS
    assert len(out) == 1  # exclusive end
    assert out.ticker.iloc[0] == "META"
    assert out.date.iloc[0] == pd.Timestamp("2021-01-04")
    assert str(out.date.dtype) == "datetime64[ns]"
    assert all(str(out[c].dtype) == "float64" for c in prices.PRICE_COLUMNS)


@pytest.mark.parametrize("problem", ["missing", "nan", "duplicate", "weekend", "wrong_ticker", "negative"])
def test_bad_downloads_rejected(problem):
    frame = provider_frame()
    if problem == "missing":
        frame = frame.drop(columns="Close")
    elif problem == "nan":
        frame.iloc[0, 0] = float("nan")
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.head(1)])
    elif problem == "weekend":
        frame.index = pd.DatetimeIndex(["2021-01-02", "2021-01-03"])
    elif problem == "wrong_ticker":
        frame.columns = pd.MultiIndex.from_product([frame.columns, ["OTHER"]])
    else:
        frame.loc[:, "Volume"] = -1
    with pytest.raises((ValueError, KeyError)):
        prices.normalize_prices(frame, "AAA", "2021-01-01", "2021-01-10")


def test_download_isolates_failures_and_canonicalizes(monkeypatch):
    provider = Mock(side_effect=[pd.DataFrame(), provider_frame(), RuntimeError("provider down")])
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=provider))
    result = prices.download(["META", "fb", "EMPTY", "ZZZ"], "2021-01-01", "2021-01-06")
    assert list(result.ticker.unique()) == ["META"]
    assert [r["ticker"] for r in result.attrs["failed_tickers"]] == ["EMPTY", "ZZZ"]
    assert provider.call_count == 3
    for call in provider.call_args_list:
        assert call.kwargs["auto_adjust"] is True
        assert call.kwargs["threads"] is False
        assert call.kwargs["end"] == "2021-01-06"


def test_main_uses_fixture_membership_and_raw_paths(monkeypatch, tmp_path, capsys):
    membership = read_table("membership", FIXTURES_DIR / "membership.parquet")
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    write_table(membership, "membership")
    provider = Mock(return_value=provider_frame())
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=provider))
    monkeypatch.setattr(sys, "argv", ["fetch_prices", "--start", "2021-01-01", "--end", "2021-01-06"])
    prices.main()
    assert provider.call_count == membership.ticker.nunique()
    raw = pd.read_csv(tmp_path / "raw/prices_raw.csv")
    assert len(raw) == 2 * membership.ticker.nunique()
    assert list(raw.columns) == prices.RAW_COLUMNS
    assert pd.read_csv(tmp_path / "raw/failed_tickers.csv").empty
    final = read_table("prices")
    assert len(final) == len(raw)
    assert not final.loc[final.ticker.eq("FFF"), "in_index"].any()
    assert "failed 0" in capsys.readouterr().out


def test_all_failed_reports_before_raising(monkeypatch, tmp_path):
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    write_table(pd.DataFrame({"ticker": ["AAA"], "start": [pd.Timestamp("2018-01-01")], "end": [pd.NaT]}),
                "membership")
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=Mock(return_value=pd.DataFrame())))
    monkeypatch.setattr(sys, "argv", ["fetch_prices"])
    with pytest.raises(RuntimeError, match="All tickers failed"):
        prices.main()
    assert list(pd.read_csv(tmp_path / "raw/failed_tickers.csv").ticker) == ["AAA"]
    assert pd.read_csv(tmp_path / "raw/prices_raw.csv").empty


@pytest.mark.parametrize("start,end", [("2023-01-01", "2023-01-01"), ("2024-01-01", "2023-01-01"),
                                      ("2023-01-01T12:00", "2024-01-01")])
def test_invalid_range_before_network(start, end):
    with pytest.raises(ValueError):
        prices.download(["AAA"], start, end)


def test_adjustment_roundoff_tolerated_without_repairing_prices():
    frame = provider_frame()
    frame.loc[frame.index[0], "High"] = frame.Close.iloc[0] - 1e-14
    out = prices.normalize_prices(frame, "AAA", "2021-01-01", "2021-01-06")
    assert out.high.iloc[0] == frame.High.iloc[0]
    frame.loc[frame.index[0], "High"] = frame.Close.iloc[0] - 0.001
    with pytest.raises(ValueError, match="OHLC range"):
        prices.normalize_prices(frame, "AAA", "2021-01-01", "2021-01-06")


def test_retry_only_failed_retains_successful_histories(monkeypatch, tmp_path):
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    membership = pd.DataFrame({"ticker": ["AAA", "BBB"],
                               "start": [pd.Timestamp("2018-01-01")] * 2, "end": [pd.NaT] * 2})
    write_table(membership, "membership")
    raw = prices.normalize_prices(provider_frame(), "AAA", "2021-01-01", "2021-01-06")
    folder = tmp_path / "raw"
    folder.mkdir()
    raw.to_csv(folder / "prices_raw.csv", index=False)
    pd.DataFrame({"ticker": ["BBB"], "reason": ["temporary failure"]}).to_csv(
        folder / "failed_tickers.csv", index=False
    )
    provider = Mock(return_value=provider_frame())
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=provider))
    monkeypatch.setattr(sys, "argv", ["fetch_prices", "--retry-failed", "--start", "2021-01-01",
                                      "--end", "2021-01-06"])
    prices.main()
    assert provider.call_count == 1 and provider.call_args.args == ("BBB",)
    assert set(read_table("prices").ticker) == {"AAA", "BBB"}
    assert pd.read_csv(folder / "failed_tickers.csv").empty
