import sys

import numpy as np
import pandas as pd
import pytest

from src.core.io import read_table, write_table
from src.core.schema import conform, validate
from src.data import build_labels, fetch_prices, quality
from src.data.build_index_history import trading_days
from src.data.indicators import FEATURES, compute_indicators


@pytest.fixture
def sample_prices():
    dates = trading_days("2021-01-04", "2021-04-30")[:60]
    close = np.arange(100, 100 + len(dates), dtype=float)
    frames = []
    for ticker in ("AAA", "BBB"):
        frames.append(pd.DataFrame({
            "date": dates, "ticker": ticker, "open": close, "high": close + 1,
            "low": close - 1, "close": close, "volume": np.arange(len(dates), dtype=float) + 100,
            "in_index": True,
        }))
    return conform(pd.concat(frames, ignore_index=True), "prices")


@pytest.fixture
def spells(sample_prices):
    dates = sample_prices.date.unique()
    return pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "start": [dates[2], dates[8], dates[0]],
        "end": [dates[4], pd.NaT, pd.NaT],
    })


def test_inclusive_spells_rejoin_and_warmup_preserved(sample_prices, spells):
    raw = sample_prices.drop(columns="in_index").sample(frac=1, random_state=1).reset_index(drop=True)
    result = fetch_prices.add_in_index(raw, spells)
    pd.testing.assert_frame_equal(result.drop(columns="in_index"), raw)
    aaa = result.loc[result.ticker.eq("AAA")].sort_values("date")
    assert aaa.in_index.iloc[:10].tolist() == [False, False, True, True, True, False, False, False, True, True]
    validate(result, "prices")


@pytest.mark.parametrize("bad", ["overlap", "unknown", "reverse", "canonical_collision"])
def test_membership_errors(sample_prices, spells, bad):
    if bad == "overlap":
        spells.loc[0, "end"] = pd.NaT
    elif bad == "unknown":
        sample_prices.loc[0, "ticker"] = "ZZZ"
    elif bad == "reverse":
        spells.loc[0, "end"] = pd.Timestamp("2000-01-01")
    else:
        spells.loc[:, "ticker"] = ["FB", "META", "BBB"]
        spells.loc[1, "start"] = spells.loc[0, "start"]
    with pytest.raises(ValueError):
        fetch_prices.add_in_index(sample_prices, spells)


def test_labels_formula_order_ties_and_trailing_nulls(sample_prices):
    sample_prices.loc[:5, ["open", "close"]] = 100.0
    sample_prices.loc[:5, "low"] = 99.0
    shuffled = sample_prices.sample(frac=1, random_state=3).reset_index(drop=True)
    labels = build_labels.make_labels(shuffled)
    pd.testing.assert_frame_equal(labels[["date", "ticker"]], shuffled[["date", "ticker"]])
    sorted_labels = labels.sort_values(["ticker", "date"])
    sorted_prices = shuffled.sort_values(["ticker", "date"])
    for h in (1, 5):
        expected = sorted_prices.groupby("ticker").close.shift(-h) / sorted_prices.close - 1
        np.testing.assert_allclose(sorted_labels[f"fwd_ret_{h}d"], expected, equal_nan=True)
        assert str(labels[f"y_{h}d"].dtype) == "Int8"
        for _, group in sorted_labels.groupby("ticker"):
            assert group[f"y_{h}d"].tail(h).isna().all()
    assert sorted_labels.y_1d.iloc[0] == 0
    assert sorted_labels.y_5d.iloc[0] == 0


def test_label_missing_target_does_not_extend_horizon(sample_prices):
    frame = sample_prices.iloc[:10].drop(index=1).reset_index(drop=True)
    labels = build_labels.make_labels(frame)
    assert pd.isna(labels.fwd_ret_1d.iloc[0])
    assert labels.fwd_ret_5d.iloc[0] == pytest.approx(105 / 100 - 1)


def test_horizon_subset_keeps_frozen_contract(sample_prices):
    result = build_labels.make_labels(sample_prices, (1,))
    validate(result, "labels")
    assert result.fwd_ret_5d.isna().all() and result.y_5d.isna().all()
    for horizons in ((), (2,), (1, 1)):
        with pytest.raises(ValueError):
            build_labels.make_labels(sample_prices, horizons)


def test_offline_finalize_and_label_clis(sample_prices, spells, monkeypatch, tmp_path):
    monkeypatch.setenv("NDX_DATA_DIR", str(tmp_path))
    write_table(spells, "membership")
    (tmp_path / "raw").mkdir()
    sample_prices.drop(columns="in_index").to_csv(tmp_path / "raw/prices_raw.csv", index=False)
    monkeypatch.setattr(fetch_prices, "download", lambda *args: pytest.fail("Offline run contacted network"))
    monkeypatch.setattr(sys, "argv", ["fetch_prices", "--finalize-only"])
    fetch_prices.main()
    monkeypatch.setattr(sys, "argv", ["build_labels", "--horizons", "1", "5"])
    build_labels.main()
    assert len(read_table("labels")) == len(sample_prices)
    assert not read_table("prices").in_index.all()
    monkeypatch.setattr(sys, "argv", ["quality", "--start", "2021-01-04", "--end", "2021-03-30"])
    quality.main()
    assert (tmp_path / "raw/quality_report.csv").exists()


def test_indicators_formulas_warmup_and_ticker_isolation(sample_prices):
    frame = sample_prices.sample(frac=1, random_state=4).reset_index(drop=True)
    result = compute_indicators(frame)
    pd.testing.assert_frame_equal(result[["date", "ticker"]], frame[["date", "ticker"]])
    group = result.loc[result.ticker.eq("AAA")].sort_values("date")
    assert group.rsi_14.iloc[:14].isna().all()
    assert group.rsi_14.iloc[14:].eq(100).all()
    assert group.volatility_20d.iloc[:20].isna().all()
    assert group.macd.iloc[:25].isna().all()
    assert group.macd_signal.iloc[:33].isna().all()
    assert group.volume_z_20d.iloc[19] == pytest.approx((119 - np.mean(np.arange(100, 120))) /
                                                      np.std(np.arange(100, 120), ddof=1))
    for ticker in ("AAA", "BBB"):
        alone = compute_indicators(frame.loc[frame.ticker.eq(ticker)])
        pd.testing.assert_frame_equal(alone.reset_index(drop=True),
                                      result.loc[result.ticker.eq(ticker)].reset_index(drop=True))
    assert list(result.columns) == ["date", "ticker", *FEATURES]


def test_flat_indicators_are_finite_after_warmup(sample_prices):
    sample_prices.loc[:, ["open", "close", "high", "low"]] = 100.0
    sample_prices.loc[:, "volume"] = 100.0
    result = compute_indicators(sample_prices)
    for _, group in result.groupby("ticker"):
        assert group.rsi_14.iloc[14:].eq(50).all()
        assert group.volume_z_20d.iloc[19:].eq(0).all()
        assert not np.isinf(group[list(FEATURES)].to_numpy()).any()


def test_quality_detects_missing_entire_ticker_jump_and_balance(sample_prices, spells):
    frame = fetch_prices.add_in_index(sample_prices, spells)
    frame = frame.loc[frame.ticker.eq("AAA")].drop(index=3).reset_index(drop=True)
    frame.loc[10:, ["open", "high", "low", "close"]] *= 2
    labels = build_labels.make_labels(frame)
    report = quality.quality_report(frame, labels, spells)
    assert {"missing_session", "missing_ticker", "missing_in_index", "adjustment_review", "class_balance"} <= set(
        report["check"]
    )
    assert not report["check"].str.startswith("label_").any()
    balance = report.loc[report["check"].eq("class_balance")]
    assert balance.value.eq(1).all()
    with pytest.raises(ValueError, match="same keys"):
        quality.quality_report(frame, labels.iloc[1:])
