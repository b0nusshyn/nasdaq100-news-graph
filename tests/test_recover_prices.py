from hashlib import sha256

import pandas as pd
import pytest

from src.data.recover_prices import archived_prices, load_reviewed_sources, merge_missing_histories


def archive():
    return pd.DataFrame({
        "date": ["2021-01-05", "2021-01-04"], "open": [20., 10.], "high": [22., 12.],
        "low": [18., 9.], "close": [20., 10.], "adj close": [10., 5.], "volume": [200., 100.],
    })


def test_dividend_adjustment_applies_to_every_ohlc_and_not_volume():
    out = archived_prices(archive(), "FB", "2021-01-01", "2021-01-06")
    assert list(out.ticker) == ["META", "META"]
    assert out.open.tolist() == [5., 10.]
    assert out.high.tolist() == [6., 11.]
    assert out.low.tolist() == [4.5, 9.]
    assert out.close.tolist() == [5., 10.]
    assert out.volume.tolist() == [100., 200.]
    assert str(out.date.dtype) == "datetime64[ns]"


def test_zero_adjustment_or_missing_adjusted_close_rejected():
    bad = archive()
    bad.loc[0, "adj close"] = 0
    with pytest.raises(ValueError, match="adjustment"):
        archived_prices(bad, "AAA", "2021-01-01", "2021-01-06")
    with pytest.raises(ValueError, match="Archive needs"):
        archived_prices(archive().drop(columns="adj close"), "AAA", "2021-01-01", "2021-01-06")


def manifest(tmp_path):
    source = tmp_path / "AAA.csv"
    archive().to_csv(source, index=False)
    row = {"ticker": "AAA", "file": "AAA.csv", "sha256": sha256(source.read_bytes()).hexdigest(),
           "source_url": "https://example.org/archive/AAA.csv", "identity_url": "https://example.org/issuer/AAA",
           "identity_review": "reviewed"}
    path = tmp_path / "reviewed.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path, row


def test_manifest_pins_bytes_and_requires_identity_review(tmp_path):
    path, row = manifest(tmp_path)
    out, provenance = load_reviewed_sources(path, "2021-01-01", "2021-01-06")
    assert len(out) == provenance[0]["rows"] == 2
    row["identity_review"] = "pending"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Identity review"):
        load_reviewed_sources(path, "2021-01-01", "2021-01-06")
    row["identity_review"] = "reviewed"
    row["sha256"] = "0" * 64
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Checksum"):
        load_reviewed_sources(path, "2021-01-01", "2021-01-06")


def test_no_splicing_adjustment_vintages_and_preserve_existing():
    from src.data.fetch_prices import add_in_index

    members = pd.DataFrame({"ticker": ["AAA", "BBB"], "start": [pd.Timestamp("2021-01-01")] * 2,
                            "end": [pd.NaT] * 2})
    a = archived_prices(archive(), "AAA", "2021-01-01", "2021-01-06")
    existing = add_in_index(a, members)
    with pytest.raises(ValueError, match="splice"):
        merge_missing_histories(existing, a, members)
    b = archived_prices(archive(), "BBB", "2021-01-01", "2021-01-06")
    merged = merge_missing_histories(existing, b, members)
    pd.testing.assert_frame_equal(merged.loc[merged.ticker.eq("AAA")].reset_index(drop=True), existing)
    assert len(merged) == 4


def test_recovery_recomputes_flags_after_issuer_membership_correction():
    from src.data.fetch_prices import add_in_index

    raw = archived_prices(archive(), "AAA", "2021-01-01", "2021-01-06")
    before = pd.DataFrame({"ticker": ["AAA", "BBB"], "start": pd.to_datetime(["2020-01-01"] * 2),
                           "end": [pd.NaT] * 2})
    existing = add_in_index(raw, before)
    corrected = before.copy()
    corrected.loc[0, "start"] = pd.Timestamp("2021-01-05")
    supplemental = archived_prices(archive(), "BBB", "2021-01-01", "2021-01-06")
    merged = merge_missing_histories(existing, supplemental, corrected)
    assert merged.loc[merged.ticker.eq("AAA"), "in_index"].tolist() == [False, True]
    pd.testing.assert_frame_equal(merged.loc[merged.ticker.eq("AAA")].drop(columns="in_index").reset_index(drop=True),
                                  existing.drop(columns="in_index"))
