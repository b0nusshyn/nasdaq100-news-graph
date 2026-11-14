import pytest

from src.core.tickers import RENAMES, canonical_ticker


def test_renamed_company_maps_to_current_symbol():
    assert canonical_ticker("FB") == "META"
    assert canonical_ticker(" fb ") == "META"


def test_current_symbols_unchanged():
    assert canonical_ticker("META") == "META"
    assert canonical_ticker("GOOG") == "GOOG"
    assert canonical_ticker("GOOGL") == "GOOGL"


def test_rename_chains_resolve(monkeypatch):
    monkeypatch.setitem(RENAMES, "AAA", "BBB")
    monkeypatch.setitem(RENAMES, "BBB", "CCC")
    assert canonical_ticker("AAA") == "CCC"


def test_rename_cycle_rejected(monkeypatch):
    monkeypatch.setitem(RENAMES, "AAA", "BBB")
    monkeypatch.setitem(RENAMES, "BBB", "AAA")
    with pytest.raises(ValueError, match="cycle"):
        canonical_ticker("AAA")


def test_check_env():
    pytest.importorskip("torch_geometric", reason="full environment only; CI uses requirements-dev.txt")
    from src.core.check_env import check

    assert check()["gat_forward"] == "ok"
