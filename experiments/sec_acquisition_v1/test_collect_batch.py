import sys
from urllib.error import HTTPError

import pytest

from experiments.sec_acquisition_v1 import collect_batch


@pytest.mark.parametrize("status", [403, 429])
def test_batch_deduplicates_and_stops(monkeypatch, tmp_path, status):
    ledger = tmp_path / "identities.csv"
    ledger.write_text("cik,identity_basis\n" +
                      "".join(f"{cik},https://www.sec.gov/Archives/test\n"
                              for cik in ("0000000001", "0000000001", "0000000002", "0000000003")))
    attempted = []

    def read(cache, cik, agent):
        assert agent is None
        attempted.append(cik)
        if cik == "0000000002":
            raise HTTPError("https://data.sec.gov", status, "test", {}, None)
        yield {}

    monkeypatch.setattr(collect_batch, "read_candidates", read)
    monkeypatch.setattr(sys, "argv", ["collect_batch", str(tmp_path), str(ledger)])
    with pytest.raises(SystemExit, match=f"HTTP {status}"):
        collect_batch.main()
    assert attempted == ["0000000001", "0000000002"]
