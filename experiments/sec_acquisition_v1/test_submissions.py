import json

import pytest

from experiments.sec_acquisition_v1.submissions import identifying_agent, read_candidates

CIK = "0000077476"


@pytest.mark.parametrize("value", ["", "Project", "Project someone@example.com", "Placeholder person@acme.org"])
def test_reject_placeholder(value):
    with pytest.raises(ValueError):
        identifying_agent(value)


def test_cached_history_only(tmp_path):
    header = f"CIK{CIK}"
    recent = dict(accessionNumber=[], form=[], filingDate=[], reportDate=[], acceptanceDateTime=[])
    name = header + "-submissions-001.json"
    root = {"cik": "77476", "filings": {"recent": recent, "files": [
        {"name": name, "filingFrom": "2019-02-01", "filingTo": "2022-01-01"},
        {"name": "never-read", "filingFrom": "2012-01-01", "filingTo": "2017-12-01"}]}}
    (tmp_path / (header + ".json")).write_text(json.dumps(root))
    older = dict(accessionNumber=["0000077476-21-000050"], form=["10-Q"], filingDate=["2021-10-05"],
                 reportDate=["2021-09-04"], acceptanceDateTime=["2021-10-05T10:00:00Z"])
    (tmp_path / name).write_text(json.dumps(older))
    result = list(read_candidates(tmp_path, CIK))
    assert len(result) == 1
    assert result[0]["source"].endswith(name)
    assert result[0]["status"] == "requires_original_accession_review"
    assert len(result[0]["sha256"]) == 64


def test_offline_cache_miss(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_candidates(tmp_path, CIK))
