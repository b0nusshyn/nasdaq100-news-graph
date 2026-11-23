import io

import pandas as pd
import pytest

from experiments.prospective_cohort_v1.live import digest
from experiments.prospective_feed_v1.archive import FIELDS, convert, summarize


def test_adapter_verifies_hash_and_preserves_dates_and_bars():
    rows = pd.DataFrame({"date": ["2026-11-19"], **dict.fromkeys(FIELDS, [10])})
    payload = rows.to_parquet(index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        convert(payload, "AAPL", digest(b"modified"))
    actual = convert(payload, "AAPL", digest(payload))
    assert actual.loc[0, "ticker"] == "AAPL"
    assert actual.loc[0, FIELDS].tolist() == [10] * len(FIELDS)
    stats = summarize(actual, pd.to_datetime(["2026-11-19", "2026-11-20"]))
    assert stats == {"latest": "2026-11-19", "valid": 1, "missing": ["2026-11-20"], "present_invalid": []}


def test_duplicate_and_invalid_volume_are_distinct_from_missing():
    rows = pd.DataFrame({"date": ["2026-11-19", "2026-11-19"], **dict.fromkeys(FIELDS, [10, 10])})
    buffer = io.BytesIO()
    rows.to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    with pytest.raises(ValueError, match="duplicate"):
        convert(payload, "AAPL", digest(payload))
    rows = rows.iloc[:1].assign(date=pd.to_datetime(["2026-11-19"]), volume=0)
    stats = summarize(rows, pd.to_datetime(["2026-11-19"]))
    assert stats["valid"] == 0 and stats["present_invalid"] == ["2026-11-19"] and not stats["missing"]
