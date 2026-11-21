import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.prospective_cohort_v1 import live


def test_gate_blocks_unapproved_source():
    cfg, _ = live.load_freeze()
    with pytest.raises(ValueError, match="readiness gate"):
        live.source_gate(live.ROOT / "source_status.json", cfg)


def test_early_close_and_no_backdating():
    freeze = "2025-07-01T20:00:00Z"
    close, deadline, target = live.timing("2025-07-03", freeze, pd.Timestamp("2025-07-03T18:00:00Z"))
    assert close == pd.Timestamp("2025-07-03T17:00:00Z")
    assert target == "2025-07-07"
    with pytest.raises(ValueError, match="next-session open"):
        live.timing("2025-07-03", freeze, deadline)
    with pytest.raises(ValueError, match="next-session open"):
        live.timing("2025-07-03", close.isoformat(), close)


def test_write_once_keeps_original_bytes(tmp_path):
    p = tmp_path / "record"
    assert live.write_once(p, b"first") == live.digest(b"first")
    with pytest.raises(FileExistsError):
        live.write_once(p, b"second")
    assert p.read_bytes() == b"first"


def test_fixture_issue_settle_and_evaluate(monkeypatch, tmp_path):
    # Synthetic fixture in pytest temp storage. It is never part of the live log.
    cfg, h = live.load_freeze()
    cfg["frozen_at"] = "2025-06-01T00:00:00Z"
    monkeypatch.setattr(live, "load_freeze", lambda: (cfg, h))
    attrs = {"source": "synthetic-test-fixture", "adjustment_policy": "fixture-only",
             "security_ids": {ticker: f"id-{ticker}" for ticker in cfg["cohort"]}}
    gate_file = tmp_path / "fixture_gate.json"
    gate_file.write_text(json.dumps(attrs))
    signal = "2025-07-03"
    schedule, pos = live.calendar(signal)
    times = schedule.index[pos+1-cfg["warmup_sessions"]:pos+1]
    bars = tmp_path / "bars.parquet"
    manifest = tmp_path / "bars.json"
    args = SimpleNamespace(day=signal, bars=bars, meta=manifest, log=tmp_path / "log", gate=gate_file)

    def set_snapshot(dates, at):
        index = pd.MultiIndex.from_product([dates, cfg["cohort"]], names=["date", "ticker"])
        values = np.arange(len(index))
        p = index.to_frame(index=False)
        price = 100 + values*.005 + np.sin(values*.013)
        for col in ["open", "close", "adj_close"]:
            p[col] = price
        p["high"], p["low"], p["volume"] = price+1, price-1, 10000+values
        p.to_parquet(bars, index=False)
        manifest.write_text(json.dumps(dict(attrs, revision=at, published_at=at, retrieved_at=at,
                                            sha256=live.digest(bars.read_bytes()))))
        monkeypatch.setattr(live, "now", lambda: pd.Timestamp(at))

    set_snapshot(times, "2025-07-03T18:00:00Z")
    live.issue(args, cfg, h, attrs)
    with pytest.raises(FileExistsError):
        live.issue(args, cfg, h, attrs)
    assert live.evaluate(args.log)["settled_sessions"] == 0
    with pytest.raises(ValueError, match="has not closed"):
        live.settle(args, cfg, h, attrs)
    set_snapshot(pd.to_datetime([signal, "2025-07-07"]), "2025-07-07T21:00:00Z")
    live.settle(args, cfg, h, attrs)
    with pytest.raises(FileExistsError):
        live.settle(args, cfg, h, attrs)
    result = live.evaluate(args.log)
    assert result["settled_sessions"] == 1
    assert result["metrics"]["selected_1"]["rows"] == 30
    assert result["metrics"]["selected_0.2"]["rows"] == 6
