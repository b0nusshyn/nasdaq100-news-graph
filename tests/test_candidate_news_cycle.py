import numpy as np
import pandas as pd
import pytest

from experiments.fnspid_candidate_audit.analyze_cycle import block_interval, paired_days


def test_paired_daily_auc_rejects_mismatched_assessment_rows():
    frame = pd.DataFrame({"date": pd.to_datetime(["2023-08-01"] * 2), "ticker": ["A", "B"],
                          "y_true": [1, 0], "p_up": [.6, .4]})
    with pytest.raises(AssertionError):
        paired_days(frame, frame.iloc[::-1])
    improved = paired_days(frame, frame.assign(p_up=[.1, .9]))
    assert improved == {"2023-08-01": 1.0}
    assert paired_days(frame, frame, frame.ticker.eq("A")) == {}


def test_block_interval_preserves_equal_window_weight_and_constant_contrasts():
    assert block_interval([[.02] * 12, [0.] * 20], np.random.default_rng(4)) == pytest.approx([.01, .01])


def test_checkpoint_writer_respects_free_disk_before_creating_a_file(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from experiments.fnspid_candidate_audit import encode_cycle

    limit = encode_cycle.MIN_FREE
    monkeypatch.setattr(encode_cycle.shutil, "disk_usage", lambda _: type("Disk", (), {"free": limit - 1})())
    target = tmp_path / "chunk.parquet"
    with pytest.raises(RuntimeError, match="safety margin"):
        encode_cycle.write_table(pd.DataFrame({"article_id": ["a"]}), target)
    assert not list(tmp_path.iterdir())
