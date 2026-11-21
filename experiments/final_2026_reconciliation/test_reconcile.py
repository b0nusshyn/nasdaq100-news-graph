from pathlib import Path

import pandas as pd

from experiments.final_2026_reconciliation.reconcile import endpoint_status, parse_page

ROOT = Path(__file__).parent


def test_pre_exit_endpoint_is_distinct_from_crossing_exit():
    first, last, after = pd.to_datetime(["2026-10-02", "2026-10-05", "2026-10-06"])
    observed = {first, last, after}
    assert endpoint_status(first, last, last, after, observed, observed) == "archive_endpoints_present_provisional"
    assert endpoint_status(last, after, last, after, observed, observed) == "crosses_exit"
    assert endpoint_status(after, after, last, after, observed, observed) == "after_last_trade"


def test_coverage_cannot_qualify_alternate_price_or_snapshot_tail():
    first, last, after = pd.to_datetime(["2026-11-18", "2026-11-19", "2026-11-20"])
    assert endpoint_status(first, last, after, last, {first, last}, {first}) == (
        "alternate_source_adjustment_unqualified"
    )
    assert endpoint_status(first, last, after, last, {first}, {first}) == "missing_observed_endpoint"
    assert endpoint_status(last, after, after, last, {last}, {last}) == "target_after_snapshot"


def test_actual_source_distinguishes_post_exit_rows():
    a = parse_page(ROOT / "raw/chartexchange_ANSS.html")
    b = parse_page(ROOT / "raw/stockanalysis_ANSS.html")
    assert a.date.max() == pd.Timestamp("2025-07-16")
    assert b.date.max() == pd.Timestamp("2025-07-17")
    assert b.loc[b.date.eq(b.date.max()), "volume"].isna().all()
