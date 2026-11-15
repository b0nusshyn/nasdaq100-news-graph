"""Survivorship and horizon leakage regression guards (Data, issue #26)."""

import numpy as np
import pandas as pd

from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.data.build_labels import make_labels
from src.data.fetch_prices import add_in_index
from src.data.indicators import FEATURES, compute_indicators


def test_every_membership_flag_matches_an_actual_spell():
    membership = read_table("membership", FIXTURES_DIR / "membership.parquet")
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    out = add_in_index(prices.drop(columns="in_index"), membership)
    expected = [any(row.ticker == spell.ticker and row.date >= spell.start and
                    (pd.isna(spell.end) or row.date <= spell.end)
                    for spell in membership.itertuples(index=False))
                for row in prices.itertuples(index=False)]
    assert out.in_index.tolist() == expected
    assert out.loc[~out.in_index].shape[0] > 0  # warm-up retained, excluded from model sample


def test_features_unchanged_when_future_is_modified():
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    cutoff = prices.date.sort_values().unique()[35]
    original = compute_indicators(prices)
    changed = prices.copy()
    changed.loc[changed.date.gt(cutoff), ["open", "high", "low", "close", "volume"]] *= 10
    altered = compute_indicators(changed)
    pd.testing.assert_frame_equal(original.loc[prices.date.le(cutoff)], altered.loc[prices.date.le(cutoff)])
    prefix = compute_indicators(prices.loc[prices.date.le(cutoff)])
    pd.testing.assert_frame_equal(prefix, original.loc[prices.date.le(cutoff)].reset_index(drop=True))
    assert list(prefix.columns)[2:] == list(FEATURES)


def test_no_label_uses_prices_beyond_its_calendar_horizon():
    prices = read_table("prices", FIXTURES_DIR / "prices.parquet")
    original = make_labels(prices)
    # Jan 4 + 5 NASDAQ sessions is Jan 11; Jan 12 and later cannot affect that label.
    changed = prices.copy()
    changed.loc[changed.date.gt("2021-01-11"), ["open", "high", "low", "close"]] *= 3
    altered = make_labels(changed)
    source = original.date.eq("2021-01-04")
    pd.testing.assert_frame_equal(original.loc[source], altered.loc[source])
    # Changing the actual target DOES change the forward return.
    changed = prices.copy()
    changed.loc[changed.date.eq("2021-01-11"), ["open", "high", "low", "close"]] *= 2
    targeted = make_labels(changed)
    assert not np.allclose(original.loc[source, "ret_5d"], targeted.loc[source, "ret_5d"])
