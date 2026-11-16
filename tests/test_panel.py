"""Task 3.6 (#29): the model input every model shares (`src.models.panel`).

One row set, and features normalised with statistics from the training fold only. Runs on the
committed fixtures: 6 tickers over 60 trading days, FFF joining the index at day 30.
"""

import dataclasses

import numpy as np
import pytest

from src.core.io import read_table
from src.core.paths import FIXTURES_DIR
from src.models.panel import FULL_WINDOW, build_panel, fit_features

W = 5
DIM = 8  # the fixtures have too few news rows for 64 components


def tables():
    return {n: read_table(n, FIXTURES_DIR / f"{n}.parquet") for n in ("prices", "labels", "news_features")}


@pytest.fixture(scope="module")
def panel():
    return build_panel(**tables())


@pytest.fixture(scope="module")
def train(panel):
    return panel.days[:40]


def fit(panel, train, **kw):
    return fit_features(panel, train, window=W, **{"embedding_dim": DIM, **kw})


def col(f, name):
    return f.x[:, :, f.names.index(name)]


# --------------------------------------------------------------------------- row set


def test_row_set_is_every_in_index_row_with_a_label(panel):
    t = tables()
    expected = t["labels"].merge(t["prices"][["date", "ticker", "in_index"]], on=["date", "ticker"])
    expected = expected[expected["in_index"] & expected["y_1d"].notna()]
    rows = panel.rows(panel.days, 1)
    assert len(rows) == len(expected)
    assert set(zip(rows["date"], rows["ticker"], strict=True)) == set(
        zip(expected["date"], expected["ticker"], strict=True))


def test_row_set_respects_membership_and_the_label_horizon(panel):
    rows = panel.rows(panel.days, 5)
    fff = rows.loc[rows["ticker"] == "FFF", "date"]
    assert fff.min() == panel.days[30]  # FFF joins at day 30: survivorship
    assert rows["date"].max() == panel.days[-6]  # the last 5 days have no 5-day label


def test_row_set_keeps_rows_with_an_incomplete_history(panel, train):
    rows = panel.rows(panel.days[:W], 1)
    assert len(rows) > 0  # first days: no full window, but still predicted
    f = fit(panel, train)
    assert (col(f, FULL_WINDOW)[: W - 1] == 0).all()


# --------------------------------------------------------------------------- price features


def test_price_statistics_come_from_in_index_training_rows_only(panel, train):
    f = fit(panel, train)
    on = np.isin(panel.days, train)[:, None] & panel.in_index
    assert np.allclose(f.stats["price"]["mean"], np.nanmean(panel.price[on], axis=0))
    later = panel.price.copy()
    later[40:] *= 1000  # wild prices after training change nothing fitted on it
    g = fit(dataclasses.replace(panel, price=later), train)
    assert np.allclose(g.stats["price"]["mean"], f.stats["price"]["mean"])
    assert np.array_equal(g.x[:40], f.x[:40])


def test_a_missing_indicator_is_zero_and_flags_the_window(panel, train):
    f = fit(panel, train)
    k = list(panel.tickers).index("AAA")
    observed = ~np.isnan(panel.price[:, k]).any(axis=1)
    first = int(np.argmax(observed))  # MACD signal warm-up ends here
    assert first > 0
    assert (col(f, "macd_signal")[:first, k] == 0).all()
    assert (col(f, FULL_WINDOW)[: first + W - 1, k] == 0).all()
    assert col(f, FULL_WINDOW)[first + W - 1, k] == 1  # every day of the window observed
    assert np.isfinite(f.x).all()


# --------------------------------------------------------------------------- news features


def test_rows_without_news_stay_exactly_zero_after_scaling(panel, train):
    f = fit(panel, train)
    no_news = col(f, "has_news") == 0
    news_cols = [i for i, n in enumerate(f.names) if n in ("sentiment", "confidence") or n.startswith("emb_pca")]
    assert no_news.any()
    assert (f.x[no_news][:, news_cols] == 0).all()


def test_sentiment_is_scaled_over_training_rows_with_news(panel, train):
    f = fit(panel, train)
    on = np.isin(panel.days, train)[:, None] & panel.in_index & (panel.news[:, :, 0] > 0)
    s = col(f, "sentiment")[on]
    assert s.mean() == pytest.approx(0, abs=1e-6) and s.std() == pytest.approx(1, abs=1e-5)


def test_embedding_components_are_standardised_on_training_news_rows(panel, train):
    f = fit(panel, train)
    on = np.isin(panel.days, train)[:, None] & panel.in_index & (panel.news[:, :, 0] > 0)
    z = f.x[on][:, [i for i, n in enumerate(f.names) if n.startswith("emb_pca")]]
    assert z.shape[1] == DIM
    assert np.allclose(z.mean(axis=0), 0, atol=1e-5) and np.allclose(z.std(axis=0), 1, atol=1e-4)


def test_pca_is_fitted_on_training_rows_only(panel, train):
    f = fit(panel, train)
    later = panel.emb.copy()
    later[np.isin(panel.emb_at[:, 0], np.flatnonzero(~np.isin(panel.days, train)))] += 50.0
    g = fit(dataclasses.replace(panel, emb=later), train)
    assert np.allclose(g.stats["pca"]["mean"], f.stats["pca"]["mean"])
    assert np.array_equal(g.x[:40], f.x[:40])


def test_news_and_embeddings_can_be_switched_off(panel, train):
    assert not any(n.startswith("emb_pca") for n in fit(panel, train, embeddings=False).names)
    names = fit(panel, train, news=(), embeddings=False).names
    assert names == ["ret_1d", "volatility_20d", "rsi_14", "macd", "macd_signal", "volume_z_20d", FULL_WINDOW]


# --------------------------------------------------------------------------- windows and look-ahead


def test_window_is_the_days_up_to_and_including_the_row_date(panel, train):
    f = fit(panel, train)
    w = f.windows(45, np.array([0, 2]))
    assert w.shape == (2, W, len(f.names))
    assert np.array_equal(w[1, -1], f.x[45, 2]) and np.array_equal(w[1, 0], f.x[45 - W + 1, 2])
    early = f.windows(1, np.array([0]))
    assert (early[0, : W - 2] == 0).all() and np.array_equal(early[0, -1], f.x[1, 0])


def test_features_on_a_day_do_not_depend_on_any_later_data(panel, train):
    day = panel.days[45]
    cut = {n: t[t["date"] <= day] for n, t in tables().items()}
    short = build_panel(**cut)
    a, b = fit(panel, train), fit(short, train)
    assert np.allclose(a.windows(45, np.arange(6)), b.windows(45, np.arange(6)))


def test_frame_gives_tabular_baselines_the_same_features(panel, train):
    f = fit(panel, train)
    rows = panel.rows(panel.days[40:], 1)
    table = f.frame(rows, panel)
    assert list(table.columns) == ["date", "ticker", "y", *f.names]
    k = list(panel.tickers).index(rows["ticker"].iloc[0])
    d = panel.day_index(rows["date"].iloc[:1])[0]
    assert np.allclose(table[f.names].iloc[0], f.x[d, k])


def test_unknown_feature_names_are_rejected(panel, train):
    with pytest.raises(ValueError, match="unknown features"):
        fit(panel, train, price=("close",))
