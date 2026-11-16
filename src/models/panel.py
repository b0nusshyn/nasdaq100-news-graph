"""Model input shared by the proposed model and every baseline. Owner: Modelling. Issues #29, #35.

The comparison table only means something if every model predicts the same rows from the same
features, normalised the same way. Baselines import this module instead of preparing their own
panel; a change here changes every model at once, which is the point.

**Row set** (`Panel.rows`): every (date, ticker) with `in_index` true and a label for the
horizon. Nothing is dropped for an incomplete history: a newly listed firm has little history of
its own, which is the case graph propagation is meant to help.

**Features** (`fit_features`), all fitted on one fold's training days only:

- Price indicators (`src.data.indicators`) are z-scored over the fold's in-index training rows.
  A missing value (indicator warm-up, days before listing) becomes 0, the training mean, and the
  `has_full_window` flag is 0 for any row whose window holds one, so "unknown" stays distinct
  from "average" (the same pattern as `has_news`).
- `sentiment` and `confidence` are z-scored over the training rows that have news, then set to 0
  on every row without news: scaling must not turn "no coverage" into a constant signal.
- FinBERT embeddings are reduced by PCA fitted on the training rows that have news, each
  component standardised over those rows, and left at exactly 0 on rows without news (PCA
  maps the zero vector to -mean @ components, not to 0, so those rows are never transformed).

Windows are the `window` trading days up to and including the row's date: every feature uses
data up to that day's close only.

**Validation and threshold** (`split_validation`, `choose_threshold`): every model holds out
the same last part of each training fold, embargoed from the days it fits on, and picks its
operating threshold there, never on the test window. AUC needs no threshold and is the
headline metric; accuracy, F1 and MCC use the chosen threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.schema import EMB_COLS
from src.data.indicators import FEATURES as INDICATORS
from src.data.indicators import compute_indicators

NEWS_SCALARS = ("has_news", "sentiment", "confidence")
FULL_WINDOW = "has_full_window"


@dataclass(frozen=True)
class Panel:
    """Dense day x ticker arrays over every trading day in `prices`. Unscaled; NaN = missing."""

    days: pd.DatetimeIndex  # D trading days
    tickers: pd.Index  # K tickers
    price: np.ndarray  # [D, K, len(INDICATORS)] float64
    in_index: np.ndarray  # [D, K] bool
    labels: dict[int, np.ndarray]  # horizon -> [D, K] float64, NaN where unlabelled
    news: np.ndarray  # [D, K, 3] has_news, sentiment, confidence; 0 where no news
    emb_at: np.ndarray  # [M, 2] (day, ticker) of each row with news
    emb: np.ndarray | None  # [M, 768] float32 FinBERT embedding of those rows, or None

    def day_index(self, dates) -> np.ndarray:
        idx = self.days.get_indexer(pd.DatetimeIndex(dates))
        if (idx < 0).any():
            raise ValueError("dates outside the panel's trading calendar")
        return idx

    def row_mask(self, dates, horizon: int) -> np.ndarray:
        """[D, K] mask of the row set on `dates`: in the index and labelled for `horizon`."""
        on = np.zeros(len(self.days), dtype=bool)
        on[self.day_index(dates)] = True
        return on[:, None] & self.in_index & ~np.isnan(self.labels[horizon])

    def rows(self, dates, horizon: int) -> pd.DataFrame:
        """The rows every model predicts on `dates`: `date`, `ticker`, `y` (int8), sorted."""
        d, k = np.nonzero(self.row_mask(dates, horizon))
        return pd.DataFrame({
            "date": self.days[d], "ticker": self.tickers[k].astype("string"),
            "y": self.labels[horizon][d, k].astype("int8"),
        })


def build_panel(
    prices: pd.DataFrame, labels: pd.DataFrame, news_features: pd.DataFrame, *, embeddings: bool = True
) -> Panel:
    """Assemble the panel from SCHEMA.md tables. `news_features` needs the `emb_*` columns only
    when `embeddings` is true."""
    days = pd.DatetimeIndex(sorted(prices["date"].unique()))
    tickers = pd.Index(sorted(prices["ticker"].unique()))
    shape = (len(days), len(tickers))

    def at(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d, k = days.get_indexer(df["date"]), tickers.get_indexer(df["ticker"])
        keep = (d >= 0) & (k >= 0)
        return d[keep], k[keep], keep

    ind = compute_indicators(prices)
    price = np.full((*shape, len(INDICATORS)), np.nan)
    d, k, keep = at(ind)
    price[d, k] = ind.loc[keep, list(INDICATORS)].to_numpy(float)

    in_index = np.zeros(shape, dtype=bool)
    d, k, keep = at(prices)
    in_index[d, k] = prices.loc[keep, "in_index"].to_numpy(bool)

    horizons = sorted(int(c[2:-1]) for c in labels.columns if c.startswith("y_") and c.endswith("d"))
    y = {h: np.full(shape, np.nan) for h in horizons}
    d, k, keep = at(labels)
    for h in horizons:
        y[h][d, k] = labels.loc[keep, f"y_{h}d"].astype("Float64").to_numpy(float, na_value=np.nan)

    news = np.zeros((*shape, len(NEWS_SCALARS)))
    covered = news_features[news_features["has_news"].astype(bool)]
    d, k, keep = at(covered)
    news[d, k] = covered.loc[keep, list(NEWS_SCALARS)].to_numpy(float)
    emb = covered.loc[keep, list(EMB_COLS)].to_numpy(np.float32) if embeddings else None
    return Panel(days, tickers, price, in_index, y, news, np.stack([d, k], axis=1), emb)


def load_panel(*, embeddings: bool = True) -> Panel:
    """`build_panel` on the artifacts in the data directory (`NDX_DATA_DIR`)."""
    from src.core.io import read_table

    news_cols = None if embeddings else ["date", "ticker", *NEWS_SCALARS]
    return build_panel(read_table("prices"), read_table("labels"), read_table("news_features", columns=news_cols),
                       embeddings=embeddings)


@dataclass(frozen=True)
class FoldFeatures:
    """Scaled features for one fold: `x[d, k]` is ticker k's feature vector on day d."""

    names: list[str]
    x: np.ndarray  # [D, K, F] float32, no NaN
    window: int
    stats: dict  # what was fitted on the training days, for inspection and tests

    def windows(self, day: int, tickers: np.ndarray) -> np.ndarray:
        """[N, window, F]: the `window` days up to and including `day` for each ticker. Days
        before the calendar starts are zeros (and `has_full_window` is 0 for such rows)."""
        lo = day - self.window + 1
        out = self.x[max(lo, 0): day + 1][:, tickers]
        if lo < 0:
            out = np.concatenate([np.zeros((-lo, *out.shape[1:]), np.float32), out])
        return out.transpose(1, 0, 2)

    def frame(self, rows: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        """Each row's own-day features, for tabular baselines: `rows` plus one column per feature."""
        d = panel.day_index(rows["date"])
        k = panel.tickers.get_indexer(rows["ticker"])
        return pd.concat([rows.reset_index(drop=True), pd.DataFrame(self.x[d, k], columns=self.names)], axis=1)


def fit_features(
    panel: Panel,
    train_dates,
    *,
    window: int,
    price: tuple[str, ...] = INDICATORS,
    news: tuple[str, ...] = NEWS_SCALARS,
    embeddings: bool = True,
    embedding_dim: int = 64,
) -> FoldFeatures:
    """Scale every feature with statistics from the in-index rows on `train_dates` only."""
    unknown = (set(price) - set(INDICATORS)) | (set(news) - set(NEWS_SCALARS))
    if unknown:
        raise ValueError(f"unknown features: {sorted(unknown)}")
    on = np.zeros(len(panel.days), dtype=bool)
    on[panel.day_index(train_dates)] = True
    train = on[:, None] & panel.in_index  # [D, K]
    parts, names, stats = [], [], {}

    cols = [INDICATORS.index(c) for c in price]
    raw = panel.price[:, :, cols]
    if price:
        vals = raw[train]
        seen = (~np.isnan(vals)).any(axis=0)  # a feature never observed in training scales to 0
        mean = np.where(seen, np.nanmean(np.where(seen, vals, 0.0), axis=0), 0.0)
        std = np.where(seen, np.nanstd(np.where(seen, vals, 0.0), axis=0), 1.0)
        std = np.where(std > 0, std, 1.0)
        parts.append(np.nan_to_num((raw - mean) / std, nan=0.0))
        names += list(price)
        stats["price"] = {"mean": mean, "std": std}

    # A row's window is complete if every selected indicator is observed on each of its days.
    bad = np.isnan(raw).any(axis=-1).astype(np.int32)
    run = np.concatenate([np.zeros((1, bad.shape[1]), np.int32), np.cumsum(bad, axis=0)])
    d = np.arange(len(panel.days))
    lo = np.maximum(d - window + 1, 0)
    full = ((run[d + 1] - run[lo]) == 0) & (d >= window - 1)[:, None]
    parts.append(full[:, :, None].astype(float))
    names.append(FULL_WINDOW)

    has_news = panel.news[:, :, 0] > 0
    for c in news:
        v = panel.news[:, :, NEWS_SCALARS.index(c)]
        if c != "has_news":
            ref = v[train & has_news]
            mean, std = (ref.mean(), ref.std()) if ref.size else (0.0, 1.0)
            v = np.where(has_news, (v - mean) / (std if std > 0 else 1.0), 0.0)
            stats[c] = {"mean": mean, "std": std}
        parts.append(v[:, :, None])
        names.append(c)

    if embeddings:
        if panel.emb is None:
            raise ValueError("embeddings requested but the panel was built without them")
        ed, ek = panel.emb_at[:, 0], panel.emb_at[:, 1]
        fit = train[ed, ek]
        if fit.sum() <= embedding_dim:
            raise ValueError(f"{fit.sum()} training rows with news: too few for {embedding_dim} PCA components")
        e = panel.emb.astype(np.float64)
        mu = e[fit].mean(axis=0)
        centred = e[fit] - mu
        eigval, eigvec = np.linalg.eigh(centred.T @ centred / (len(centred) - 1))
        comp = eigvec[:, ::-1][:, :embedding_dim]  # columns: top components, largest first
        z = (e - mu) @ comp
        z_std = z[fit].std(axis=0)
        z /= np.where(z_std > 0, z_std, 1.0)
        dense = np.zeros((len(panel.days), len(panel.tickers), embedding_dim))
        dense[ed, ek] = z  # rows without news are never transformed: they stay exactly 0
        parts.append(dense)
        names += [f"emb_pca_{i}" for i in range(embedding_dim)]
        stats["pca"] = {"mean": mu, "components": comp, "std": z_std, "explained": eigval[::-1][:embedding_dim]}

    x = np.concatenate(parts, axis=-1).astype(np.float32)
    return FoldFeatures(names, x, window, stats)


def split_validation(train_days: pd.DatetimeIndex, fraction: float, embargo: int):
    """(fit days, validation days): the last `fraction` of a fold's training days validate, and the
    `embargo` days before them are used for neither, so a label on the last fitted day cannot
    reach into the validation days."""
    n_val = max(1, math.ceil(fraction * len(train_days)))
    fit = train_days[: len(train_days) - n_val - embargo]
    if len(fit) == 0:
        raise ValueError("training window too short for the validation split and embargo")
    return fit, train_days[-n_val:]


def choose_threshold(y: np.ndarray, p: np.ndarray, *, candidates: int = 99) -> float:
    """The cutoff on P(up) that maximises MCC on validation rows (`y` labels, `p` probabilities).

    Candidates are quantiles of `p`, so the threshold always splits the predictions: a model whose
    probabilities all sit above 0.5 still gets a cutoff inside their range. Returns 0.5 when no
    candidate gives a defined MCC (constant predictions or one-class labels).
    """
    y, p = np.asarray(y).astype(bool), np.asarray(p, dtype=float)
    if len(y) == 0 or y.all() or not y.any():
        return 0.5
    t = np.unique(np.quantile(p, np.linspace(0.01, 0.99, candidates)))
    up = p[None, :] >= t[:, None]  # [thresholds, rows]
    tp, fp = (up & y).sum(1).astype(float), (up & ~y).sum(1).astype(float)
    fn, tn = y.sum() - tp, (~y).sum() - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.where(denom > 0, denom, 1), -np.inf)
    return float(t[np.argmax(mcc)]) if np.isfinite(mcc).any() else 0.5
