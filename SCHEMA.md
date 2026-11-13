# Data contract

> **Status: FROZEN** on Fri 13 Nov 2026 (issue #2). Change it only with the whole team's
> agreement, and record the change under [Decisions](#decisions).

Every artifact is a Parquet file in the data directory (the shared Google Drive folder; see
[`data/README.md`](data/README.md)). The machine-readable copy of this file is
[`src/core/schema.py`](src/core/schema.py); `tests/test_schema.py` fails if the two disagree,
so edit both together.

Write artifacts with `src.core.io.write_table(df, "<name>")` and read them with
`read_table("<name>")`. Both validate against this contract. Synthetic versions of every file
are in `tests/fixtures/` (regenerate with `python -m src.core.fixtures`).

## Conventions

- **`date`** is the trading day the row belongs to: tz-naive, midnight, `datetime64[ns]`.
  Market data uses the US/Eastern calendar.
- **`ticker`** is the company's **current** Yahoo Finance symbol, upper case. A renamed company
  keeps one ticker for its whole history (`FB` → `META`): map old symbols with
  `src.core.tickers.canonical_ticker` everywhere a ticker enters the pipeline (membership, prices,
  news matching). `GOOGL` and `GOOG` are separate share classes, so separate tickers.
- **News timing**: an article published after 16:00 ET, or on a non-trading day, belongs to the
  next trading day.
- **Folds**: `fold` is the walk-forward fold index from `src.eval.splitter` (0-based). `-1`
  means "valid in every fold".
- **dtype** names: `date`, `string`, `bool`, `int8/16/32`, `float32/64`, and `Int8` (nullable int).

## `membership`

File: `membership.parquet` · Owner: Data · Issue #6

One row per continuous spell a ticker spent in the NASDAQ-100.

| Column | dtype | nullable | Description |
|---|---|---|---|
| `ticker` | string | no | |
| `start` | date | no | First trading day in the index |
| `end` | date | yes | Last trading day in the index; null if still a member |

**Key:** `ticker`, `start`

## `prices`

File: `prices.parquet` · Owner: Data · Issues #7, #14

One row per trading day per ticker that was ever a member, including days outside the index
(needed for indicator warm-up). Filter on `in_index` before training or evaluating.

| Column | dtype | nullable | Description |
|---|---|---|---|
| `date` | date | no | |
| `ticker` | string | no | |
| `open` | float64 | no | Split- and dividend-adjusted |
| `high` | float64 | no | Adjusted |
| `low` | float64 | no | Adjusted |
| `close` | float64 | no | Adjusted |
| `volume` | float64 | no | |
| `in_index` | bool | no | Member of the NASDAQ-100 on `date` |

**Key:** `date`, `ticker`

## `labels`

File: `labels.parquet` · Owner: Data · Issue #15

Same rows as `prices`. `ret_hd = close[t+h] / close[t] - 1` over trading days; `y_hd = 1` if
`ret_hd > 0`, else `0` — a flat return (`ret_hd = 0`) is "not up", so `y_hd = 0`. The last `h` days of each ticker are null.

| Column | dtype | nullable | Description |
|---|---|---|---|
| `date` | date | no | |
| `ticker` | string | no | |
| `ret_1d` | float64 | yes | Forward 1-day return (used by the backtest) |
| `ret_5d` | float64 | yes | Forward 5-day return |
| `y_1d` | Int8 | yes | 1 = up over 1 day |
| `y_5d` | Int8 | yes | 1 = up over 5 days |

**Key:** `date`, `ticker`

## `article_tickers`

File: `article_tickers.parquet` · Owner: NLP · Issues #16, #17

One row per article per ticker it mentions, already assigned to its trading day. An article
naming two firms has two rows; this is what co-mention edges are built from.

| Column | dtype | nullable | Description |
|---|---|---|---|
| `article_id` | string | no | Stable id from FNSPID |
| `date` | date | no | Trading day after the 16:00 ET cutoff rule |
| `ticker` | string | no | |

**Key:** `article_id`, `ticker`

## `news_features`

File: `news_features.parquet` · Owner: NLP · Issues #23, #24

One row for **every** in-index (`date`, `ticker`), whether or not it had news. Several articles
on one ticker-day are combined with a plain (unweighted) mean. Rows without
news have `has_news = false`, `n_articles = 0`, and zeros everywhere else; models must use
`has_news` to tell "no coverage" from "neutral coverage".

| Column | dtype | nullable | Description |
|---|---|---|---|
| `date` | date | no | |
| `ticker` | string | no | |
| `has_news` | bool | no | `n_articles > 0` |
| `n_articles` | int32 | no | Articles assigned to this ticker-day |
| `sentiment` | float32 | no | Mean of FinBERT `p_pos - p_neg`, in [-1, 1] |
| `confidence` | float32 | no | Mean FinBERT max class probability |
| `emb_0` … `emb_767` | float32 | no | Mean FinBERT [CLS] embedding |

**Key:** `date`, `ticker`

## `edges`

File: `edges.parquet` · Owner: Modelling · Issues #8, #21, #22, #28

Directed edges; an undirected relation is stored in both directions. No self-loops.
`corr` edges: for each ticker, its 10 peers with the highest training-window return correlation
(`CORR_TOP_K` in `src/core/schema.py`); a pair selected from either side is stored both ways.

| Column | dtype | nullable | Description |
|---|---|---|---|
| `fold` | int16 | no | Fold whose training window produced the weight; -1 for `sector` |
| `src` | string | no | Ticker the signal comes from |
| `dst` | string | no | Ticker the signal goes to |
| `edge_type` | string | no | One of `sector`, `comention`, `corr` |
| `weight` | float32 | no | 1 for `sector`; co-mention count; return correlation |

**Key:** `fold`, `src`, `dst`, `edge_type`

## `predictions`

File: `predictions/<model>.parquet` · Owners: Modelling (proposed model), Evaluation (baselines)

One file per model, one row per test (`date`, `ticker`) per horizon. Every model must cover the
same rows so they can be compared pairwise.

| Column | dtype | nullable | Description |
|---|---|---|---|
| `model` | string | no | `gat_lstm`, `lstm_news`, … ; matches the file name |
| `horizon` | int8 | no | 1 or 5 |
| `fold` | int16 | no | Walk-forward fold of the test window |
| `date` | date | no | |
| `ticker` | string | no | |
| `p_up` | float32 | no | Predicted probability of `y = 1` |
| `y_true` | int8 | no | Realised label |

**Key:** `model`, `horizon`, `date`, `ticker`

## Decisions

Frozen Fri 13 Nov 2026 in issue #2.

1. **Renamed tickers** — one ticker per company, using the current symbol (`FB` → `META`).
   Rename map: `src/core/tickers.py`; Data extends it while building membership (task 1.1).
2. **Ties** — a flat forward return counts as not up: `y = 0`. Every row with a full horizon keeps a label.
3. **News aggregation** — plain mean of sentiment, confidence and embedding across a ticker-day's articles.
4. **`corr` sparsity** — top-k with k = 10 per ticker, by training-window correlation.
