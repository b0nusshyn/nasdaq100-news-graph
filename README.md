# Stock News Propagation for NASDAQ-100 Trading Prediction

Predicting daily price direction for NASDAQ-100 constituents by propagating news signals
through a company relationship graph.

**Status:** in development · results pending

---

## Motivation

When NVIDIA reports strong earnings, the market prices that news into NVDA within seconds.
A model that reacts to the headline is already too late.

But NVIDIA does not operate alone. Other firms manufacture its chips and license it designs.
A large order reaches those firms too — and the market may take longer to notice.

Most news-driven models are **first-order**: they match each article to the company it names.
This project tests whether a **second-order** signal exists, and whether it is predictable.

---

## Research question

> Does propagating news signals through a company relationship graph improve price-direction
> prediction, compared to a model that associates each article only with the company it names?

**Task:** binary classification (up / down)
**Universe:** NASDAQ-100 constituents
**Period:** 2019–2023, daily frequency
**Horizons:** 1 day and 5 days

---

## Method

The model has four stages.

```
News articles ──► FinBERT ──► sentiment + embeddings
                                      │
Price history ──► LSTM ──► per-company hidden state
                                      │
                                      ▼
                         Graph Attention Network
                    (propagates signals along company edges)
                                      │
                                      ▼
                           Dense layer ──► P(up)
```

**1. News encoding.** FinBERT produces a sentiment score and a 768-dimensional embedding for
each headline. Named-entity matching assigns each article to one or more tickers.

**2. Graph construction.** Nodes are companies. Edges come from three sources:

| Edge type | Source | Rationale |
|---|---|---|
| `sector` | GICS classification | Firms in the same industry share demand shocks |
| `comention` | Articles naming both firms, training window of each fold | Empirically the strongest spillover channel (Ge et al., 2025) |
| `corr` | Return correlation net of the market, training window of each fold | Captures links not visible in text |

**3. Signal propagation.** A Graph Attention Network passes news features from the company
named in an article to its neighbours. Attention weights are learned, so the model infers
which relationship types carry signal rather than having them specified by hand.

**4. Temporal modelling.** An LSTM encodes recent price history and technical indicators per
company. Its hidden state is the node feature the GAT operates on.

### Design notes

- **Sparse news.** On a given day most constituents have no news at all. A binary `has_news`
  flag distinguishes "no coverage" from "neutral coverage" — the model must not conflate them.
  This sparsity is also the motivation for the graph: propagation fills in companies with no
  direct coverage.
- **Ordering.** The LSTM runs before the GAT, so the graph operates on temporal summaries
  rather than raw daily features. This follows Feng et al. (2019) and keeps tensor shapes
  tractable.
- **Signed propagation.** Sector edges are not assumed to transmit the same
  sign. A shock that rotates market share within an industry moves
  competitors in opposite directions, so propagation weights are left
  unconstrained and the sign is learned rather than imposed.

---

## Dataset

| Source | Content | Access |
|---|---|---|
| Yahoo Finance | Daily OHLCV for all constituents | `yfinance` |
| Wikipedia | Historical NASDAQ-100 membership changes | Scraped once |
| FNSPID | 15.7M financial news items, 1999–2023, timestamp-aligned to prices | HuggingFace |
| Investing.com | News and earnings dates, used to check coverage gaps | Public RSS |

**Survivorship bias.** Membership is reconstructed historically from the Wikipedia changes
table. Training on today's constituent list would restrict the sample to firms that survived
and inflate every reported metric.

**Look-ahead bias.** Articles published after 16:00 ET are assigned to the next trading day.
Both `corr` and `comention` edges are rebuilt per fold from that fold's training window only.
Co-mention counts aggregated over the full sample would tell the model which firms are
discussed together in the future. Only `sector` edges are static.

### Data contract

All intermediate artifacts are Parquet files under `data/`. The schema is fixed before
implementation begins — see [`SCHEMA.md`](SCHEMA.md).

| File | Key columns |
|---|---|
| `prices.parquet` | `date`, `ticker`, OHLCV, `in_index` |
| `news_features.parquet` | `date`, `ticker`, `sentiment`, `confidence`, `has_news`, `emb_0…emb_767` |
| `edges.parquet` | `src`, `dst`, `edge_type`, `weight` |
| `labels.parquet` | `date`, `ticker`, `y_1d`, `y_5d` |

---

## Evaluation

### Baselines

Baselines are arranged as a ladder so that each step isolates one component.

| # | Method | Isolates |
|---|---|---|
| 1 | Buy-and-hold | Whether any signal exists |
| 2 | ARIMA | Classical time-series, price only |
| 3 | Logistic regression | Simplest ML baseline |
| 4 | XGBoost | Strongest conventional tabular method |
| 5 | LSTM, price only | The contribution of news |
| 6 | **LSTM + news, no graph** | **The contribution of the graph** |

Baseline 6 is the decisive comparison. It is the proposed model minus the single component
under test. If the full model does not beat it by a statistically significant margin, the
propagation hypothesis is not supported, and that is the result we report.

### Protocol

- **Walk-forward validation**, not a random split. Each fold trains on a contiguous past
  window and tests on the window immediately following it.
- **Embargo** of 5 trading days — the longest horizon in the study — between every train and
  test window. The same embargo applies to all horizons so that every model is evaluated on
  identical folds.
- **Normalisation statistics** are fitted on the training fold only.
- **Metrics:** accuracy, F1, MCC, AUC.
- **Backtest:** cumulative return, Sharpe ratio and maximum drawdown, net of transaction costs
  assuming 5 bps per side.

---

## Repository structure

```
.
├── data/                  # Parquet artifacts (gitignored; shared via Google Drive, see data/README.md)
├── src/
│   ├── core/              # Shared: paths, schema validation, Parquet I/O, synthetic fixtures
│   ├── data/              # Price ingestion, index history, labels, indicators
│   ├── nlp/               # FNSPID filtering, ticker matching, FinBERT encoding
│   ├── graph/             # Edge construction
│   ├── models/            # Shared LSTM encoder, GAT + LSTM, training
│   │   └── baselines/     # Baselines 1–6, one module each
│   └── eval/              # Walk-forward splitter, metrics, backtest, significance, comparison
├── configs/               # gat_lstm.yaml, walk_forward.yaml (folds shared by every model)
├── tests/
│   └── fixtures/          # Synthetic copy of every Parquet artifact
├── notebooks/
│   ├── demo.ipynb         # End-to-end run
│   └── project_story.ipynb  # The project's findings, read from results/ (runs without the data)
├── results/               # Tables and figures
├── report/                # Report, one Markdown file per section
├── SCHEMA.md              # Data contract
├── CHECKLIST.md           # Task breakdown and deadlines
└── README.md
```

---

## Setup

Requires Python 3.10+. CI uses Python 3.11. On macOS with Homebrew,
install the OpenMP runtime required by XGBoost before running the full suite:

```bash
brew install libomp
```

```bash
git clone https://github.com/b0nusshyn/nasdaq100-news-graph.git
cd nasdaq100-news-graph
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch Geometric must match your CUDA and PyTorch versions. If installation fails locally,
run the model stages on Colab, where compatible wheels are preinstalled. Check your install with:

```bash
python -m src.core.check_env    # versions + one tiny GAT forward pass
```

For the shared code and tests only, `pip install -r requirements-dev.txt` is enough
(CI's light job):

```bash
ruff check .
pytest -rs    # -rs lists the tests skipped for a missing heavy dependency
```

Tests that need torch, scikit-learn, statsmodels, xgboost or transformers skip on the light set.
The figure integration tests also need `Rscript` on PATH and working R PNG graphics with Cairo.
Install R (for example, `brew install r` on macOS or `r-base-core` on Ubuntu) and check its
graphics support before running the full suite with the same no-skip setting as CI:

```bash
Rscript --vanilla -e 'stopifnot(capabilities("png"), capabilities("cairo")); f <- tempfile(fileext=".png"); png(f, type="cairo"); plot(1:3); dev.off(); stopifnot(file.exists(f))'
NDX_NO_SKIPS=1 pytest -rs
```

The no-skip run allows skips only for tests marked `real_data` when private
team artifacts are unavailable.

### Data location

Artifacts live in the team's Google Drive folder. Point `NDX_DATA_DIR` at it (Colab and local
instructions in [`data/README.md`](data/README.md)). To build against synthetic data before the
real artifacts exist, set `NDX_DATA_DIR=tests/fixtures`.

### Building the dataset

```bash
# Price data and labels
python -m src.data.build_index_history
# Prices run past 2023 on purpose: the main experiment is bounded to 2019–2023 by
# configs/walk_forward.yaml, but the lead-lag persistence check needs prices through 2026.
python -m src.data.fetch_prices --start 2018-01-01 --end 2026-11-30
python -m src.data.build_labels --horizons 1 5

# News features (GPU recommended)
python -m src.nlp.encode_news --input data/raw/fnspid_filtered.csv

# Graph edges: every walk-forward fold in one run. corr and comention are rebuilt from
# each fold's training window; sector edges are static (fold -1) and shared by every fold.
# Also writes edges_manifest.json: parameters, input hashes and build time per fold and type.
python -m src.graph.build_edges --types sector comention corr
# --fold N rebuilds fold N's corr and comention rows, but it also rewrites the sector rows
# that the other four folds read, and leaves those folds' own rows from earlier runs. The
# file then mixes builds (the run warns); rebuild in full, without --fold, before a hand-off.
python -m src.graph.build_edges --types sector comention corr --fold 0
# Measurements behind the comention defaults (SCHEMA.md decision 6)
python -m src.graph.comention_report --output results/tables
# Measurements behind the corr defaults (SCHEMA.md decision 7)
python -m src.graph.corr_report --output results/tables
```

### Training and evaluation

```bash
python -m src.models.train --config configs/gat_lstm.yaml
# Smoke run on one fold: prints wall-clock time and metrics, writes no predictions
python -m src.models.train --config configs/gat_lstm.yaml --folds 0
# Baselines 6 and 5: the same training loop with the GAT, then the news, removed
python -m src.models.train --config configs/lstm_news.yaml
python -m src.models.train --config configs/lstm_price.yaml
# The ladder 5 -> 6 -> proposed per fold and horizon, with the horizon-pattern flag
python -m src.models.ladder_report --output results/tables
python -m src.eval.run_baselines --all
python -m src.eval.compare --output results/
```

---

## Results

The tables report held-out results. Each classification metric is an unweighted
mean of the five walk-forward fold scores.
Within-day AUC, the headline metric, averages daily cross-sectional ROC AUC over dates
with both classes within each fold. Accuracy, positive-class F1 and MCC use each model's
validation-selected fold threshold, with a fixed always-up rule for Buy-and-Hold.
The three neural core models use the official T1 runs at seed 42.
[Task 4.10 production comparison](results/comparison_with_classical/tables/model_comparison.csv)

**Horizon: 1 day**

| Model | Accuracy | F1 | MCC | Within-day AUC | Net Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buy-and-Hold | 0.5057 | 0.6716 | N/A | 0.5000 | N/A |
| ARIMA | 0.4925 | 0.4719 | 0.0026 | 0.5005 | -2.7774 |
| Logistic Regression | 0.4999 | 0.5065 | 0.0095 | 0.5057 | -2.2162 |
| XGBoost | 0.5054 | 0.6034 | 0.0132 | 0.5036 | -2.5100 |
| LSTM Price Only | 0.4930 | 0.4334 | 0.0010 | 0.4995 | -2.6899 |
| LSTM + News, No Graph | 0.5032 | 0.4962 | 0.0032 | 0.5051 | -2.4323 |
| **Proposed GAT-LSTM** | 0.4994 | 0.3934 | 0.0024 | 0.5094 | -1.7502 |

**Horizon: 5 days**

| Model | Accuracy | F1 | MCC | Within-day AUC | Net Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buy-and-Hold | 0.5213 | 0.6844 | N/A | 0.5000 | N/A |
| ARIMA | 0.5114 | 0.5273 | 0.0121 | 0.5027 | -1.2366 |
| Logistic Regression | 0.5033 | 0.4378 | -0.0044 | 0.4904 | -0.9561 |
| XGBoost | 0.5082 | 0.5204 | 0.0093 | 0.4988 | -0.8439 |
| LSTM Price Only | 0.5023 | 0.6145 | -0.0096 | 0.4811 | -1.3398 |
| LSTM + News, No Graph | 0.4986 | 0.4853 | -0.0004 | 0.5072 | -0.9806 |
| **Proposed GAT-LSTM** | 0.5040 | 0.5575 | 0.0047 | 0.4992 | -0.5506 |

GAT-LSTM had the highest within-day AUC among the seven core models at 1 day,
followed by Logistic Regression and Baseline 6. At 5 days, Baseline 6 led,
followed by ARIMA. GAT-LSTM scored higher than Baseline 6 at 1 day (0.5094 vs
0.5051) and lower at 5 days (0.4992 vs 0.5072). These are descriptive rankings.
The [Task 4.9 paired fold MCC analysis](results/tables/significance_mcc.json)
found no statistically significant difference at either horizon (Holm-adjusted
*p* = 1.0000 for both). Inference covers MCC at seed 42 and leaves AUC
significance and model equivalence unestablished.

Net Sharpe is annualized from one continuous, cost-adjusted backtest per
model and horizon. The strategies rank stocks by predicted probability,
long the top 20% and short the bottom 20%, with half the capital on each side
and non-overlapping holding periods. The simulation assumes idealized same-close
execution and charges 5 bps per side. It omits slippage, borrow fees, liquidity
constraints and delisting recovery. Buy-and-Hold's constant up calls make MCC
undefined. Its Sharpe is unavailable because unresolved ALXN and XLNX
corporate-action valuations prevent completion of the separate long-only
initial-member basket.

[Task 4.13](results/tables/tuning_baselines.md) selected T1 for both LSTM
baselines using retrospective validation within-day AUC. Its validation scores
are separate from the held-out metrics above.

### Ablation

The Task 4.10 seed-42 predictions cover the same eligible stock-date rows and
labels in all five folds for every variant (64,082 at 1 day and 64,070 at 5 days).
The [comparison manifest](results/comparison_with_classical/tables/comparison_manifest.json)
records training commit `b5a88ed9` for all six ablations and `dd811f1c` for
the three official T1 core neural models. Comparisons with the full model are
descriptive across those training commits. The final column uses
[Task 3.9](results/tables/ablations_3_9.md) and shows the mean and min–max
range across seeds 42, 43 and 44 of the five-fold mean within-day AUC.

Dropping news removes `has_news`, sentiment, confidence and embeddings. Dropping
an edge type removes that type only. The deal-arbitrage variant removes the
ADI–MXIM, AMD–XLNX and AZN–ALXN pairs from `corr`. The combined variant drops
news features and `comention` edges while retaining `sector` and `corr`.

**Horizon: 1 day**

| Variant | Within-day AUC (seed 42) | Fold ROC AUC (seed 42) | MCC (seed 42) | Within-day AUC across seeds, mean (range) |
| --- | ---: | ---: | ---: | ---: |
| Proposed (full model) (`gat_lstm`) | 0.50942 | 0.4996 | 0.0024 | 0.50651 (0.50357–0.50942) |
| Drop news features (`gat_lstm_no_news`) | 0.50299 | 0.4990 | 0.0024 | 0.50238 (0.49833–0.50582) |
| Drop sector (`gat_lstm_no_sector`) | 0.50728 | 0.5041 | 0.0086 | 0.50521 (0.50383–0.50728) |
| Drop corr (`gat_lstm_no_corr`) | 0.50726 | 0.5011 | 0.0048 | 0.50544 (0.50395–0.50726) |
| Drop comention (`gat_lstm_no_comention`) | 0.51109 | 0.5084 | 0.0154 | 0.50739 (0.50343–0.51109) |
| Drop deal-arb corr pairs (`gat_lstm_no_deal_arb`) | 0.50941 | 0.4995 | 0.0003 | 0.50650 (0.50356–0.50941) |
| Drop news features and comention (`gat_lstm_no_comention_no_news`) | 0.50159 | 0.5005 | 0.0112 | 0.50244 (0.50142–0.50433) |

**Horizon: 5 days**

| Variant | Within-day AUC (seed 42) | Fold ROC AUC (seed 42) | MCC (seed 42) | Within-day AUC across seeds, mean (range) |
| --- | ---: | ---: | ---: | ---: |
| Proposed (full model) (`gat_lstm`) | 0.49918 | 0.5054 | 0.0047 | 0.49875 (0.49711–0.49996) |
| Drop news features (`gat_lstm_no_news`) | 0.49018 | 0.4931 | 0.0052 | 0.49525 (0.49013–0.50545) |
| Drop sector (`gat_lstm_no_sector`) | 0.49719 | 0.5024 | -0.0088 | 0.49961 (0.49719–0.50138) |
| Drop corr (`gat_lstm_no_corr`) | 0.50094 | 0.5090 | 0.0164 | 0.50100 (0.49832–0.50374) |
| Drop comention (`gat_lstm_no_comention`) | 0.50525 | 0.5086 | 0.0161 | 0.50351 (0.50103–0.50525) |
| Drop deal-arb corr pairs (`gat_lstm_no_deal_arb`) | 0.49918 | 0.5054 | 0.0043 | 0.49863 (0.49689–0.49982) |
| Drop news features and comention (`gat_lstm_no_comention_no_news`) | 0.49634 | 0.5046 | 0.0147 | 0.50000 (0.49549–0.50818) |

Fold ROC AUC is the mean of ROC AUC computed over all observations within
each fold. At 5 days, dropping co-mention edges gave the highest mean within-day
AUC across seeds among the graph variants. Removing the deal-arbitrage pairs
barely changed mean within-day AUC at either horizon. The seed ranges describe
run-to-run variation.

---

## Interpretability

Each prediction is accompanied by the attention weights that produced it. For example, a
prediction that ARM will rise can be traced to an NVIDIA earnings article and the edge type
through which the signal travelled.

Baseline 6 cannot produce this explanation, because it has no graph to trace through. On days
when a company has no coverage of its own, the baseline has nothing to point at.

---

## Team

| Role | Member | Responsibility |
|---|---|---|
| Data | Chotima | Price pipeline, index history, news filtering |
| NLP | Kornchawal | FinBERT encoding, entity matching, timestamp alignment |
| Modelling | Shayanis | Graph construction, GAT + LSTM |
| Evaluation | Nonlada | Baselines, walk-forward protocol, backtest |

---

## References

1. Cohen, L. & Frazzini, A. (2008). Economic links and predictable returns. *Journal of Finance*, 63(4).
2. Ge, S., Li, S. & Zheng, H. (2025). Diamond cuts diamond: News co-mention momentum spillover prevails in China. *Journal of Banking & Finance*, 171.
3. Feng, F., He, X., Wang, X., Luo, C., Liu, Y. & Chua, T.-S. (2019). Temporal relational ranking for stock prediction. *ACM TOIS*, 37(2).
4. Du, H., Lv, L., Wang, H. & Guo, A. (2025). TGNS: A transformer-based graph neural network for stock trend forecasting. *Information Sciences*, 720.
5. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P. & Bengio, Y. (2018). Graph attention networks. *ICLR*.
6. Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063.
7. Dong, Z. et al. (2024). FNSPID: A comprehensive financial news dataset in time series. arXiv:2402.06698.
8. Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2).

---

## Disclaimer

This is an academic project. Nothing here is investment advice, and the models are not
intended for live trading.
