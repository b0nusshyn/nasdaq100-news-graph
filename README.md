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
| `comention` | Articles naming both firms | Empirically the strongest spillover channel (Ge et al., 2025) |
| `corr` | Return correlation, training window only | Captures links not visible in text |

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
Return correlations used as edge weights are computed within the training window only.

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
- **Embargo period** between train and test equal to the prediction horizon, so a 5-day label
  cannot overlap the test window.
- **Normalisation statistics** are fitted on the training fold only.
- **Metrics:** accuracy, F1, MCC, AUC.
- **Backtest:** cumulative return, Sharpe ratio and maximum drawdown, net of transaction costs.

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
│   └── demo.ipynb         # End-to-end run
├── results/               # Tables and figures
├── report/                # Report, one Markdown file per section
├── SCHEMA.md              # Data contract
├── CHECKLIST.md           # Task breakdown and deadlines
└── README.md
```

---

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch Geometric must match your CUDA and PyTorch versions. If installation fails locally,
run the model stages on Colab, where compatible wheels are preinstalled. Check your install with:

```bash
python -m src.core.check_env    # versions + one tiny GAT forward pass
```

For the shared code and tests only, `pip install -r requirements-dev.txt` is enough
(this is what CI runs):

```bash
ruff check .
pytest
```

### Data location

Artifacts live in the team's Google Drive folder. Point `NDX_DATA_DIR` at it (Colab and local
instructions in [`data/README.md`](data/README.md)). To build against synthetic data before the
real artifacts exist, set `NDX_DATA_DIR=tests/fixtures`.

### Building the dataset

```bash
# Price data and labels
python -m src.data.build_index_history
python -m src.data.fetch_prices --start 2018-01-01 --end 2023-12-31
python -m src.data.build_labels --horizons 1 5

# News features (GPU recommended)
python -m src.nlp.encode_news --input data/raw/fnspid_filtered.csv

# Graph edges
python -m src.graph.build_edges --types sector comention corr
```

### Training and evaluation

```bash
python -m src.models.train --config configs/gat_lstm.yaml
python -m src.eval.run_baselines --all
python -m src.eval.compare --output results/
```

---

## Results

Pending. The table below will be completed after the evaluation runs.

| Model | Accuracy | F1 | MCC | AUC | Sharpe (net) |
|---|---|---|---|---|---|
| Buy-and-hold | — | — | — | — | — |
| ARIMA | — | — | — | — | — |
| Logistic regression | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| LSTM (price only) | — | — | — | — | — |
| LSTM + news, no graph | — | — | — | — | — |
| **Proposed (GAT + LSTM)** | — | — | — | — | — |

An ablation study will report the contribution of each edge type and of the news features
independently.

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