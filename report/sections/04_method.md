# Method

> Owner: Modelling · Issue #47 · Due Fri 20 Nov

For every NASDAQ-100 member *i* on trading day *t*, the model outputs P(*y*<sub>*i,t*</sub><sup>(*h*)</sup> = 1),
the probability that the close *h* trading days later (*h* = 1 or 5) is above the close on *t*.
It may use only information available at *t*'s close. The model has three parts:

```
per company, last 20 trading days            per day, all members in the index
price indicators + news features ──► LSTM ──► h_i ──┬──────────────────────────► [ h_i ; ELU(g_i) ] ──► linear ──► P(up)
                                                    └─► GAT, one per edge type ─► g_i  (summed over types)
                                                          sector · comention · corr
```

A separate model is trained for each horizon and each of five walk-forward folds
(`configs/walk_forward.yaml`, shared by every model). Each fold trains on a rolling window of
504 trading days, skips a 5-day embargo (the longer horizon) and tests on the next 126 days.
The five test windows are consecutive half-years from 8 January 2021 to 12 July 2023. Within a
fold, every scaling statistic, every PCA component and the `comention` and `corr` edge weights
are fitted on that fold's training window only.

## Inputs

**Row set.** A prediction is made for every (day, ticker) with `in_index` true on that day and a
label for the horizon (`src/models/panel.py`, shared by every model, including the baselines). A
row is never dropped for incomplete history. A newly listed firm has little price history of its
own, and that is the case propagation is meant to help.

**Features.** Each company-day has a vector of 74 features:

| Group | Features | Scaling (training fold only) |
|---|---|---|
| Price (6) | trailing 1-day return, 20-day volatility, RSI(14), MACD, MACD signal, 20-day volume z-score | z-score over in-index training rows; missing → 0 |
| Completeness (1) | `has_full_window`: 1 if every indicator is observed on all 20 days of the window | none |
| News scalars (3) | `has_news`, mean FinBERT `sentiment`, mean `confidence` | z-score over training rows **with news**, then 0 on rows without |
| Embedding (64) | mean FinBERT [CLS] embedding, reduced by PCA | PCA fitted on training rows with news, each component standardised; exactly 0 on rows without news |

Two of these choices keep "unknown" separate from "average":

- **No news.** Rows without news are never passed through the scaler or the PCA. PCA maps the
  zero vector to −mean·components, not to 0, so transforming them would give every uncovered
  company-day the same non-zero embedding. With `has_news` = 0 and all other news features at 0,
  "no coverage" stays distinct from neutral coverage.
- **Missing indicators.** An indicator in warm-up, or on a day before listing, is set to 0, the
  training mean. `has_full_window` = 0 marks the window, so the model can tell an imputed value
  from an observed average one.

The LSTM reads a **20-trading-day window** ending on, and including, day *t*. Every feature in it
uses data up to that day's close only. FNSPID gives only a date for almost every article, so an
article dated *D* counts from the first trading day after *D* (Section 3).

## Graph construction

Nodes are companies. Edges are directed, and an undirected relation is stored in both
directions. There are no self-loops, because the company's own state reaches the output through
a skip connection (below). Two share classes of one company (GOOG/GOOGL, FOX/FOXA,
LBTYA/LBTYK) are never joined by a `comention` or `corr` edge, since their near-identical prices
and shared news would dominate both weights. On each day, the graph is restricted to the
companies in the index that day. The edges are built by `src/graph/build_edges.py` into
`edges.parquet` (SCHEMA.md).

| Type | Source | Weight | Sparsity | Refitted per fold |
|---|---|---|---|---|
| `sector` | GICS industry | 1 | every pair in the same industry | no (static) |
| `comention` | articles naming both firms | co-mention Jaccard, [0, 1] | ≥ 2 shared articles, top 10 per ticker | yes, training window |
| `corr` | daily returns | partial correlation given the market, [−1, 1] | ≥ 126 shared days, top 10 per ticker | yes, training window |

For both top-k types, a pair selected from either side is kept in both directions. A ticker's
degree can therefore exceed 10.

Undirected pairs per fold, with the number of companies that have at least one edge of the type
and their mean degree:

| Fold | `comention` pairs | companies | mean degree | `corr` pairs | companies | mean degree |
|---|---|---|---|---|---|---|
| 0 | 920 | 143 | 12.9 | 858 | 122 | 14.1 |
| 1 | 914 | 143 | 12.8 | 843 | 121 | 13.9 |
| 2 | 908 | 143 | 12.7 | 863 | 123 | 14.0 |
| 3 | 734 | 135 | 10.9 | 841 | 122 | 13.8 |
| 4 | 756 | 135 | 11.2 | 831 | 120 | 13.8 |

`sector` edges are the same in every fold: 940 pairs over 169 companies, mean degree 11.1,
maximum 24 (Software). All selected `corr` weights are positive (median 0.22–0.25), although
the type allows negative ones. `comention` weights are small (median 0.004–0.005), because
Jaccard divides by each firm's total coverage.

### `sector` edges

Two companies are linked when they share a GICS *industry* (`src/graph/gics.csv`; 182
ever-members in 42 industries). The industry level sits between the 11 sectors, too coarse to
separate semiconductors from software, and the 59 sub-industries, where 29 companies would have
no peer at all (13 at industry level). The classification is the pre-March-2023 GICS structure
for every company, the one in force for almost the whole 2019–2023 window. 117 companies are
taken from Wikipedia's S&P 500 table, 11 of them reverted to their pre-2023 code, and 65 are
filled by hand. The 43 hand-filled companies inside the window are cross-checked against
Yahoo's industry: 26 agree, 3 disagree and 14 have no Yahoo entry. `sector` is the only edge
type not refitted per fold. GICS is assigned by a classification committee, not learned from
the returns being predicted.

### `comention` edges

The weight is the Jaccard overlap of the two firms' training-window articles,
*w*<sub>*ab*</sub> = co<sub>*ab*</sub> / (*n*<sub>*a*</sub> + *n*<sub>*b*</sub> − co<sub>*ab*</sub>), where an article naming *k* companies
counts 1 / (*k* − 1) toward every total (SCHEMA.md decision 6). Both adjustments strip a
component shared by every pair:

- **Coverage.** A raw count ranks heavily covered pairs first. AAPL–MSFT (775 shared articles)
  then outranks genuine peers such as EA–TTWO (181). Jaccard divides by the firms' own coverage,
  which puts EA–TTWO first in fold 4 (0.136, against 0.116 for AAPL–MSFT).
- **Roundups.** Articles naming many firms are mostly market roundups. The 10% of
  multi-company articles naming three or more firms produce 48% of all pairs. The 1 / (*k* − 1)
  weight discounts them smoothly, without a hard cutoff on *k*.

A pair needs at least 2 articles naming both firms, and each ticker keeps its 10 peers with the
highest weight. A count threshold alone left the maximum degree at 41–130 of about 140 nodes. A
minimum of 5 would remove about 25 low-coverage firms, the ones propagation is meant to reach.
Measurements: [`comention_report.md`](../../results/tables/comention_report.md).

### `corr` edges

The weight is the partial correlation of two firms' daily returns given the equal-weight
market, with each stock's beta fitted on the fold's training window (SCHEMA.md decision 7). For
stock *i*, the residual *e*<sub>*i,t*</sub> = *r*<sub>*i,t*</sub> − α<sub>*i*</sub> − β<sub>*i*</sub>*m*<sub>*t*</sub> is computed, where *m*<sub>*t*</sub> is the mean
return of the members in the index that day, and pairs are correlated on these residuals. Three
choices matter:

- **Market adjustment.** On raw returns every pair carries the market (mean pair correlation
  about 0.4), so top-k ranks pairs by beta similarity. MSFT then becomes a 67-degree hub and the
  median selected edge is 0.52–0.60. On residuals the maximum degree is 31, the median selected
  edge is 0.22–0.25, and the overlap with `sector` edges rises from 24–31% to 29–35%.
- **Equal-weight market, not QQQ.** QQQ is cap-weighted with AAPL near a tenth of it, so
  subtracting it from AAPL would partly subtract AAPL from itself.
- **Minimum overlap of 126 days.** Under the null, a correlation's standard error is about
  1 / √(*n* − 1). Across about 7,600 candidate pairs, the largest spurious value lands near 3.8
  standard errors: 0.49 at 60 shared days, above most genuine edges, and 0.34 at 126. At 126
  days, at most 2 of about 120 candidates per fold lose all `corr` edges; they keep their other
  types.

The choice of market adjustment matters most in volatile regimes. In folds 0–2 it barely
moves the selected edges, but in folds 3–4, whose training windows include the 2022 rate
shock, beta dispersion widens (p10–p90 from 0.6–1.4 to 0.4–1.7), and subtracting the market
with beta = 1 instead of the fitted beta raises the median selected edge from 0.23 to 0.32–0.34.
Measurements: [`corr_report.md`](../../results/tables/corr_report.md).

**Validation: pending acquisitions.** Without any deal data, `corr` edges pick up the
stock-for-stock acquisitions between index members. In fold 4, ADI–MXIM, AMD–XLNX and AZN–ALXN
are each the top peer of one side (partial correlation 0.72–0.88), on 164, 282 and 138 shared
days, just above the 126-day minimum: those edges exist only for the months the deal was pending
before the target delisted. ADI–MXIM and AMD–XLNX are already selected in folds 0–2, as
genuine semiconductor peers, before the deals dominate. The all-cash MSFT–ATVI deal, pending
throughout folds 3–4, is not picked up (−0.04 in fold 4): a cash offer pins the target to a
fixed price rather than to the acquirer's stock. So the edges respond to how two prices are
actually tied together, not to a label saying a relationship exists. See Limitations for why
these particular edges are a different mechanism from the one under test; the `deal_arb`
ablation removes them.

## Model

The model is `GATLSTM` in `src/models/gat_lstm.py`.

**Company encoder.** A one-layer LSTM with 64 hidden units reads each company's 20-day window
of the 74 features. Its final hidden state, after dropout of 0.1, is the company's state
**h**<sub>*i*</sub> ∈ ℝ<sup>64</sup> (`CompanyEncoder`, `src/models/encoder.py`). The same weights are shared by every
company. The LSTM runs before the graph, so propagation acts on temporal summaries rather than
raw daily features, following Feng et al. (2019).

**Graph attention, one layer per edge type.** For each edge type *r*, a graph attention layer
(Veličković et al., 2018; PyTorch Geometric `GATConv`) with *M* = 4 heads computes a message
for company *i* from its in-neighbours *N*<sub>*r*</sub>(*i*) on that day:

- **Attention:** α<sup>(*r,m*)</sup><sub>*ij*</sub> = softmax<sub>*j* ∈ *N*<sub>*r*</sub>(*i*)</sub> LeakyReLU(**a**<sub>dst</sub><sup>⊤</sup>**W h**<sub>*i*</sub> + **a**<sub>src</sub><sup>⊤</sup>**W h**<sub>*j*</sub> + **a**<sub>e</sub><sup>⊤</sup>**W**<sub>e</sub>*w*<sub>*ij*</sub>)
- **Message:** **g**<sub>*i*</sub><sup>(*r*)</sup> = (1/*M*) Σ<sub>*m*</sub> Σ<sub>*j*</sub> α<sup>(*r,m*)</sup><sub>*ij*</sub> **W**<sup>(*r,m*)</sup>**h**<sub>*j*</sub> + **b**<sup>(*r*)</sup>

Here *w*<sub>*ij*</sub> is the edge weight, which enters the attention score as an edge feature, not as a
fixed multiplier, and the four heads are averaged. Each type has its own parameters, the
weight matrix and attention vectors (both written without the superscript *r,m* above) and the
bias **b**<sup>(*r*)</sup>.

**Combining the types.** The three messages are summed, **g**<sub>*i*</sub> = Σ<sub>*r*</sub> **g**<sub>*i*</sub><sup>(*r*)</sup>. A linear layer
reads the company's own state next to the graph signal and outputs the logit:

logit<sub>*i*</sub> = **v**<sup>⊤</sup>[**h**<sub>*i*</sub> ; ELU(**g**<sub>*i*</sub>)] + *c*, and P(up) = σ(logit<sub>*i*</sub>).

Three properties follow from this design:

- **Signed propagation.** The message transforms are unconstrained, and each type has its own.
  A type can therefore learn to pass a neighbour's state on with the opposite sign: a shock that
  shifts market share within an industry moves competitors in opposite directions. The sign is
  learned, not imposed.
- **The proposed model contains Baseline 6.** The skip connection [**h**<sub>*i*</sub> ; ·] means the
  graph adds to the company's own state rather than replacing it. Built with no edge types, the
  class has no GAT branch and reduces to logit = **v**<sup>⊤</sup>**h**<sub>*i*</sub> + *c*: that is Baseline 6 (LSTM +
  news, no graph). Both models are this class, trained by the same function, and the encoder is
  built first, so under one seed both start from identical encoder weights. Baseline 5 is
  Baseline 6 without the news features. Any gap between the proposed model and Baseline 6 is
  therefore the graph's (`tests/test_models.py`).
- **Isolated nodes.** A company with no in-neighbour of type *r* on a day receives only that
  type's bias **b**<sup>(*r*)</sup>. Attention is a softmax over a company's incoming edges of one type,
  so it sums to 1 whether or not the graph moves the prediction, and a company's only edge of a
  type gets attention 1 whatever its weight. Section 7 therefore reports each prediction's
  graph contribution beside its attention.

The proposed model has 88,385 parameters: 35,840 in the encoder, 52,416 in the three attention
layers and 129 in the output layer. Baseline 6 has 35,905.

## Training

**Batches.** Each trading day is one graph: its nodes are the members in the index that day,
with that fold's edges between them. Eight days, drawn in random order, form one optimiser step
as a single disjoint graph, so no message crosses from one day to another. The loss is binary
cross-entropy averaged over the labelled nodes of the batch. Optimisation uses Adam (learning
rate 10<sup>−3</sup>, weight decay 10<sup>−4</sup>) with gradients clipped at norm 1.0, for at
most 40 epochs.

**Validation split.** The last 10% of each fold's 504 training days (51 days) is held out. The 5
days before it are used for neither fitting nor validation, the same embargo as between training
and test, so that a 5-day label on the last fitted day cannot reach into the validation days.
That leaves 448 fitted days, or 56 optimiser steps per epoch. The test window is never used for
any choice: not the checkpoint, the threshold or the configuration.

**Early stopping on within-day AUC.** Validation is checked every 14 steps (4 times per
epoch), and the checkpoint with the highest validation within-day AUC is kept: for each day,
the AUC among that day's companies, averaged over days. Training stops after 4 epochs (16
checks) without improvement. Within-day AUC is the project's headline metric, chosen because
the research question is cross-sectional. It scores only how the day's companies are ranked
against each other. Validation loss also rewards calling each
day's overall level, which is market timing and has nothing to do with propagation between
companies. The two criteria chose checkpoints more than one epoch apart in 29 of 60 tuning
fold-horizons. The validation-loss optimum is logged beside the chosen checkpoint in every run
record (`predictions/<model>_run.json`).

**Operating threshold.** AUC needs no threshold. For accuracy, F1, MCC and the backtest, each
fold-horizon model uses the cutoff on P(up) that maximises MCC on its validation days. The
candidates are the 1st–99th percentiles of the validation predictions, so the cutoff always
splits them. The threshold is recorded in the run record, since it is not a column of
`predictions`.

**Seeds.** Seed 42 gives the official predictions handed to Evaluation. Seeds 43 and 44 are
used to measure seed-to-seed variation in the ablations (`results/tables/ablations_3_9.md`).
Retraining at seed 42 reproduces the official predictions bit for bit. On a CPU a
fold-horizon takes 15–50 seconds (5–16 epochs before early stopping).

## Configuration search

The search (task 3.12, `src/models/tune.py`) logged six configurations of the proposed model.
Every score is on the validation split inside each training fold, never the test windows, as
the mean over five folds. Selection takes the highest validation within-day AUC averaged over
both horizons, so a single configuration is shared by both horizons. Validation MCC is logged
beside it.

| Config | Batch days | LR | Hidden | Heads | Dropout | Val within-day AUC 1d | 5d | Mean | Val MCC |
|---|---|---|---|---|---|---|---|---|---|
| T0 | 32 | 1e-3 | 64 | 4 | 0.1 | 0.531 | 0.498 | 0.515 | 0.058 |
| **T1** | **8** | **1e-3** | **64** | **4** | **0.1** | **0.543** | **0.512** | **0.528** | **0.043** |
| T2 | 8 | 3e-4 | 64 | 4 | 0.1 | 0.537 | 0.511 | 0.524 | 0.058 |
| T3 | 8 | 1e-4 | 64 | 4 | 0.1 | 0.532 | 0.506 | 0.519 | 0.060 |
| T4 | 16 | 3e-4 | 64 | 4 | 0.1 | 0.533 | 0.505 | 0.519 | 0.057 |
| T5 | 8 | 3e-4 | 64 | 4 | 0.3 | 0.537 | 0.514 | 0.525 | 0.064 |

Full table, with checkpoint steps and run times: [`tuning_gat_lstm.md`](../../results/tables/tuning_gat_lstm.md).

- **Why batch size was searched first.** T0, the starting point, made only 14 optimiser steps per
  epoch, and in the first fold-0 run validation loss bottomed out after the first epoch. Fewer days per step (T1–T3, T5)
  gives 56 steps per epoch, and every eight-day configuration beat T0 on validation.
- **The choice.** T1 is chosen, but it leads the runner-up T5 by only 0.002. Each score is the
  best of many validation checks on the same ~50 days per fold, so it is optimistic, and the
  margin is within selection noise. T1 is also the best configuration at 1 day; at 5 days T5 is
  0.002 better.
- **Baselines.** Baselines 5 and 6 are trained at T1's settings too
  (`configs/lstm_price.yaml`, `configs/lstm_news.yaml`, held equal to `configs/gat_lstm.yaml` by
  a test). The ladder 5 → 6 → proposed then changes one component per step. Task 4.13 tunes the
  two baselines separately (`results/tables/tuning_baselines.md`).

How far the validation gains carried over to the test windows is reported in Section 6.

## Ablations and graph contribution

**Ablations.** Each ablation is the same training run with one component removed
(`python -m src.models.train --ablate …`, written as `gat_lstm_no_<what>`):

- one edge type at a time (`sector`, `comention`, `corr`);
- the news features, embeddings included (`news`), which keeps the graph and so differs from
  Baseline 5;
- the `corr` edges of the three acquisition pairs above, ADI–MXIM, AMD–XLNX and AZN–ALXN, with
  their other edges kept (`deal_arb`).

Dropping all three edge types gives exactly Baseline 6 at the proposed model's settings. Each
row is run at three seeds (`src/models/ablations.py`).

**Graph contribution.** For each test prediction, `GATLSTM.explain` recomputes the logit with
**g**<sub>*i*</sub> set to zero, and with each edge type's message removed in turn. The difference from the
full logit is the graph's contribution to that prediction, overall and per type. The
contribution is also split into a day-level shift (its mean over the day's companies) and a
within-day part. Only the within-day part can change a ranking. These outputs, with attention
weights per edge, are in `results/attention/` and are used in Section 7. Removing the message
keeps the trained dense layer, so this is the model without its graph branch, not Baseline 6
(which has its own trained head).

## Leakage controls

| Invariant | Where it is enforced |
|---|---|
| Survivorship: a ticker is in the sample only while `in_index` is true | `Panel.row_mask`; daily graphs include only in-index nodes |
| News after the close belongs to the next trading day | News alignment (Section 3); features for *t* use data to *t*'s close |
| Scaling, PCA, `comention` and `corr` weights fitted on training days only | `fit_features`; `build_edges` per fold |
| Walk-forward folds with a 5-day embargo, shared by every model | `src/eval/splitter.py`, `configs/walk_forward.yaml` |
| Model selection never sees the test window | embargoed validation split for the checkpoint, threshold and configuration |
| No coverage stays distinct from neutral sentiment | `has_news`; news features exactly 0 on rows without news |
