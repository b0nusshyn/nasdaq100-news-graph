# Attention and graph contribution (task 3.10, #42)

Owner: Modelling. This is an analysis output for NLP's interpretability work (2.9, 2.10). It is not a pipeline artifact, so it is not in `SCHEMA.md`, and columns may be renamed without a team-wide change.

Written by `python -m src.models.attention`. The script retrains the official proposed model (`configs/gat_lstm.yaml`, tuned config T1, seed 42) on every walk-forward fold and horizon. It writes only if the retrained test predictions equal `predictions/gat_lstm.parquet` bit for bit, so every row below explains exactly the prediction Evaluation scores. Test windows only.

## Read attention together with contribution

**Attention says which neighbour, not how much.** Attention is a softmax over a company's incoming edges of one type. It sums to one whether or not the graph changed the prediction.

**`graph_logit` says how much.** Before treating an attention pattern as an explanation, check it:
- **Small `graph_logit`:** the graph barely moved the prediction. Concentrated attention on that prediction shows where the GAT looked, not what drove the forecast.
- **Overall:** within-day AUC with and without the graph signal is in `summary.md`. The 3.9 ablations found no detectable graph step (`results/tables/ablations_3_9.md`).

**In this model, attention is close to uniform** (`summary.md`):
- **Entropy:** 0.99–1.00 of its maximum for every edge type and horizon.
- **Top-weighted edge:** its median share is 1.03–1.08 times the 1/k it would get with no preference.

The GAT is effectively averaging over each company's neighbours of a type. A single edge's weight therefore says little beyond the fact that the edge exists.

## Files

| File | Where | Rows |
|---|---|---|
| `summary.md` | here (committed) | concentration per edge type, contribution distribution, top candidates |
| `top_contributions.csv` | here (committed) | the 50 predictions per horizon with the largest \|`graph_logit`\| |
| `gat_lstm_contribution.parquet` | `<data>/attention/` (team Drive) | one per test prediction (horizon, date, ticker) |
| `gat_lstm_edges.parquet` | `<data>/attention/` (team Drive) | one per incoming edge of a test prediction (horizon, date, src, dst, edge_type) |
| `gat_lstm_node_attention.parquet` | `<data>/attention/` (team Drive) | one per (prediction, edge type) with at least 2 incoming edges |
| `checkpoints/gat_lstm_h<h>_fold<f>.pt`, `checkpoints.json` | `<data>/attention/` (team Drive) | the 10 fold models (state dicts) and their config |

`<data>` is `$NDX_DATA_DIR` (default `data/`).

## `gat_lstm_contribution.parquet`

| Column | Description |
|---|---|
| `horizon`, `fold`, `date`, `ticker` | the prediction; same rows as `predictions/gat_lstm.parquet` |
| `y_true` | realised label |
| `has_news` | the ticker had news that day |
| `p_up` | the model's prediction (equals `predictions/gat_lstm.parquet`) |
| `p_up_no_graph` | the same model with the graph signal set to zero |
| `logit`, `logit_no_graph` | the same two, as logits |
| `graph_logit` | `logit - logit_no_graph`: how much the graph moved this prediction |
| `graph_logit_within_day` | `graph_logit` minus its mean over that day's predictions: the part that can change the day's ranking |
| `graph_logit_<type>` | `logit` minus the logit with that edge type's message removed (`sector`, `comention`, `corr`) |
| `in_degree_<type>` | incoming edges of that type on that day |

**Graph off.** The model's dense layer sees `[h ; elu(g)]`, where `h` is the company's own LSTM state and `g` is the summed GAT messages. "Graph off" sets `g = 0`: this model with its GAT branch removed. It is **not** Baseline 6, which has its own trained head. The model never saw `g = 0` during training, so read `graph_logit` as a sensitivity, not as a retrained counterfactual.

**Day-level and within-day parts.** Most of `graph_logit` is a shift shared by every ticker that day: 58% of its variance at 1 day and 67% at 5 days. That shift is market timing and doesn't change the day's ranking, which is what within-day AUC, the headline, scores. Use `graph_logit_within_day` to judge whether the graph changed how this ticker ranked against the others. `top_contributions.csv` is ranked on it.

**Per-type contributions.** ELU makes the edge types non-additive, so the `graph_logit_<type>` columns need not sum to `graph_logit`.

## `gat_lstm_edges.parquet`

| Column | Description |
|---|---|
| `horizon`, `fold`, `date` | the prediction the edge feeds |
| `src`, `dst` | the signal comes from `src` and goes to `dst`, the predicted ticker |
| `edge_type` | `sector`, `comention` or `corr` |
| `weight` | the edge weight from `edges.parquet` for this fold (1 for `sector`) |
| `alpha_0` … `alpha_3` | attention of each of the 4 heads; each sums to 1 over a `dst`'s incoming edges of one type on one day |
| `alpha_mean` | mean over heads, which also sums to 1 per group |
| `src_has_news` | `src` had news on `date` |
| `src_news_days` | days with news for `src` in the model's input window (the `window_days` trading days up to and including `date`) |

**Edges don't carry a single day's news.** `src`'s LSTM state, which is what the edge passes on, summarises its whole input window, not only `date`. To trace an edge back to an article, look up `src`'s articles in `article_tickers` over the window, not just on `date`.

## `gat_lstm_node_attention.parquet`

This table covers only predictions with at least 2 incoming edges of a type, because a single edge always gets weight 1.

| Column | Description |
|---|---|
| `horizon`, `fold`, `date`, `ticker`, `edge_type` | the prediction and one of its edge types |
| `k` | incoming edges of that type |
| `entropy` | Shannon entropy of the attention divided by log k, averaged over heads: 1 = uniform, 0 = all weight on one edge |
| `top_share` | the largest `alpha_mean` |
| `uniform_share` | 1 / k, what `top_share` would be with no preference |

## Recomputing one prediction

```python
from src.models.attention import explain
row, edges = explain("2023-05-17", "SGEN", horizon=1)   # needs the checkpoints in <data>/attention/
```

`explain` rebuilds the fold's features and graph from the training window, loads the checkpoint, and returns the prediction's contribution row and its incoming edges. It refuses checkpoints saved under a different `configs/gat_lstm.yaml`. If it can't run, the Parquet tables hold the same numbers.
