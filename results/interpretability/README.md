# Worked interpretability examples (task 2.10, #43)

Owner: NLP. Three test predictions of the official proposed model (`gat_lstm`, tuned config T1, seed 42), each traced from the prediction back through its incoming edges to the neighbours' headlines. Regenerate everything with:

```
python -m src.nlp.examples
```

This needs Modelling's 3.10 outputs in `<data>/attention/` (see `results/attention/README.md`) and NLP's raw files in `<data>/raw/`. The tracing code is `src/nlp/interpret.py` (2.9). One example from the command line:

```
python -m src.nlp.interpret --date 2023-05-17 --ticker SGEN --horizon 1
```

## Files

| File | Content |
|---|---|
| `<name>.png` | figure for the report: (a) how much each edge type moved the logit, (b) attention per neighbour against the uniform share 1/k, (c) daily headline sentiment of the main neighbour over the 20-day input window |
| `<name>.md` | the full trace: prediction, contribution per edge type, top 3 neighbours per type with their strongest headlines |
| `<name>.csv` | every (shown neighbour, article in its window) row behind the trace |
| `sentiment_alignment.csv` | over all test predictions: does the graph push the way the neighbours' news points? |

## How the examples were picked

All three are 1-day predictions taken from the 100 candidates in `results/attention/top_contributions.csv`, which Modelling ranked by the largest within-day graph contribution. They were chosen to show three different situations, not as the best cases. They are not a random sample, so they illustrate the mechanism and say nothing about how often it happens.

## The examples

### 1. SGEN, 17 May 2023: a target without news, moved by one weak edge (`sgen_2023-05-17_h1`)

- **Prediction:** P(up) 0.91, against 0.49 without the graph; the price went up.
- **Graph contribution:** +2.41 on the logit. Most of it (+1.98) came through the comention type.
- **The edge behind it:** SGEN has a single comention neighbour, VRTX, with edge weight 0.003. Because it is the only edge of its type, it gets attention 1.0 by construction (a softmax over one edge).
- **The news:** SGEN had no news that day. VRTX had 16 news days in the window, with mean sentiment +0.03, mostly Q1 earnings coverage with mixed signs.

This is the project's mechanism in its plainest form: a company with no coverage of its own receives a signal through the graph. The trace also shows the weakness:
- The edge carrying the signal is almost the weakest possible.
- The neighbour's news is close to neutral.

The same lone VRTX edge drives 6 of the top 50 1-day candidates for SGEN, pushing in both directions, with 4 right and 2 wrong. WBD's lone comention edge from CMCSA (weight 0.005) appears 4 times. **Limitation:** the edge weight enters the GAT only as an input to the attention score (`edge_dim=1`). A softmax over a single edge is 1.0 whatever its weight, so a near-zero comention edge can pass on its neighbour's whole message.

### 2. LCID, 17 April 2023: an intuitive pair (`lcid_2023-04-17_h1`)

- **Prediction:** P(up) 0.04, against 0.49 without the graph; the price went down.
- **Graph contribution:** -3.03 on the logit, split roughly evenly between corr (-1.15) and comention (-1.13), with sector adding -0.61.
- **Where the attention went:** RIVN, the other listed EV start-up, gets the highest attention in every edge type. In corr it gets 0.37 against a uniform share of 0.10, on the largest corr weight (0.56).
- **The news:** RIVN's headlines are few (6 days) and mildly positive (+0.10), e.g. "How Long Will Rivian's Cash Last?".

The graph found the economically sensible neighbour and the forecast was right. The push, however, is negative while the neighbour's news is mildly positive. What travelled along the edge is RIVN's state, which includes its price history, not its headline sentiment.

### 3. GOOG, 9 May 2023: concentrated attention, wrong forecast (`goog_2023-05-09_h1`)

- **Prediction:** P(up) 0.04, against 0.49 without the graph; the price went up.
- **Graph contribution:** -3.06 on the logit, mostly from corr (-1.73) and sector (-0.88).
- **Where the attention went:** among 9 corr neighbours, META gets 0.47 against a uniform share of 0.11, one of the clearest concentrations in the test set. META also leads the sector type (0.62 against 0.50).
- **The news:** META's headlines in the window are clearly positive (+0.26), covering its Q1 beat on 26–28 April.

The graph pushed GOOG down hard while the neighbour it attended to had positive news, and the forecast was wrong. GOOGL on the same day is an almost identical case (row 2 of `top_contributions.csv`). The GAT can single out a neighbour, but here it did not turn that neighbour's good news into an up signal for GOOG.

## Across all test predictions (`sentiment_alignment.csv`)

The examples suggest that the direction of the graph's push does not follow the neighbours' news. To check this beyond three cases, every test prediction was compared on two quantities:
- **Graph push:** `graph_logit_within_day`, the part of the graph's effect that can change the day's ranking.
- **Neighbours' news:** the attention-weighted mean headline sentiment of the neighbours over the input window, also demeaned within the day. Neighbours without news in the window are left out.

| Horizon | Predictions | Spearman | Same sign |
|---|---|---|---|
| 1 day | 64,082 | +0.011 | 49.8% |
| 5 days | 64,070 | -0.002 | 48.5% |

There is no relation: agreement is at coin-flip level. This fits the other evidence:
- **News features (3.9):** they add a small within-day AUC gain at 1 day (+0.004 to +0.005, positive in 3 of 3 seeds; `results/tables/ablations_3_9.md`).
- **Comention edges (3.9):** they add nothing (-0.0009 with news features). These are the edges built from the news.
- **Attention (3.10):** close to uniform (`results/attention/summary.md`).

Together: news helps a company's own forecast, but the graph does not pass on the neighbours' headline sentiment in any direction-consistent way.

## How to read these traces (for the report)

- **Attention:** it is a softmax over a company's incoming edges of one type. It ranks neighbours within a type and sums to 1 per type. A lone edge always gets 1.0.
- **How much the graph mattered:** this is `graph_logit`, not the attention. Only the within-day part changes the ranking that within-day AUC scores.
- **What an edge carries:** the neighbour's LSTM state over the whole 20-day window, built from price, news and embedding features. Articles are therefore listed over the window, and a sentiment sign can't be read straight off an edge.
- **Reach:** with one GAT layer, only direct neighbours can move a prediction.
- **Status:** these are pointers for discussion, not causal evidence. The "graph off" logit sets the GAT output to zero in a model never trained that way, so it is a sensitivity, not a counterfactual.
