# Interpretability

> Owner: NLP · Issue #48 · Due Fri 20 Nov

The research question rests on a mechanism: news about one company should move the forecast
for related companies through the graph. This section opens single predictions to see whether
that is what the trained model does. It uses three tools:

- **Per-prediction outputs (task 3.10):** attention weights and graph contributions written by
  Modelling for every test prediction.
- **Tracing (task 2.9):** follows a prediction back to the source articles.
- **Worked examples (task 2.10):** three traced cases plus one test over all predictions.

Everything here describes the official proposed model (tuned config T1, seed 42) on the
walk-forward test windows.

## Tracing a prediction

For one prediction, `src/nlp/interpret.py` builds the chain

```
headline (FinBERT sentiment) → neighbour --[edge type, attention, edge weight]--> company → P(up)
```

1. It takes the prediction's contribution row and its incoming edges from the 3.10 outputs.
2. It orders the edge types by how much each moved the logit.
3. Within each type it keeps the three neighbours with the highest attention.
4. It lists each neighbour's articles over the model's 20-trading-day input window, with
   headline and sentiment.

Articles are listed over the whole window, not just the prediction day, because an edge passes
on the neighbour's LSTM state, which summarises that whole window. The counts of days with news
agree with Modelling's own `src_news_days`.

## Reading attention and contribution

- **Attention is per edge type.** The model has one graph-attention layer per edge type (sector,
  co-mention, correlation) and sums their outputs. Attention is a softmax over a company's
  incoming edges of one type, so it ranks neighbours within a type, sums to 1 per type, and
  can't be compared across types.
- **Attention says which neighbour, not how much.** The graph's effect is measured separately,
  as the change in the logit when the graph signal is set to zero (`graph_logit`). Removing one
  edge type at a time gives its share. Only the part that differs between companies on the same
  day (`graph_logit_within_day`) can change the ranking that the headline metric scores.
- **Attention is nearly uniform.** Its entropy is 0.99–1.00 of the maximum for every edge type
  and horizon. The top edge's median share is only 1.03–1.08 times the uniform share 1/k
  (`results/attention/summary.md`). For most predictions the GAT averages its neighbours.
- **Only direct neighbours count.** There is one graph layer, so a signal does not travel two
  hops.
- **"Graph off" is a sensitivity, not a counterfactual.** It is computed in a model that never
  trained without its graph. The traces below point to evidence; they are not causal
  explanations.

## Worked examples

The three examples are 1-day predictions taken from Modelling's 100 candidates with the
largest within-day graph contribution (`results/attention/top_contributions.csv`). They were
chosen to show three different situations and are not a random sample. Figures, full traces and
article lists are in `results/interpretability/`.

In each figure:
- **(a)** the logit change per edge type;
- **(b)** each neighbour's attention against the uniform share (vertical tick), with the edge
  weight and the neighbour's window news;
- **(c)** the daily headline sentiment of the main neighbour.

### SGEN, 17 May 2023: signal for a company without news

![SGEN trace](../results/interpretability/sgen_2023-05-17_h1.png)

- **Prediction:** P(up) 0.91, against 0.49 with the graph off. The price rose.
- **Where it came from:** of the +2.41 graph contribution, +1.98 came through a single
  co-mention edge from Vertex (VRTX). SGEN had no news that day.

This is the mechanism the project proposes: a company without coverage receiving a signal from
a related one. The trace also shows two weaknesses:
- **The edge is nearly the weakest possible.** Its weight is 0.003, yet it gets attention 1.0
  because it is SGEN's only co-mention edge, and a softmax over one edge is 1 whatever its
  weight.
- **The neighbour's news is close to neutral.** VRTX's 16 news days in the window, mostly Q1
  earnings coverage, average +0.03.

The same lone edge appears in 6 of the 50 strongest 1-day candidates, pushing both up and down
(4 right, 2 wrong).

### LCID, 17 April 2023: the economically sensible neighbour

![LCID trace](../results/interpretability/lcid_2023-04-17_h1.png)

- **Prediction:** P(up) 0.04, against 0.49 with the graph off. The price fell.
- **Where it came from:** the graph pushed the logit by −3.03, split between correlation
  (−1.15), co-mention (−1.13) and sector (−0.61) edges.
- **Who it attended to:** Rivian (RIVN), the other listed EV start-up, gets the highest
  attention in every edge type: 0.37 against a uniform 0.10 among ten correlation neighbours,
  on the largest correlation weight (0.56).

The graph found the neighbour a human would pick, and the forecast was right. But RIVN's few
headlines (6 days) were mildly positive (+0.10) while the push was strongly negative. What
travelled along the edge was RIVN's state, built from its price history as well as its news,
not its headline sentiment.

### GOOG, 9 May 2023: concentrated attention, wrong forecast

![GOOG trace](../results/interpretability/goog_2023-05-09_h1.png)

- **Prediction:** P(up) 0.04, against 0.49 with the graph off. The price rose.
- **Where it came from:** the graph pushed the logit by −3.06, mostly through correlation
  (−1.73) and sector (−0.88) edges.
- **Who it attended to:** Meta (META) gets 0.47 of the correlation attention against a uniform
  0.11. That is 4.2 times the uniform share, more concentrated than 99.7% of 1-day predictions
  on correlation edges.
- **The neighbour's news:** META's headlines were clearly positive (+0.26), led by its Q1
  earnings beat on 26–28 April.

The model singled out one neighbour, which had good news, and pushed GOOG down. GOOGL on the
same day is almost identical.

## Does the graph pass on the neighbours' news?

All three examples share a pattern: the direction of the graph's push does not follow the
neighbours' headline sentiment. To test this on every test prediction, we compared two
quantities:

- **The push:** `graph_logit_within_day`.
- **The neighbours' news:** the attention-weighted mean headline sentiment of the neighbours
  over the input window, demeaned within the day. Neighbours without news in the window are
  left out.

| Horizon | Predictions | Spearman ρ | Same sign |
|---|---|---|---|
| 1 day | 64,082 | +0.011 | 49.8% |
| 5 days | 64,070 | −0.002 | 48.5% |

There is no relation: the signs agree at coin-flip rate (`results/interpretability/sentiment_alignment.csv`).
This fits with three results from the ablations (Section 3; `results/tables/ablations_3_9.md`):

- News features raise the graph model's within-day AUC (+0.004 to +0.005 at 1 day, all three
  seeds).
- The same features barely help the LSTM without a graph (+0.0016, one seed of three).
- The co-mention edges, built from the news, add nothing.

So news helps the graph model, but not by "good news at a neighbour, so up". Two channels
remain that this test cannot see:
- the 64 embedding components, which carry more than the sentiment score;
- interactions between a neighbour's news and its prices inside the LSTM state.

The test only detects a monotone, same-direction relation, and we did not separate these
channels.

## Limitations

- **Attention near uniform:** single weights carry little information beyond the edge's
  existence. A trace is informative mainly when attention is concentrated, as for GOOG.
- **Lone edges take full attention:** an edge that is a company's only one of its type gets
  attention 1.0 regardless of its weight. A near-zero co-mention edge can then pass on its
  neighbour's whole message, as for SGEN. Normalising attention across edge types, or scaling
  messages by edge weight, would prevent this.
- **Biased examples:** the examples come from the predictions the graph moved most. They show
  the mechanism, not how often it occurs.
- **Graph off is not a retrained model:** it measures sensitivity to the graph signal, not
  what a model trained without the graph would predict. That comparison is Baseline 6
  (Section 6).
- **Direction only:** the alignment test checks sentiment direction only (see above).
