# Candidate news experiment

## Decision

The full candidate data pipeline succeeded. The controlled model comparison did not establish a reproducible gain in within-day ranking. Price Only remains the default experimental control. Extra-session news produced the highest mean 1-day development AUC at 0.5003, but the effect reversed between seeds. At 5 days both news policies lost to Price Only in both reviewed windows at seed 42. The official GAT-LSTM and its results are unchanged.

## Inputs, encoding, and coverage

The experiment starts from the frozen external archive intersection in [reconstruction_v1](reconstruction_v1). The reconstruction retains exact article IDs and original article–ticker pairs, records 29,658 missing article IDs and quarantined links, and supplies two exchange-session availability proxies. Raw publication times and feed arrivals were never independently verified. Midnight UTC is treated as date-only. The first policy assigns the next eligible session under the frozen rules, and the second delays that assignment by one more exchange session. Neither policy can prove that archived content existed at the assigned time.

The local ProsusAI/finbert snapshot was pinned to `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`. Encoding used CPU float32, four PyTorch threads, batches of 64, 128 maximum tokens, and final-layer CLS embeddings. Sentiment is positive probability minus negative probability and confidence is the maximum class probability. All **156,452** ordered IDs were encoded once in **39** validated Zstandard-compressed Parquet chunks, with 768 finite embedding coordinates each. No article failed or was dropped. Resumable chunks are tied to the source and snapshot fingerprint in the encoding plan. No resume was needed for this completed run. The full encode-and-aggregate job took **1,686 seconds**, used **1.1 GiB** of compressed output, and left about **3.3 GiB** free. The writer checked the 2 GiB floor and reserved expected write space before each Parquet write.

| Policy | Assigned original pairs before in-index filter | Assigned links in daily tables | Ticker-days | Ticker-days with candidate news |
|---|---:|---:|---:|---:|
| Next session | 229,179 | 211,988 | 128,519 | 58,287 |
| Extra session | 229,168 | 211,988 | 128,519 | 58,290 |

Each daily table uses the same in-index price-date population as the historical feature table and exactly matches the per-day counts in the frozen candidate audit. Short chunks of article outputs were accumulated into float64 sums, with results stored as float32 means. Each article embedding is shared across ticker links. Reload checks verified hashes, finite daily data, exact article order against the frozen headline table, and zeros on rows with no assigned candidate articles. Those zeros represent absent candidate evidence. Missing-source and quarantined-source counts remain separate audit fields.

[Encoding manifest](encoded_cycle_v1/manifest.json) · [Article parts](encoded_cycle_v1/article_features) · [Next-session features](encoded_cycle_v1/next_session_news.parquet) · [Extra-session features](encoded_cycle_v1/extra_session_news.parquet)

## Design and run accounting

The feature arms were Price Only, Price + Next-Session News, and Price + Extra-Session News, with the graph disabled in each arm. All used six price indicators, a 20-session sequence, one 64-hidden-unit LSTM layer, dropout 0.1, Adam learning rate 0.001 with weight decay 0.0001, BCE, gradient clipping 1.0, batch size eight trading days, a 16-epoch cap, within-day AUC validation every 14 updates, and four epochs of patience. Both news arms add the news-presence flag, sentiment, confidence, and 64 PCA components. Scalers and PCA were fit only on the weight-fitting days. The classification cutoff was selected by validation MCC. Training AUC below is the fixed training-probe score at the saved checkpoint.

| Window | Weight fit | Checkpoint and cutoff | Later development assessment |
|---|---|---|---|
| Late A | 2022-04-28 to 2023-04-28 | 2023-05-08 to 2023-07-12 | 2023-07-20 to 2023-10-02 |
| Late B | 2022-07-21 to 2023-07-21 | 2023-07-31 to 2023-10-02 | 2023-10-10 to 2023-12-21 |

Each split has 252 fit sessions, 45 checkpoint sessions and 52 assessment sessions, with five unused sessions between partitions. Assessment populations are matched exactly on ticker, date and target. Late A has 5,252 eligible rows at both horizons. Late B has 5,251 at 1 day and 5,247 at 5 days, excluding unavailable SGEN forward targets under the existing rule. Forward labels were checked against exact future closes by the existing verification loader. These two windows had already been examined in the price study. Late B's checkpoint window overlaps Late A's assessment window. Seed changes do not create new market periods, and overlapping 5-day labels reduce effective replication.

The first twelve fits cover both arms of time delay and the baseline at both horizons on Late A and Late B at seed 42. A command intended for seed 43 omitted the seed flag and ran six 1-day seed-42 duplicates under a misleading filename. All six predictions matched the original predictions bit for bit. They are retained and counted against the budget, but excluded from aggregate comparisons. The corrected seed-43 run used the final six fits to repeat the 1-day comparison across both windows. **24 fits executed, 18 distinct fits analyzed.** The CLI now rejects a filename that disagrees with the requested seed, and run rows explicitly record the seed. A preflight check of that safeguard did not run a fit. No training job failed.

[Late A seed 42](cycle_seed42_w0.jsonl) · [Late B seed 42](cycle_seed42_w1.jsonl) · [6 repeated seed-42 fits with misleading filename](cycle_seed43_h1.jsonl) · [Verified seed 43](cycle_seed43_h1_verified.jsonl) · [Contrasts and descriptive block intervals](cycle_analysis_v1.json)

## Official historical performance

| Horizon | Accuracy | Within-day AUC | MCC |
|---|---:|---:|---:|
| 1d | 0.4994 | 0.5094 | 0.0024 |
| 5d | 0.5040 | 0.4992 | 0.0047 |

These are historical results on a different evaluation population and cannot serve as a matched improvement baseline for the new late-2023 development study.

## New matched results

| Horizon | Arm | Fits | Train AUC | Val AUC | Assessment AUC | Accuracy | Fit-majority accuracy | MCC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | price_only | 4 | 0.5222 | 0.5072 | 0.4962 | 0.5048 | 0.5369 | -0.0026 | 0.6172 |
| 1 | next_session | 4 | 0.5635 | 0.5091 | 0.4940 | 0.5277 | 0.5369 | -0.0022 | 0.5038 |
| 1 | extra_session | 4 | 0.5271 | 0.5059 | 0.5003 | 0.5352 | 0.5369 | 0.0097 | 0.4172 |
| 5 | price_only | 2 | 0.4993 | 0.5081 | 0.4954 | 0.5531 | 0.5226 | 0.0046 | 0.6332 |
| 5 | next_session | 2 | 0.6048 | 0.5202 | 0.4857 | 0.5762 | 0.5226 | -0.0068 | 0.4891 |
| 5 | extra_session | 2 | 0.6141 | 0.5251 | 0.4837 | 0.5284 | 0.5226 | -0.0257 | 0.6475 |

### Run ledger

| Seed | Window | Horizon | Arm | Checkpoint / steps | Train AUC | Val AUC | Assessment AUC | Accuracy | MCC | F1 | Fit-majority accuracy | Seconds |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | Late A | 1 | price_only | 28/168 | 0.4959 | 0.5178 | 0.5033 | 0.4985 | 0.0133 | 0.5652 | 0.5221 | 4.4 |
| 42 | Late A | 1 | next_session | 14/154 | 0.4905 | 0.5004 | 0.5175 | 0.5183 | -0.0064 | 0.0938 | 0.5221 | 4.7 |
| 42 | Late A | 1 | extra_session | 14/154 | 0.4975 | 0.4827 | 0.5125 | 0.5228 | 0.0196 | 0.2193 | 0.5221 | 6.1 |
| 42 | Late A | 5 | price_only | 28/168 | 0.4981 | 0.5339 | 0.4971 | 0.4990 | 0.0283 | 0.5138 | 0.4282 | 5.0 |
| 42 | Late A | 5 | next_session | 294/434 | 0.6894 | 0.5332 | 0.4820 | 0.5377 | -0.0288 | 0.2193 | 0.4282 | 16.2 |
| 42 | Late A | 5 | extra_session | 294/434 | 0.6860 | 0.5393 | 0.4820 | 0.4442 | -0.0406 | 0.5366 | 0.4282 | 14.5 |
| 42 | Late B | 1 | price_only | 126/266 | 0.5620 | 0.5049 | 0.4826 | 0.5239 | -0.0256 | 0.6541 | 0.5517 | 6.9 |
| 42 | Late B | 1 | next_session | 84/224 | 0.5759 | 0.5200 | 0.4848 | 0.5437 | -0.0002 | 0.6906 | 0.5517 | 6.9 |
| 42 | Late B | 1 | extra_session | 14/154 | 0.5574 | 0.5261 | 0.5095 | 0.5492 | 0.0327 | 0.6830 | 0.5517 | 5.5 |
| 42 | Late B | 5 | price_only | 126/266 | 0.5005 | 0.4823 | 0.4936 | 0.6072 | -0.0191 | 0.7527 | 0.6169 | 7.8 |
| 42 | Late B | 5 | next_session | 28/168 | 0.5203 | 0.5071 | 0.4894 | 0.6146 | 0.0153 | 0.7589 | 0.6169 | 5.8 |
| 42 | Late B | 5 | extra_session | 28/168 | 0.5422 | 0.5109 | 0.4854 | 0.6125 | -0.0109 | 0.7584 | 0.6169 | 5.7 |
| 42 (repeat) | Late A | 1 | price_only | 28/168 | 0.4959 | 0.5178 | 0.5033 | 0.4985 | 0.0133 | 0.5652 | 0.5221 | 4.3 |
| 42 (repeat) | Late A | 1 | next_session | 14/154 | 0.4905 | 0.5004 | 0.5175 | 0.5183 | -0.0064 | 0.0938 | 0.5221 | 4.7 |
| 42 (repeat) | Late A | 1 | extra_session | 14/154 | 0.4975 | 0.4827 | 0.5125 | 0.5228 | 0.0196 | 0.2193 | 0.5221 | 4.5 |
| 42 (repeat) | Late B | 1 | price_only | 126/266 | 0.5620 | 0.5049 | 0.4826 | 0.5239 | -0.0256 | 0.6541 | 0.5517 | 7.1 |
| 42 (repeat) | Late B | 1 | next_session | 84/224 | 0.5759 | 0.5200 | 0.4848 | 0.5437 | -0.0002 | 0.6906 | 0.5517 | 8.3 |
| 42 (repeat) | Late B | 1 | extra_session | 14/154 | 0.5574 | 0.5261 | 0.5095 | 0.5492 | 0.0327 | 0.6830 | 0.5517 | 6.3 |
| 43 | Late A | 1 | price_only | 14/154 | 0.4735 | 0.4921 | 0.5134 | 0.4897 | 0.0357 | 0.6385 | 0.5221 | 4.0 |
| 43 | Late A | 1 | next_session | 406/512 | 0.5906 | 0.5109 | 0.4891 | 0.5013 | 0.0095 | 0.5268 | 0.5221 | 15.0 |
| 43 | Late A | 1 | extra_session | 14/154 | 0.4813 | 0.4938 | 0.5015 | 0.5192 | -0.0077 | 0.0596 | 0.5221 | 5.4 |
| 43 | Late B | 1 | price_only | 70/210 | 0.5573 | 0.5139 | 0.4855 | 0.5071 | -0.0336 | 0.6109 | 0.5517 | 6.4 |
| 43 | Late B | 1 | next_session | 140/280 | 0.5970 | 0.5050 | 0.4845 | 0.5475 | -0.0119 | 0.7040 | 0.5517 | 9.9 |
| 43 | Late B | 1 | extra_session | 56/196 | 0.5722 | 0.5210 | 0.4777 | 0.5494 | -0.0058 | 0.7067 | 0.5517 | 7.1 |

The JSONL files retain configuration and input fingerprints, complete validation histories, threshold and score distribution, coverage strata, and links and hashes for each Parquet prediction file. Missing-source subgroups sometimes contain no date with both classes, so their within-day AUC is undefined.

## News contribution and stability

- At 1 day, seed 42 favored both proxies in both windows. Next-session gains were +0.0142 and +0.0022. Extra-session gains were +0.0092 and +0.0269. Seed 43 reversed all four contrasts, with next-session deltas −0.0243 and −0.0010 and extra-session deltas −0.0119 and −0.0078.
- Across seeds and windows, next-session news changed 1-day AUC by −0.0022 and extra-session news by +0.0041. Paired 10-session block intervals were approximately [−0.0165, +0.0138] and [−0.0091, +0.0160]. These are descriptive because there are only two dependent historical periods. Seeds were averaged within date before bootstrapping.
- At 5 days, next-session and extra-session news reduced mean AUC by 0.0096 and 0.0116. Both lost in both windows at seed 42, and the respective descriptive intervals still include zero. No second 5-day seed was run.
- The favorable seed-42 next-session 1-day contrasts were not concentrated in rows with candidate news. Restricting daily comparisons to those rows yielded +0.0008 in Late A and −0.0258 in Late B, while rows without assigned candidate news yielded +0.0300 and +0.0265. The delayed arm had a stronger covered-row contrast in Late B at seed 42 (+0.0422), which reversed at seed 43 (−0.0052). These conditional rankings are descriptive, and their AUCs do not add up to the overall AUC.
- Only 32 Late A and 21 Late B test rows per 1-day fit carried a known missing-source flag. The subgroup had only one or two dates with both classes. Its effect cannot be estimated reliably. Excluding flagged rows left the overall contrast directions and magnitudes very similar, but does not rule out other archive biases.
- Next-session news raised the mean training probe at 1 day from 0.5222 to 0.5635 while lowering assessment AUC from 0.4962 to 0.4940. At 5 days both news arms raised the probe to about 0.60–0.61, while assessment AUC fell to about 0.48. Extra-session 1-day news was closer to the price control in train-to-assessment gap, but offered no stable ranking gain.
- The highest individual assessment AUC was 0.5175 (1d, Late A, next-session, seed 42). The highest accuracy was 61.46% (5d, Late B, next-session, seed 42), below the 61.69% fit-majority baseline on that window. That model predicted Up on 98.17% of rows. Neither 0.75 target was met. No independently verified holdout was available or scored.

## Graph, tuning, and investment implications

The experiment reached the authorized 24-fit cap. The seed reversals and 5-day ranking losses also provide no reason to spend additional fits on graph complexity or hyperparameter changes on these dates. No graph variants or hyperparameter sweeps were run in this cycle. Prior controlled graph work also reported unstable effects in this project. In particular, the static industry relation still lacks verified historical vintage, and the source dates used for co-mentions remain proxy-timed.

Price Only is the defensible control with the settings above and no news inputs or graph. Extra-session news is a provisional 1-day research arm, not a selected replacement. At 5 days Price Only ranked better in both measured periods. There is no new evidence of profitability or a durable stock-ranking edge and no new portfolio backtest. Further realistic backtesting would be informative only after stable ranking evidence on unused periods. Missing original articles, potentially incorrect archive dates, unknown feed arrival and article revisions, retrospective prices and model selection, overlapping outcomes, and changing market periods all limit interpretation.

## Verification and reproducibility

Relevant full-encoding and daily payloads are under [encoded_cycle_v1](encoded_cycle_v1). Assessment predictions are in the four `cycle_*.predictions` directories beside their ledgers. [analyze_cycle.py](analyze_cycle.py) recomputes the paired contrasts and verifies prediction hashes. [encode_cycle.py](encode_cycle.py) resumes a matching incomplete encoding plan. The shared training and preprocessing modules were unchanged this cycle. The study runner was extended to save predictions, score distributions, and explicit seed provenance. Tests cover disk guards and paired evaluation.

All 30 locally available official comparison source hashes matched the stored manifest. The original data, predictions, checkpoints, results, documentation and prior experiments were preserved. No Git operations or downloads were performed.

**Recommendation:** retain Price Only as a research control and stop tuning this archive on the same windows. Obtain a genuinely unused period and credible item-level availability or a defensible audited source before interpreting a news or graph contrast as fresh evidence. No additional files or permissions are required to reproduce this exploratory cycle locally.

Final code verification recorded **101 passing tests** with the local FinBERT overlay. Lint passed for the changed Python files. The seed-filename guard rejected a deliberately mismatched invocation before creating output. The independent daily-file recheck initially used a mixed boolean/numeric array in its finiteness assertion. Restricting that assertion to numeric feature columns passed for both complete daily tables.
