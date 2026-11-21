# NASDAQ-100 direction and selective prediction study

## Decision

No model established a repeatable predictive gain. The strongest retrospective ranking candidate was a shallow boosted regressor trained on cross-sectionally demeaned forward returns. It ranked 1-day directions better than logistic regression in two windows and worse in the third. The best full-population accuracy was 53.75%, and best within-day AUC was 0.5395. Four nominal 5% slices exceeded 70%, but all were single-direction calls that tied their same-subset majority. No independent evaluation occurred.

## 1. Historical audit and data availability

The official artifacts report 1d accuracy/AUC/MCC of **0.4994/0.5094/0.0024** and 5d **0.5040/0.4992/0.0047**. These are historical official results. Every table below is a new retrospective experiment and has a different evaluation population.

The targets are strictly positive adjusted close-to-close returns at the exact first or fifth future exchange session. Missing future quotes produce missing labels. The existing verified loader recomputed exact-session forward returns for the local legacy-schema file before mapping names in memory. Historical membership is applied as of each date. The dataset audit documents five missing XLNX in-index sessions during its halt, and archive recovery kept acquired issuers on their own price series. The adjusted quotes are retrospective and their historical adjustment vintage is uncertain. No new label defect was established. The older inner-validation preprocessing problem is documented in the preceding study. The LSTM here uses weight-fit-only preprocessing. New logistic imputation and scaling fit on the weight-fit partition. Tree missing-value routing, bins and splits are learned on that same partition.

Official forecasts are generated using finalized session-close inputs. This study uses price and volume through that close and performs no executable-price or same-close backtest. No news features, date-derived provider timestamps, graph edges, or post-close source joins entered a feature. Forecasts are matched by date, ticker, and binary outcome. Five-day outcomes overlap within assessments, so inference uses whole dates in blocks and remains descriptive.

The official held-out predictions show a substantial day-level component. The variance of daily mean score was 62.0% of total score variance at 1d and 68.0% at 5d. Daily mean score correlations with realized daily Up rate were −0.041 and −0.021. Stock Up rates on positive versus negative equal-weight forward market days were 73.09% versus 28.05% at 1d and 71.45% versus 28.59% at 5d. An **unavailable ex-post oracle** choosing the realized majority direction each day would average 73.14% and 71.76%. The actual pooled always-Up baseline was 50.57% and 52.12%. These results show that the label is strongly market-sensitive, while the official score level did not reliably forecast that market component. A common score for every stock gets 0.50 within-day AUC regardless of whether it calls the day correctly.

Both existing price copies stop on December 29, 2023. The local membership file extends into 2026, but no matching adjusted prices or exit valuations were found. The credentials-free public Yahoo probe returned HTTP 429. No post-2023 source bars were accepted or mixed into the panel. The 101-security fixed-as-of cohort was defined before the probe, and no survivor-only substitute was used. **2026 remains unscored.** [Source attempt](data_availability.json) · [Frozen design](protocol.md)

## 2. Models, features, and experimental design

The market-only input is same-close equal-weight member log return at 1, 5, and 20 sessions, 20-session market volatility, positive-return breadth, and cross-sectional dispersion. The stock input adds one- and five-session stock returns, 20-session volatility-scaled momentum, 20-session market-relative momentum, 20-session realized volatility and rolling beta, trailing five-session log high/low range, five-session log-volume change, distance from the 40-session moving average, five-session market-relative return, and point-in-time member percentile ranks of five-session returns and volatility. Missing session bars are not forward-filled. Sector classification is used only for descriptive coverage, because the static GICS map has no independently verified historical vintages. A prefix-invariance test confirms future price rows do not change earlier features.

- **Market logistic:** market inputs only, median imputation, scaling, L2 logistic with C=0.25. Tests market-direction prediction without cross-sectional variation.
- **Stock logistic:** same estimator with market and stock inputs.
- **Boosted binary:** histogram gradients on original direction. 90 iterations, learning rate 0.05, seven leaves, minimum leaf count 150, L2=8, no early stopping.
- **Boosted relative:** same tree settings with squared-error regression on the stock's forward return minus the mean eligible forward return on its fit date. The relative outcome is only a training objective. Assessment labels remain original absolute direction.
- **Price Only LSTM:** six original indicators and window completeness, no news or graph, lookback 20, hidden size 64, one layer, dropout 0.1, Adam LR 0.001, weight decay 0.0001, clipping 1.0, eight trading days per batch, BCE, checks every 14 updates, four epochs patience, and a 16-epoch cap. This uses the existing architecture and optimizer settings in the matched study harness.
- **Relative 45:** 45 rather than 90 boosting iterations, all other settings fixed. This is the only follow-up tuning change. [Follow-up design](followup_protocol.md)

The 20 initial fits cover five models, two windows and both horizons. Six follow-up fits test 1-day regularization across three periods, including the third-period matched controls. **26 total fits** at seed 42, six distinct configurations. There was no seed replication of the neural reference. No news, graph model, or larger search was justified. All fits completed. The 26 model/scoring stages took **38.4 seconds** combined on local CPU, excluding setup. The study used two CPU threads for the model and numerical libraries. The directory occupied about **7.8 MiB** before this report. Approximately 10 GiB free was observed at the end, above the 2 GiB floor.

| Assessment window | Fit | Calibration or checkpoint | Assessment |
|---|---|---|---|
| W0 | 2020-04-15–2021-04-14 | 2021-04-22–2021-07-01 | 2021-07-12–2021-12-31 |
| W1 | 2021-04-21–2022-04-19 | 2022-04-27–2022-07-08 | 2022-07-18–2022-12-30 |
| W2 | 2021-10-19–2022-10-18 | 2022-10-26–2023-01-06 | 2023-01-17–2023-06-30 |

Every fit has 252 sessions, then five unused sessions, 50 calibration sessions, then five unused sessions before assessment. W2 was evaluated at 1d only. Calibration uses a one-score standardized logistic model with C=1 and a 0.5 decision threshold. The LSTM's checkpoint also uses the calibration block, which can make its validation scores optimistic. The calibration can reverse a score if its direction on calibration is negative. Rank IC uses the native, uncalibrated score and continuous forward-return response. The models do not see their own assessment outcomes during fitting, threshold selection, or calibration. All windows are within the previously studied archive.

## 3. Complete development results

| Window | Horizon | Model | Rows | Val AUC | Assessment AUC | Accuracy | Fit majority | Assessment majority | MCC | F1 | Rank IC | Up share |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | market_logistic | 12433 | 0.5000 | 0.5000 | 0.5181 | 0.5203 | 0.5203 | 0.0007 | 0.6706 | NA | 0.9427 |
| 0 | 1 | stock_logistic | 12433 | 0.4958 | 0.5103 | 0.5210 | 0.5203 | 0.5203 | 0.0121 | 0.6746 | 0.0128 | 0.9519 |
| 0 | 1 | boosted_binary | 12433 | 0.5071 | 0.5009 | 0.5095 | 0.5203 | 0.5203 | -0.0029 | 0.6184 | -0.0042 | 0.7650 |
| 0 | 1 | boosted_relative | 12433 | 0.4972 | 0.5150 | 0.5202 | 0.5203 | 0.5203 | 0.0037 | 0.6820 | -0.0220 | 0.9883 |
| 0 | 1 | price_lstm | 12433 | 0.5037 | 0.4877 | 0.5197 | 0.5203 | 0.5203 | 0.0026 | 0.6785 | -0.0297 | 0.9738 |
| 0 | 5 | market_logistic | 12425 | 0.5000 | 0.5000 | 0.5326 | 0.5326 | 0.5326 | NA | 0.6951 | NA | 1.0000 |
| 0 | 5 | stock_logistic | 12425 | 0.4688 | 0.5014 | 0.5326 | 0.5326 | 0.5326 | NA | 0.6951 | -0.0069 | 1.0000 |
| 0 | 5 | boosted_binary | 12425 | 0.4951 | 0.5062 | 0.5375 | 0.5326 | 0.5326 | 0.0638 | 0.6968 | -0.0010 | 0.9931 |
| 0 | 5 | boosted_relative | 12425 | 0.5042 | 0.5286 | 0.5326 | 0.5326 | 0.5326 | NA | 0.6951 | -0.0380 | 1.0000 |
| 0 | 5 | price_lstm | 12425 | 0.5238 | 0.5395 | 0.5326 | 0.5326 | 0.5326 | NA | 0.6951 | -0.0573 | 1.0000 |
| 1 | 1 | market_logistic | 11925 | 0.5000 | 0.5000 | 0.4868 | 0.4813 | 0.5187 | -0.0378 | 0.3951 | NA | 0.3672 |
| 1 | 1 | stock_logistic | 11925 | 0.5082 | 0.5119 | 0.4855 | 0.4813 | 0.5187 | -0.0387 | 0.4054 | 0.0232 | 0.3839 |
| 1 | 1 | boosted_binary | 11925 | 0.5035 | 0.4971 | 0.5194 | 0.4813 | 0.5187 | 0.0140 | 0.1543 | 0.0260 | 0.0870 |
| 1 | 1 | boosted_relative | 11925 | 0.4957 | 0.5316 | 0.4932 | 0.4813 | 0.5187 | 0.0024 | 0.5744 | 0.0476 | 0.7094 |
| 1 | 1 | price_lstm | 11925 | 0.4848 | 0.4928 | 0.4823 | 0.4813 | 0.5187 | -0.0083 | 0.6269 | 0.0230 | 0.9062 |
| 1 | 5 | market_logistic | 11925 | 0.5000 | 0.5000 | 0.4993 | 0.4884 | 0.5116 | -0.0348 | 0.1402 | -0.1135 | 0.0940 |
| 1 | 5 | stock_logistic | 11925 | 0.4777 | 0.5266 | 0.4801 | 0.4884 | 0.5116 | -0.0634 | 0.2687 | 0.0407 | 0.2226 |
| 1 | 5 | boosted_binary | 11925 | 0.5029 | 0.5228 | 0.5052 | 0.4884 | 0.5116 | -0.0266 | 0.0685 | 0.0631 | 0.0428 |
| 1 | 5 | boosted_relative | 11925 | 0.4843 | 0.5085 | 0.5091 | 0.4884 | 0.5116 | 0.0086 | 0.3550 | 0.0360 | 0.2727 |
| 1 | 5 | price_lstm | 11925 | 0.5063 | 0.4734 | 0.5116 | 0.4884 | 0.5116 | NA | 0.0000 | 0.0578 | 0.0000 |
| 0 | 1 | relative_45 | 12433 | 0.4937 | 0.5098 | 0.5210 | 0.5203 | 0.5203 | 0.0121 | 0.6762 | -0.0203 | 0.9591 |
| 1 | 1 | relative_45 | 11925 | 0.5036 | 0.5172 | 0.4804 | 0.4813 | 0.5187 | -0.0222 | 0.5864 | 0.0328 | 0.7749 |
| 2 | 1 | relative_45 | 11615 | 0.5354 | 0.4939 | 0.4833 | 0.4839 | 0.5161 | -0.0095 | 0.0427 | -0.0125 | 0.0238 |
| 2 | 1 | boosted_relative | 11615 | 0.5410 | 0.4949 | 0.4831 | 0.4839 | 0.5161 | -0.0123 | 0.1207 | -0.0185 | 0.0718 |
| 2 | 1 | stock_logistic | 11615 | 0.5283 | 0.5022 | 0.5062 | 0.4839 | 0.5161 | 0.0313 | 0.3658 | 0.0031 | 0.2626 |
| 2 | 1 | price_lstm | 11615 | 0.5449 | 0.5131 | 0.5066 | 0.4839 | 0.5161 | 0.0194 | 0.4660 | 0.0156 | 0.4080 |

## 4. Selective accuracy

For the exploratory relative-return candidate, each nominal level is a frozen calibration quantile of absolute distance from 0.5. The primary operating point was predeclared at nominal 20%. Realized coverage may shift. Intervals in the diagnostics resample contiguous 10-session blocks across all assessment dates. Very few selected dates make even block intervals unreliable. Down accuracy is undefined where there are no Down calls.

| Window | Horizon | Nominal coverage | Actual coverage | Rows | Dates | Accuracy | Same-subset majority | Up share | Up accuracy | Down accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.05 | 0.0699 | 869 | 112 | 0.6041 | 0.6122 | 0.9850 | 0.6098 | 0.2308 |
| 0 | 1 | 0.1 | 0.1275 | 1585 | 119 | 0.5527 | 0.5577 | 0.9899 | 0.5558 | 0.2500 |
| 0 | 1 | 0.2 | 0.2326 | 2892 | 122 | 0.5557 | 0.5598 | 0.9931 | 0.5581 | 0.2000 |
| 0 | 1 | 0.5 | 0.5352 | 6654 | 122 | 0.5424 | 0.5443 | 0.9968 | 0.5435 | 0.1905 |
| 0 | 5 | 0.05 | 0.0555 | 689 | 122 | 0.5327 | 0.5327 | 1.0000 | 0.5327 | NA |
| 0 | 5 | 0.1 | 0.1440 | 1789 | 122 | 0.5690 | 0.5690 | 1.0000 | 0.5690 | NA |
| 0 | 5 | 0.2 | 0.2554 | 3173 | 122 | 0.5645 | 0.5645 | 1.0000 | 0.5645 | NA |
| 0 | 5 | 0.5 | 0.5088 | 6322 | 122 | 0.5536 | 0.5536 | 1.0000 | 0.5536 | NA |
| 1 | 1 | 0.05 | 0.0195 | 233 | 91 | 0.4979 | 0.5107 | 0.6266 | 0.5068 | 0.4828 |
| 1 | 1 | 0.1 | 0.0563 | 671 | 106 | 0.4903 | 0.5261 | 0.3204 | 0.5256 | 0.4737 |
| 1 | 1 | 0.2 | 0.1244 | 1484 | 112 | 0.4643 | 0.5384 | 0.2776 | 0.5049 | 0.4487 |
| 1 | 1 | 0.5 | 0.4002 | 4772 | 117 | 0.5044 | 0.5159 | 0.5748 | 0.5177 | 0.4864 |
| 1 | 5 | 0.05 | 0.0075 | 90 | 44 | 0.3222 | 0.5556 | 0.3889 | 0.3429 | 0.3091 |
| 1 | 5 | 0.1 | 0.0297 | 354 | 94 | 0.3757 | 0.5508 | 0.2712 | 0.3646 | 0.3798 |
| 1 | 5 | 0.2 | 0.0868 | 1035 | 111 | 0.4618 | 0.5150 | 0.1662 | 0.4302 | 0.4681 |
| 1 | 5 | 0.5 | 0.3647 | 4349 | 117 | 0.4997 | 0.5079 | 0.1725 | 0.4760 | 0.5046 |
| 2 | 1 | 0.05 | 0.0297 | 345 | 111 | 0.5043 | 0.5159 | 0.0290 | 0.3000 | 0.5104 |
| 2 | 1 | 0.1 | 0.0993 | 1153 | 115 | 0.4788 | 0.5152 | 0.0130 | 0.2667 | 0.4815 |
| 2 | 1 | 0.2 | 0.1663 | 1932 | 115 | 0.4907 | 0.5067 | 0.0140 | 0.4074 | 0.4919 |
| 2 | 1 | 0.5 | 0.3250 | 3775 | 115 | 0.4845 | 0.5147 | 0.0135 | 0.4706 | 0.4847 |

Nominal 20% coverage by ticker, sector, and trailing market regime is in [subgroup coverage](selective_20pct_subgroups.csv). Exact thresholds, Up and Down counts, and intervals are in [diagnostics.json](diagnostics.json). All models and all coverage levels are in the fit ledgers.

Four nominal 5% slices reached 70% retrospectively. W0 5d boosted binary reached 81.76% on 488 rows from ten dates (3.93% coverage), W0 5d market logistic reached 77.06% on 510 rows from five dates (4.10%), and W0 5d stock logistic reached 73.53% on 801 rows from 22 dates (6.45%). All three predicted Up on every selected row. W1 5d market logistic reached 78.43% on 102 rows from **one date** (0.86%), all Down. Every slice exactly matched its own selected-subset majority. None represents a reproducible 70% stock-specific signal. The predeclared 20% operating point for the relative regressor fell below its own subset majority in the three 1d windows and tied or trailed it at 5d.

## 5. Ranking, trading relevance, and next decision

At 1d the relative model's assessment AUC was **0.5150, 0.5316, 0.4949**, versus stock logistic **0.5103, 0.5119, 0.5022**. The first two matched gains reversed on W2. All paired 10-day block intervals against stock logistic include zero. The 45-round variant scored **0.5098, 0.5172, 0.4939**, so shallower training by round count did not improve the conclusion. At 5d, relative regression beat stock logistic in W0 and lost in W1. The Price Only LSTM achieved the highest individual AUC (0.5395, W0 5d) and then fell to 0.4734 in W1. The binary tree's 53.75% accuracy peak narrowly beat the 53.26% majority baseline for that window and was not a generalizable 70% result.

The 90-round relative regressor is the strongest retrospective 1d ranking *candidate* by mean AUC on the three windows, not an independently validated replacement. Keep stock logistic and the existing Price Only LSTM as controls. There is no basis to promote a selective policy for practical use. Single-direction high-confidence slices can result from selecting a favorable market day, without ranking stocks within that day. The data show regime sensitivity and weak calibration transfer. No portfolio backtest was run. Executable next-prices and corporate-action dispositions would be required for a credible net-return evaluation, along with stable prospective ranking evidence.

**Recommendation:** stop tuning this historical archive. Secure licensed or otherwise authorized, complete cohort quotes and documented corporate-action treatments for 2024–2026, then use 2024–2025 for selection and evaluate a genuinely unused period once under a frozen protocol. Do not replace the published model from these results.

## 6. Reproducibility and tests

[Initial fit ledger](runs.jsonl) and [follow-up ledger](followup_runs.jsonl) record split dates, parameters, per-run metrics, LSTM histories and data hashes. [Initial predictions](predictions) and [follow-up predictions](followup_predictions) hold separate Zstandard Parquet files for all 26 fits. The scripts are [study.py](study.py) and [summarize.py](summarize.py). The initial runner hash differs from the current file because follow-up CLI support and lint fixes were added after the initial runs. No shared modelling source or official artifact was modified. No dependencies were installed or Git operations performed.

**46 targeted tests passed** and lint passed. The tests cover future-data invariance, market-membership filtering, exact labels, splits, and threshold-based abstention accounting. The public HTTP probe produced no usable price artifact. The first in-memory audit had an index-alignment error for 5d, which was corrected before recording the diagnostics. No model fits were lost or failed.
