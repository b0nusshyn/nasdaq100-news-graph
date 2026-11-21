# Offline improvement study — absolute positive return

## Executive summary

Completed 16 bounded model fits on the original strictly-positive 1d and 5d direction targets. A date-grouped XGBoost ranker was tested against logistic regression and histogram boosting, then ablated with lagged-beta market residual features. One 1d window favored the ranker by 0.0245 AUC against histogram boosting, with uncertainty spanning zero. The other 1d window showed only +0.0023, and both 5d gains were below 0.002. Every full-population accuracy was below the matched majority. No credible improvement or deployable winner was established. Both nominal 0.63 goals were missed, and these windows provide retrospective development evidence only. [All metrics and regimes](tables.md)

## Prior evidence and limiting factors

| Prior study | Observed result and implication |
|---|---|
| [Official GAT-LSTM and Price Only](../../report/sections/06_results.md) and [ablations](../../results/tables/ablations_3_9.md) | Official GAT AUC 0.5094 and 0.4992, Price Only 0.4995 and 0.4811. The historical graph ladder showed a small 1d gain across seeds, with overlapping folds. Existing tuning and ablations give little evidence for another architecture search. |
| [Neural tuning](../gat_lstm_tuning/run.py) and [edge experiments](../gat_edge_weight_v1/report.md) | The prior neural pairwise arm had a large training-to-assessment gap. The original attention already learns from edge weights. Signed message weighting and centering showed no repeatable benefit. The 5d original attention fell from 0.5404 to 0.4363 across windows at seed 43, indicating transfer instability. |
| [FNSPID and FinBERT](../fnspid_candidate_audit/experiment_report_v1.md) | Recovery and encoding succeeded, but original publication and arrival timestamps were unverified. Proxy news raised fit AUC with weak assessment transfer and both news arms lost to Price Only at 5d. Limited and poorly timed event information remains a plausible limit, without a verified incremental news signal. |
| [Direction studies](../market_direction_v1/report.md) and [restricted cohort](../post2023_validation_v1/report.md) | Historical logistic, boosted controls, relative-return regressors and LSTMs showed weak or unstable gains. Forward market direction strongly changes stock class proportions. Daily mean official scores explained 62% and 68% of score variance at 1d and 5d yet had negative correlations with realized daily Up rates. This combination can hurt accuracy and offers weak cross-sectional discrimination. |
| [Outperformance](../cohort_outperformance_v1/report.md) and [decision collapse](../boosted_decision_v1/report.md) | The 1d boosted binary outperformance model had raw AUC 0.5284 and 0.5273 on a different target. Earlier Platt maps compressed decisions, and the 2024 map inverted score order. No class-mapping defect was found. Rank quotas improved retrospective decisions weakly, with no stable selective advantage. These results do not transfer as evidence of absolute direction. |
| [Historical price reconciliation](../final_2026_reconciliation/report.md) | Missing historical issues, inconsistent vendor volume and corporate-action coverage block the frozen 101-security 2026 assessment. The pinned restricted source can support matched development comparisons, with revised adjustment vintages and liquidity selection limiting inference. |

The dominant measured issue in this cycle is the gap between ranking fit and assessment. It is consistent with overfitting or unrepeated patterns under regime shifts. It does not establish that the direction target is intrinsically unpredictable. The BCE versus within-date ranking mismatch motivated the new objective. Switching the learner and objective together prevents attribution to loss alone. Past studies already covered market-relative momentum, rolling beta, volatility, breadth, dispersion and simple trend features. Rolling beta applied as a lagged hedge to each daily stock return, then aggregated and ranked, was the distinct feature hypothesis here.

## Data and feature availability

The study used the audited pinned public Yahoo Finance archive as the only bar source, via the existing pre-2026 30-security restricted loader. The 30 names come from the December 2023 anchor selected by prior notional turnover. Each exchange session has all 30 bars in the data read. No external prices were spliced. The target is close(t+h)/close(t)−1 > 0 with adjusted prices and exact future Nasdaq session endpoints. No eligible assessment rows were removed by return sign or model. The archive has retrospective adjustment values, with point-in-time adjustment vintages unverified. The restricted cohort favors large liquid names and cannot represent historical membership of the full index.

New features are the trailing five- and 20-session sums of each security's one-day log return less its estimated market beta times the fixed cohort's same-day average log return, and the same-session percentile ranks of both sums. Beta uses 20 trailing returns and is shifted one session. The return being residualized is available through the signal close. Prefix-invariance tests perturb future prices and confirm identical earlier feature rows. All assessment rows have sufficient historical warmup. The exact baseline feature list and four new fields are in the manifest.

SEC submissions and company facts are promising for fundamentals (including as-filed revenue, margin and leverage), but a company-facts period end or latest frame value does not establish public availability at that date. This cycle did not obtain validated historical CIK-to-issue mappings, accession-level fact/acceptance joins and original earnings release timestamps or historic analyst expectations. Earnings drift and verified announcement timing remain untested. No SEC, event or analyst field was invented or used. Sector attribution requires historical classifications, which the prior repository audit could not verify from its static map. Earnings, static sector and extreme-move error slicing were excluded from this bounded experiment. Trend and volatility diagnostics used only already available historical inputs. [SEC EDGAR API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

## Models and protocol

Each fit used 252 sessions, and the cutoff used 50 strictly earlier sessions. The 2024 fit was March 3, 2023 to March 4, 2024, cutoff April 24 to July 5, 2024, and assessment July 15 to December 20, 2024 (113 dates and 3,390 rows). The 2025 fit was March 1, 2024 to March 4, 2025, cutoff April 24 to July 7, 2025, and assessment July 15 to December 19, 2025 (112 dates and 3,360 rows). All split boundaries passed the five-session horizon check.

- Logistic used median imputation, fit-only standardization and L2 C=.25.
- Histogram boosted binary used 90 iterations, learning rate .05, 7 leaves, 150 minimum rows per leaf and L2=8.
- Pairwise XGBoost used binary relevance from the same unchanged label and ordered query IDs grouping all observations within each date, 90 trees, depth 2, learning rate .05 and L2=8. The protocol's shorthand about all pairs denotes a 30-pair-per-sample budget using the library's mean sampler. It does not guarantee exhaustive distinct pair enumeration.
- Pairwise plus residuals uses the identical learner and adds only the four residual features.
- Each arm used an earlier balanced-accuracy cutoff over 19 raw-score quantiles. These scores are not calibrated probabilities. Seed 42 was fixed; the budget prioritized matched arms, two date ranges and both horizons over seed replication. GAT and regime-specific fits were excluded given prior instability and budget. [Ranking documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)

The first attempted rank fit stopped on the inherited object-typed breadth column during XGBoost matrix creation. The two already fitted controls were retained. Casting feature columns to float fixed the adapter defect. The failed rank invocation fitted no trees. All 16 planned fits completed, with one fit budgeted for each arm/period/horizon combination and no tuning after assessment.

## Evaluation, uncertainty and selection

| Horizon | Best within-day AUC | Best full-population accuracy |
|---|---|---|
| 1d | 0.5224, 2025 pairwise | 52.59%, same fit, majority 52.77% |
| 5d | 0.5149, 2024 logistic | 53.81%, 2025 logistic, majority 54.46% |

Paired 95% intervals use 2,500 circular blocks of 10 dates to respect within-date dependence and overlapping 5d outcomes. Contrasts are descriptive after previous study access. The 1d ranker beat boosted binary by +0.0023 [−0.0271, +0.0317] and +0.0245 [−0.0032, +0.0507] in 2024 and 2025. Its fit AUC was 0.7665 and 0.7572. Adding residual features gave +0.0055 and −0.0112 at 1d and −0.0015 and −0.0027 at 5d. No ablation interval excluded zero. 2025 1d ranker performed better in rising and high-volatility subgroups, but those patterns did not reproduce in 2024. In the 2025 falling 5d subgroup its 71.82% accuracy was below the subgroup majority of 72.27%, with AUC 0.4887 and only 22 dates. That subgroup does not satisfy the full-population goal. No regime model was justified.

Retain the logistic and histogram controls. The base date-grouped 1d ranker is a research candidate only, with the precise configuration above and without the four residual features. It did not demonstrate stable accuracy, cross-period ranking superiority over the strongest matched model, or cross-seed robustness. No optional alternative targets were fit. Stop tuning these windows. A future study requires genuinely distinct point-in-time information and a qualified unseen period. The 101-security 2026 lock and live workflow remain untouched.

## Files, tests and reproduction

Only this isolated directory was written. It contains the frozen protocol, `study.py`, `summarize.py`, `test_offline_rank.py`, `execution.log`, `tables.md`, this report and `EXPERIMENT_STATUS.md`. The `runs/` directory contains the original and resume manifests, actual resumed source snapshot, 16-fit ledger, 16 fitted estimators and 16 prediction Parquets. Matched populations and binary labels were checked across arms, and accuracy and AUC were replayed from all saved predictions. Forty-five tests passed, including grouping, prefix invariance, label alignment and split checks. Ruff passed. Disk stayed above the safety margin. No Git, paid services, official modifications or live operations.

To reproduce, use `.venv/bin/python -B -m experiments.offline_rank_v1.study` with a fresh sibling experiment copy and no existing `runs/`, then run its `summarize` module. The resume manifest pins the effective execution source. The current runner differs from the resumed snapshot only in manifest-write formatting. The 16-fit budget is exhausted.
