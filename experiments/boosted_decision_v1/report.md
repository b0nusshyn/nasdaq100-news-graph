# Boosted classifier decision replay

## Finding

The one-class output came from compressed probabilities and a calibration-period base-rate shift. The saved classifier scores, probability column and target encoding replayed correctly. A daily rank quota restored useful direction variation and modestly beat the matched majority point estimate in both windows, but neither date-block interval for that advantage excluded zero. Changing the global cutoff or calibration alone was weaker. No confidence rule showed a reproducible selective advantage.

## Diagnosis

The saved boosted estimator and calibrator both exposed classes [0, 1]. The positive column matched the sigmoid of the raw classifier score, and replayed assessment keys, labels, raw scores and calibrated probabilities matched the saved artifacts. The label was unchanged and denotes strictly greater adjusted return than the same-period equal-weight cohort. There was no evidence of a class swap, feature/label alignment error, column-selection bug or cutoff implementation defect.

The fitted Platt map operates on standardized margins. Its slope was −0.01327 in 2024 and +0.00486 in 2025. The intercepts were −0.09875 and +0.04534, tracking earlier positive prevalences of 47.53% and 51.13%. The 2024 assessment probability range was 0.46565–0.49652, so every call at 0.5 was negative. The 2025 range was 0.50698–0.51463, so every call was positive even though only 47.95% of the assessment rows outperformed. Median daily probability spread was 0.01450 and 0.00344. In 2024 the negative slope also reversed score ranking exactly, from raw AUC 0.5284 to calibrated AUC 0.4716. In 2025 it preserved the raw AUC of 0.5273.

The 50 calibration dates supplied 1,500 rows per year, but same-session stocks are dependent and the effective sample is smaller. Raw AUC on the earlier 25 dates was 0.5471 and 0.5306, then fell to 0.5074 and 0.4952 on calibration. Platt logistic fits pooled probability likelihood, which need not agree with equal-weight daily AUC. These results show limited and shifting score-label signal across periods. They do not identify a unique minimum calibration sample size. The raw scores still ranked stocks above chance retrospectively, although their probability mapping offered limited value.

## Frozen interventions

The [protocol](protocol.md) fixed four rules. The default uses saved Platt probabilities at 0.5. The earlier cutoff maximizes balanced accuracy among 19 calibration probability quantiles and uses the same saved probabilities. Its cutoffs were 0.47030 and 0.50966. Increasing isotonic calibration pools 25 earlier dates and the original 50 calibration dates, permitted because the boosted estimator did not use the earlier dates for checkpoint selection. The daily rank quota uses raw score order and earlier observed prevalence, rounding to 14 of 30 positive calls in 2024 and 15 of 30 in 2025. The latter rounding does not assume half the cohort outperforms. Observed daily outperformance ranged from 20% to 70% in 2024 and 30% to 70% in 2025. Ties use ticker order. All rules use the same 3,390 and 3,360 stock-days over 113 and 112 sessions. Isotonic required two post-fit calibration fits. No boosted model or feature transformer was retrained.

## Full-coverage results

AUC uses the continuous score associated with each rule, independent of the cutoff. The daily rank score nearly reproduces raw AUC, with tiny differences from deterministic score tie resolution. Precision is reported as zero if a rule issues no positive calls.

| Year | Rule | Accuracy | Majority | MCC | Balanced accuracy | Precision | Recall | F1 | AUC | Positive / negative |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | default | 0.5018 | 0.5018 | — | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.4716 | 0 / 3390 |
| 2024 | threshold | 0.4997 | 0.5018 | 0.0048 | 0.5012 | 0.4989 | 0.9307 | 0.6496 | 0.4716 | 3151 / 239 |
| 2024 | isotonic | 0.5021 | 0.5018 | 0.0026 | 0.5011 | 0.5007 | 0.2208 | 0.3065 | 0.5179 | 745 / 2645 |
| 2024 | rank_call | 0.5218 | 0.5018 | 0.0435 | 0.5217 | 0.5215 | 0.4885 | 0.5044 | 0.5284 | 1582 / 1808 |
| 2025 | default | 0.4795 | 0.5205 | — | 0.5000 | 0.4795 | 1.0000 | 0.6482 | 0.5273 | 3360 / 0 |
| 2025 | threshold | 0.5036 | 0.5205 | 0.0350 | 0.5147 | 0.4890 | 0.7865 | 0.6030 | 0.5273 | 2591 / 769 |
| 2025 | isotonic | 0.5188 | 0.5205 | 0.0346 | 0.5173 | 0.4981 | 0.4817 | 0.4897 | 0.5242 | 1558 / 1802 |
| 2025 | rank_call | 0.5259 | 0.5205 | 0.0518 | 0.5259 | 0.5054 | 0.5270 | 0.5160 | 0.5274 | 1680 / 1680 |

### Calibration quality

| Year | Platt Brier | Isotonic Brier | Prior-only Brier | Platt log loss | Isotonic log loss |
|---|---:|---:|---:|---:|---:|
| 2024 | 0.25068 | 0.25132 | 0.25052 | 0.69450 | 0.77512 |
| 2025 | 0.25050 | 0.24933 | 0.25059 | 0.69414 | 0.69173 |

The threshold change recovered both classes, but its accuracy stayed below the matched majority in both periods. It improved accuracy in 2025 and reduced it slightly in 2024. Isotonic recovered both classes and improved point accuracy in both periods, with a negligible advantage over majority in 2024 and a shortfall in 2025. It improved Brier and log loss in 2025 and worsened both in 2024. There is no consistent probability-calibration repair. The rank rule improved accuracy and balanced accuracy in both periods. Its accuracy advantage over the selected-population majority was +2.01 and +0.54 percentage points. The descriptive paired 95% block intervals were approximately [−0.47, +3.48] and [−1.43, +2.38] percentage points. The position quota is a decision convention and does not supply probability estimates.

## Confidence selection

The original 20% nominal cutoff selected 23.36% in 2024 and 7.56% in 2025. No assessment margin tied any frozen cutoff at 5%, 10%, 20% or 50%. The 2025 score distribution shifted and narrowed inside an already compressed, all-positive probability band. Absolute distance from 0.5 selected only the high-score tail there, while in 2024 it selected only negative calls. The rank alternative controls the daily quota without labeling a rank as confidence probability. With 30 names, nearest-integer rounding yields two selections per day at nominal 5%. Asymmetric quotas sometimes select only one class at the smallest cut.

| Year | Nominal | Method | Actual | Rows | Dates | Positive / negative | Accuracy [95% interval] | Subset majority |
|---|---:|---|---:|---:|---:|---|---|---:|
| 2024 | 5% | original | 7.05% | 239 | 77 | 0 / 239 | 51.05% [43.69%, 59.64%] | 51.05% |
| 2024 | 5% | rank | 6.67% | 226 | 113 | 0 / 226 | 54.42% [47.79%, 61.50%] | 54.42% |
| 2024 | 10% | original | 11.95% | 405 | 98 | 0 / 405 | 50.37% [45.22%, 56.38%] | 50.37% |
| 2024 | 10% | rank | 10.00% | 339 | 113 | 61 / 278 | 51.92% [45.72%, 58.70%] | 53.39% |
| 2024 | 20% | original | 23.36% | 792 | 112 | 0 / 792 | 49.24% [45.59%, 53.64%] | 50.76% |
| 2024 | 20% | rank | 20.00% | 678 | 113 | 226 / 452 | 52.65% [48.82%, 56.93%] | 53.83% |
| 2024 | 50% | original | 47.26% | 1602 | 113 | 0 / 1602 | 47.82% [45.36%, 50.37%] | 52.18% |
| 2024 | 50% | rank | 50.00% | 1695 | 113 | 729 / 966 | 53.33% [50.97%, 55.81%] | 53.04% |
| 2024 | 100% | original | 100.00% | 3390 | 113 | 0 / 3390 | 50.18% [48.70%, 51.67%] | 50.18% |
| 2024 | 100% | rank | 100.00% | 3390 | 113 | 1582 / 1808 | 52.18% [50.44%, 53.94%] | 50.18% |
| 2025 | 5% | original | 1.25% | 42 | 22 | 42 / 0 | 47.62% [22.45%, 70.23%] | 52.38% |
| 2025 | 5% | rank | 6.67% | 224 | 112 | 112 / 112 | 47.77% [42.62%, 53.57%] | 54.91% |
| 2025 | 10% | original | 2.11% | 71 | 32 | 71 / 0 | 47.89% [33.33%, 63.58%] | 52.11% |
| 2025 | 10% | rank | 10.00% | 336 | 112 | 169 / 167 | 50.30% [45.83%, 55.06%] | 57.14% |
| 2025 | 20% | original | 7.56% | 254 | 67 | 254 / 0 | 46.85% [40.80%, 53.69%] | 53.15% |
| 2025 | 20% | rank | 20.00% | 672 | 112 | 336 / 336 | 51.79% [48.51%, 54.91%] | 53.87% |
| 2025 | 50% | original | 34.82% | 1170 | 91 | 1170 / 0 | 50.94% [48.43%, 53.28%] | 50.94% |
| 2025 | 50% | rank | 50.00% | 1680 | 112 | 837 / 843 | 52.08% [50.27%, 53.81%] | 52.74% |
| 2025 | 100% | original | 100.00% | 3360 | 112 | 3360 / 0 | 47.95% [46.40%, 49.43%] | 52.05% |
| 2025 | 100% | rank | 100.00% | 3360 | 112 | 1680 / 1680 | 52.59% [51.43%, 53.81%] | 52.05% |

At 20% actual rank coverage, accuracy improved over the original rule's differently selected rows but still lost to its own matched subset majority in both windows. The sole positive selective point difference was 2024 at 50% rank coverage, +0.29 percentage points with an interval spanning zero. No point estimate reached 70% in this replay. Reported intervals are descriptive 10-session circular date-block bootstrap intervals. They account for shared daily outcomes better than row-wise resampling and should not be treated as independent confirmation after prior inspection of these windows.

## Decision and files

Keep the raw score plus the prior-prevalence daily quota as a **candidate** decision rule and log both outputs separately. The saved probability calibrator should remain a documented reference. The data support a decision-layer change that prevents the observed collapse, but the measured lift is too uncertain to justify a new predictor training sweep or deployment. For independent validation, secure a comparable chronological dataset covering the frozen 101 identities through their last trading sessions, with venue, adjustment and corporate-action records. Pass the full readiness gate, freeze the selected candidate and its earlier prevalence procedure, then score the unused period once. The existing 2026 lock remained in force.

New files are this report, `protocol.md`, `replay.py`, `test_replay.py`, and the isolated `runs/` directory containing daily score summaries, decision Parquets, two fitted isotonic maps, result and hash JSONs, and the executed runner snapshot. The final runner differs from the snapshot only in import formatting. To reproduce from the repository root, run `.venv/bin/python -B -m experiments.boosted_decision_v1.replay` in a fresh copy with no `runs/` directory. Seven focused and existing tests passed. Ruff passed. No shared code, earlier models or artifacts were overwritten. No Git operations were used.
