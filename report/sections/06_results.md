# Results

> Owner: Evaluation · Issue #50 · Due Fri 20 Nov

<!-- Main table, proposed vs Baseline 6 significance, ablations (#41), equity curves. -->

## Proposed model versus Baseline 6

The following are production results for the official tuned T1 runs at seed 42,
from [`significance_mcc.json`](../../results/tables/significance_mcc.json). Both runs
record training commit `dd811f1ca50ce748ef542fa5d1e9b8cd2247cd03`. The comparison uses
exactly matched evaluation keys and realised labels across all five folds for each
horizon. MCC uses each model's own recorded validation-selected threshold. The
statistical procedure and its assumptions are described in the Evaluation protocol.

### Per-fold MCC

Differences are GAT-LSTM minus Baseline 6. MCC values and differences are rounded to
six decimal places from the full-precision artifact.

| Horizon | Fold | Matched observations | GAT-LSTM MCC | Baseline 6 MCC | Paired difference |
|---|---:|---:|---:|---:|---:|
| 1d | 0 | 12,852 | 0.025179 | 0.029907 | -0.004728 |
| 1d | 1 | 12,837 | -0.005045 | -0.013214 | +0.008170 |
| 1d | 2 | 12,829 | 0.005378 | -0.000814 | +0.006193 |
| 1d | 3 | 12,838 | -0.022290 | -0.002602 | -0.019688 |
| 1d | 4 | 12,726 | 0.009017 | 0.002496 | +0.006521 |
| 5d | 0 | 12,852 | -0.022250 | 0.013153 | -0.035403 |
| 5d | 1 | 12,829 | 0.043398 | 0.014243 | +0.029155 |
| 5d | 2 | 12,825 | -0.007209 | -0.025135 | +0.017926 |
| 5d | 3 | 12,838 | 0.005369 | -0.007835 | +0.013204 |
| 5d | 4 | 12,726 | 0.004262 | 0.003727 | +0.000535 |

### Paired inference across folds

Each mean difference gives equal weight to the five folds. The 95% intervals are
secondary paired whole-fold bootstrap intervals; p-values are from the primary
exact two-sided sign-flip test. Holm adjustment accounts for the two horizons.

| Horizon | Matched observations | Mean MCC difference | 95% bootstrap interval | Raw p-value | Holm-adjusted p-value |
|---|---:|---:|---|---:|---:|
| 1d | 64,082 | -0.000707 | [-0.011124, +0.007115] | 1.0000 | 1.0000 |
| 5d | 64,070 | +0.005083 | [-0.015960, +0.021473] | 0.6875 | 1.0000 |

### Interpretation and limitations

GAT-LSTM has a slightly lower average fold MCC at 1 day and a slightly higher one
at 5 days. Neither horizon rejects the null under the specified sign-flip procedure
at family-wise alpha 0.05; both adjusted p-values are 1.0000. Both descriptive
bootstrap intervals include zero. These findings do not establish equivalence,
prove the absence of a graph effect, or demonstrate superiority at either horizon.

Only five fold differences are available, and the minimum attainable two-sided
p-value is 0.0625. Shared training data and temporal dependence further limit the
interpretation of folds as independent evidence, particularly for overlapping
five-day outcomes. The intervals have uncertain small-sample coverage. These are
MCC results for a single training seed and configuration, not a significance test
of headline within-day AUC or a measure of variability across seeds. Original
training-input dataset hashes are unavailable, although the source prediction and
run-record hashes are preserved with the results.
