# Evaluation protocol

> Owner: Evaluation · Issue #50 · Due Fri 20 Nov

<!-- Baseline ladder, walk-forward with embargo, metrics, backtest with costs, significance test. -->

## Proposed model versus Baseline 6

Task 4.9 compares GAT-LSTM (`gat_lstm`) with LSTM + News, No Graph (`lstm_news`)
using MCC, separately for the 1-day and 5-day horizons. This is a separate endpoint
from the modelling reports' headline within-day AUC. The comparison uses the official
tuned T1 runs at training seed 42. The numerical results, configurations, source
artifact hashes and training commit are preserved in
[`significance_mcc.json`](../../results/tables/significance_mcc.json).

### Paired observations and operating thresholds

Predictions are aligned by fold, horizon, date and ticker. Both models must have
exactly the same evaluation keys and realised labels, covering all five folds for
each horizon. Missing or duplicate keys, invalid probabilities or binary labels,
and incomplete run records cause an error; unmatched observations are not silently
removed. Each model's MCC is computed with the shared Evaluation metric implementation
at its own validation-selected threshold from its associated run record. Test outcomes
are not used to select or adjust thresholds.

For each fold, the contrast is GAT-LSTM MCC minus Baseline 6 MCC. The reported effect
is the unweighted mean of the five paired fold differences, rather than pooled MCC
or a mean weighted by the number of observations. Positive differences favour
GAT-LSTM on this endpoint.

### Significance test and confidence intervals

The primary test is an exact two-sided paired sign-flip test of the mean fold
contrast. It enumerates all 32 sign assignments, retaining the pairing between
models within every fold. Its exact interpretation is conditional on joint
invariance under independent fold sign flips under the null; equal mean MCC alone
does not guarantee that assumption.

A secondary 95% percentile bootstrap interval resamples whole paired folds. All
3,125 ordered samples are enumerated, with endpoints calculated using the empirical
inverse CDF (`inverted_cdf`). Individual ticker-day rows are not treated as independent
resampling units. Both calculations are deterministic and require no random sampling
seed. These intervals are descriptive and are not simultaneous confidence intervals
across horizons.

Holm adjustment is applied to the two raw horizon p-values, with family-wise alpha
0.05. With five folds, the smallest attainable two-sided sign-flip p-value is
0.0625, so this design cannot reject at 5% even before adjustment. The analysis
therefore has limited power to distinguish the models.

### Assumptions and scope

Chronological folds are not necessarily independent replications. Overlapping
training windows, market dependence and overlapping five-day outcomes can undermine
the sign-invariance assumption. The bootstrap interval can also have poor coverage
with five potentially dependent folds and must not override the primary test.
Non-rejection does not establish equivalence or the absence of a graph effect.

Conclusions apply to the recorded T1, seed-42 runs, not to stability across training
seeds or to the within-day AUC endpoint. Prediction and run records alone cannot
independently prove causal feature preparation, historical membership or validation
threshold provenance. The source training-input dataset hashes were not recorded;
source prediction and run-record hashes are retained in the result artifact.
