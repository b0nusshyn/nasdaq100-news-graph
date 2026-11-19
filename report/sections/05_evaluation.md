# Evaluation protocol

> Owner: Evaluation · Issue #50 · Due Fri 20 Nov

<!-- Baseline ladder, walk-forward with embargo, metrics, backtest with costs, significance test. -->

## Walk-forward evaluation

The comparison uses five rolling folds shared by every model and both prediction
horizons. Each fold contains 504 outer training sessions, a five-session embargo,
and 126 test sessions. The window advances by 126 sessions at each step. For
each test date, the eligible universe contains stocks that were NASDAQ-100 members
on that date and have an observed label for the given horizon. The positive class
is a strictly positive close-to-close return over the next one or five trading
sessions. Predictions must match the eligible date–ticker keys and realized
labels in every fold and horizon.
[Comparison manifest](../../results/comparison_with_classical/tables/comparison_manifest.json)

The baseline ladder starts with the Buy-and-Hold reference, followed by ARIMA,
Logistic Regression, XGBoost, LSTM Price Only (Baseline 5), and LSTM + News,
No Graph (Baseline 6). Baseline 6 shares the proposed GAT-LSTM's encoder and
news inputs and provides the closest comparison for graph propagation.

## Classification metrics

Accuracy is the fraction of correct classifications. Positive-class F1 is the
harmonic mean of precision and recall for upward moves. Matthews correlation
coefficient (MCC) measures agreement between predicted and observed labels using
all four confusion-matrix counts. It is undefined when a marginal count
makes the denominator zero. For each fitted model, a fold predicts up when
the probability meets or exceeds the cutoff selected to maximize MCC on
internal validation predictions. The Buy-and-Hold prediction reference uses
a constant score and always predicts up.

ROC AUC measures the ranking of positive versus negative outcomes using
predicted probabilities. Fold ROC AUC ranks all observations in the test fold.
The headline metric, **within-day AUC**, computes ROC AUC separately across
stocks on each date and averages over dates with both classes. It measures
stock ranking within a session. The summary gives each fold equal weight for
all classification metrics. A metric undefined in any fold is reported as
N/A in the summary.
[Fold scores and thresholds](../../results/comparison_with_classical/tables/metrics_by_fold.csv)

## Portfolio backtest

At entry, strategies rank eligible stocks by predicted probability, take
equal-weight long positions in the top 20% and equal-weight short positions in
the bottom 20%, and allocate half the capital to each side. Positions are held
for one or five trading sessions, with non-overlapping holding periods and a
transaction cost of 5 basis points per side. Execution at the signal day's
close is idealized. The simulation omits slippage, borrow fees, liquidity
constraints, and assumed recoveries for delistings or other unresolved
corporate actions.

The main portfolio statistics use one continuous run per model and horizon,
with wealth carried forward and a single entry schedule across fold boundaries.
Per-fold diagnostics restart the portfolio and entry schedule in each test fold.
A scheduled exit can occur after the last signal of the fold. Net Sharpe
annualizes mean net holding-period return divided by its sample standard
deviation, using 252 trading sessions per year and adjusting for holding-period
length.

Buy-and-Hold is evaluated as a separate long-only portfolio. The basket starts
with equal allocations to the historical members at inception, retains the
remaining positions, and holds proceeds from removals in cash. Later entrants
are excluded. The calculation requires observed prices to value positions
through removal or exit. Unresolved valuations make the affected benchmark
windows unavailable.
[Long-only benchmark records](../../results/comparison_with_classical/tables/buy_and_hold_benchmark.csv)

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

## Retrospective validation tuning

Task 4.13 evaluated six candidates (T0–T5) at seed 42 for each LSTM baseline.
For each fold, weight fitting used the initial portion of the outer training
window and validation used its final 10%, separated by a five-session internal
embargo. Checkpoint selection and early stopping used validation within-day AUC.
Validation MCC used the cutoff chosen on the validation observations. Each
candidate required valid within-day AUC for all ten fold–horizon combinations.
Selection maximized the unweighted mean of those ten AUCs, with exact ties
resolved in T0–T5 order. Each baseline retained one configuration for both
horizons.

The study used validation predictions and labels for selection. Later validation
windows overlap earlier test periods, so selection is retrospective. Shared
preprocessing, including PCA where applicable, was fitted on the outer training
fold, including internal validation. Reusing validation observations to select
checkpoints, thresholds and configurations can make validation scores optimistic.
[Task 4.13 study protocol](../../results/tables/tuning_baselines.md)
