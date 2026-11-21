# GAT width, stochastic edge regularization, and chronological checkpoint selection

## Result

Completed **20 new fits** on the original positive-return labels. A smaller GAT improved the best individual five-day development observation to **55.96% accuracy, 0.547553 within-day AUC and 0.1354 MCC** on the seed-42 2022 window. The matched majority was **51.16%**. The advantage did not reproduce across seeds or periods. The worst-half checkpoint criterion materially improved one unstable five-day comparison in 2023, but remained below majority accuracy and had negative MCC. There is no configuration to promote over the guaranteed fallback. Neither 63% full-population accuracy nor 0.63 within-day AUC was reached. [All width and drop-edge results](tables.md) · [All checkpoint criterion results](../gat_checkpoint_robust_v1/tables.md) · [Best-model registry](best_model_registry.json)

## Why these three hypotheses

The prior architecture, news, calibration, weighting and ranking evidence is summarized in the [prior graph study](../persistent_graph_v1/report.md) and [submission registry report](../submission_readiness_v1/report.md). Fixed topology selection and seed averaging gave no repeatable benefit. Fit scores sometimes greatly exceeded chronological assessments, and the original five-day seed-43 checkpoint for 2023 fell from 0.5452 validation AUC to 0.4363 on the outer window. The alternative explanations tested here were latent width, sensitivity to graph neighbors during gradient updates, and an unstable mean validation objective. These are controlled changes rather than a hyperparameter sweep or new target. No verified future input leak or production defect was found.

- **Reduced width** changes 64 to 32 latent units for the LSTM and GAT while keeping four heads and the other matched settings. Earlier multi-parameter official tuning used different features and graph conditions.
- **Drop-edge** keeps the original model width and graph. On each training batch it masks 25% of unordered pairs in expectation, grouping directed arcs of the same pair and aligning attributes. Each daily graph is disjoint, so pair masking cannot cross session boundaries. Checkpoint selection and inference use the full graph. This differs from deterministic persistence and fixed sparse graphs.
- **Worst-half validation** keeps the original model and optimizer, but selects and stops on the smaller of the first and second chronological halves' mean within-day AUC. Folds have 25 validation sessions per half by design. Validation and fit AUC in the assessment tables use ordinary within-day AUC of the selected model, computed outside the scoped metric substitution. The historical field named `best_val_within_day_auc` in this experiment's training metadata carries the worst-half selection statistic. It is not the ordinary validation AUC. This intervention was tested only at five days, the unstable horizon that motivated it.

All comparisons used the original historical price-plus-correlation inputs, exact prior eligible keys and direction labels, the same 252 weight-fit and 50 validation sessions, five-session gaps, and fit-only graph and feature estimation. Window 0 assessed July 18–December 30, 2022. Window 1 assessed July 24–December 20, 2023. Original-attention controls were reused at seeds 42 and 43 and Price Only at seed 42. Every new run restored its earlier-selected checkpoint and used validation-selected classification thresholds. No model used outer scores to choose a threshold or checkpoint. The windows were inspected before this research and are retrospective. [Regularization protocol](protocol.md) · [Worst-half protocol](../gat_checkpoint_robust_v1/protocol.md)

## Interpretation

| Selected comparison | Accuracy | Majority | MCC | Within-day AUC |
|---|---:|---:|---:|---:|
| Original fallback, 2022 5d seed 42 | 55.48% | 51.16% | 0.1114 | 0.5438 |
| Width 32, 2022 5d seed 42 | 55.96% | 51.16% | 0.1354 | 0.5476 |
| Width 32, 2022 5d seed 43 | 51.30% | 51.16% | 0.0125 | 0.5296 |
| Width 32, 2023 5d seed 42 | 52.50% | 52.93% | −0.0047 | 0.4999 |
| Width 32, 2023 5d seed 43 | 47.79% | 52.93% | −0.0539 | 0.4611 |
| Original, 2023 5d seed 43 | 48.82% | 52.93% | −0.0702 | 0.4363 |
| Worst-half, 2023 5d seed 43 | 51.55% | 52.93% | −0.0316 | 0.5012 |

The smaller model's 2022 seed-42 AUC contrast versus the original was **+0.0037** with a paired interval **[−0.0071, +0.0143]**, and its accuracy lift was +0.48 percentage points vs the original. The 2022 seed-43 contrast reversed to −0.0108. Width reduction gave a 2023 seed-43 AUC increase from a weak 0.4363 baseline, but accuracy worsened. Drop-edge barely moved one-day ranking and did not help five-day ranking reliably. It peaked at 0.5430 at five days in 2022 seed 43. The worst-half criterion selected step 64 instead of 320 in 2023 seed 43 and improved AUC by **0.0649 [0.0142, 0.1243]**. Its 2023 seed-42 fit was identical in assessment to the original. In 2022 it tied at seed 43 and fell by 0.0079 at seed 42. This supports checkpoint sensitivity as one explanation for instability, with no evidence of robust superiority.

The highest archived historical accuracy in the registry remains **61.46%** from a separate 2023 five-day news run, below its **61.69%** majority. The highest persisted five-day AUC, MCC and accuracy lift now belong to the narrower 2022 seed-42 run. The highest persisted one-day AUC remains **0.5316** for 2022 boosted relative-return regression evaluated on the original absolute direction label. The smaller model is worth preserving as an individual research result. Use the already verified original fallback for the eventual demonstration. No notebook work was performed.

The paired intervals are descriptive 10-session date-block comparisons, preserving cross-sectional dependencies and overlapping five-session labels. No significance claim accounts for the number of already inspected variants and windows. Separate architecture width and checkpoint robustness both failed to transfer strongly across the two periods. Mixing them, adding another auxiliary loss, and testing regime-specific variants on these same outer windows would spend additional selection degrees of freedom on a repeatedly reused assessment. With no newly qualified point-in-time event/fundamental inputs or fresh held-out data, these are lower-value paths for this cycle. The 2026 score gate remains closed.

## Reproduction and files

The two new experiment directories contain the protocols, isolated training scripts, regression tests, logs, machine-readable manifests, full ledgers, fitted checkpoint state dictionaries and prediction Parquets. `gat_regularization_v1` additionally contains `verify.py`, `registry.py`, the combined report and registry. Both directories contain status and result tables. No production modelling file, historical artifact or notebook was changed.

All 20 checkpoint replays matched saved probabilities within 1e-7. Tests verify pair-mask symmetry, inference behavior, chronology, selection and existing model/graph behavior. **90 tests passed and Ruff passed.** Total recorded model training time was about 132 seconds and free disk stayed around 8.8 GiB. Recreate training in fresh sibling copies without existing `runs/` directories by using the `study` module in each directory. From the repository root, verify the saved fits and fallback with:

```sh
.venv/bin/python -B -m experiments.gat_regularization_v1.verify
.venv/bin/python -B -c 'from experiments.gat_regularization_v1.verify import verify; from experiments.gat_checkpoint_robust_v1.study import ROOT; verify(ROOT, 4)'
.venv/bin/python -B -m experiments.historical_demo_v1.replay development
```

[Highest new AUC checkpoint](runs/w0_h5_s42_small.pt) · [Its predictions](runs/w0_h5_s42_small.parquet) · [Worst-half checkpoint](../gat_checkpoint_robust_v1/runs/w1_h5_s43_worst_half.pt) · [Its predictions](../gat_checkpoint_robust_v1/runs/w1_h5_s43_worst_half.parquet) · [Fallback provenance](../historical_demo_v1/development_replay.json)
