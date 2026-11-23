# Joint direction and ranking objective

## Audit and design

The standard GAT trainer optimizes BCE only. The earlier neural `within_day_pairwise` branch replaced BCE completely. The separate boosted ranking study used grouped trees, not GAT-LSTM. The new isolated objective preserves direction BCE and adds a distinct positive-versus-negative same-session term. No tickers are paired across dates, and unlabelled nodes are excluded. Pair loss is stable softplus of negative logit minus positive logit, averaged inside each date and then across eligible mixed-class dates. Equal scores have a useful gradient. Single-class dates contribute BCE and no pairwise term.

The original price-plus-correlation model uses 64 hidden units, four heads, a 20-session lookback, one recurrent layer and dropout .1. Fit uses 252 sessions and validation 50, with five-session embargo boundaries. Adam learning rate .001, weight decay .0001, eight dates per batch, clip 1.0, at most 12 epochs, epoch validation and four epochs patience. Fit-only price normalization and correlation graph creation use the unchanged shared code. All original adjusted positive-return labels are verified in memory against the exact future exchange close. The outer populations match both the archived same-seed 64-unit and 32-unit BCE controls. Window 0 is July 18–December 30, 2022, and window 1 is July 24–December 20, 2023. Both were already inspected. No 2026 data entered this experiment.

For each horizon, period and seed, both λ values (.10 and .30) fitted without assessment inference. Standard within-day validation AUC chose the checkpoint and lambda, with smaller lambda breaking ties. The chosen checkpoint was logged before the outer forward pass. Validation MCC chose the direction cutoff. BCE validation loss remained unchanged. The candidate loser has a checkpoint, validation predictions and fit/validation scores, but no outer score. The first pair of fits doubled as the feasibility check and counted once.

A distinct fixed follow-up evaluated mild loss weighting for high-confidence supervision on large realized moves. It assigns an eligible weight-fit row .75 + .50 times its own date's percentile rank of absolute forward return. BCE is weighted only at training time. Labels, evaluation population, feature values, validation objective, and scores at inference are unchanged. Using fit-period returns as training supervision is separated from inference. Unweighted targets outside the mapped fit dates raise an error. This tests the potential noisiness of small return signs rather than ranking or changing the positive-return target. [Joint protocol](protocol.md) · [Weighting protocol](../gat_margin_weight_v1/protocol.md)

The shared production model and original trainer were preserved. The trial trainer is an isolated copy with a training-objective hook and an option to return selected weights before assessment. The import audit uncovered an undefined default model constant in the unused outer-prediction branch of the copied trainer. It was fixed by importing the shared constant. All recorded runs fit with that branch disabled. The pre-format and pre-import versions actually executed are retained and checked against their manifests.

## Results and decision

**Twenty-four fits completed**: 16 joint candidates (eight selected outer assessments, eight validation-only), plus eight predeclared magnitude-weighted BCE runs. No original or 32-unit control was refit. Both horizons and seeds 42/43 are present in both chronological windows. Approximate summed model-fit times were 161 seconds for joint and 96 seconds for magnitude weighting, with two CPU threads per runner. Some work overlapped. [All model scores and paired comparisons](tables.md)

| Selected observation | Accuracy | Matched majority | MCC | Within-day AUC |
|---|---:|---:|---:|---:|
| 2022 5d seed 42 original fallback | 55.48% | 51.16% | 0.1114 | 0.5438 |
| 2022 5d seed 42 32-unit BCE | 55.96% | 51.16% | 0.1354 | 0.5476 |
| 2022 5d seed 42 joint λ=.10 | 49.30% | 51.16% | −0.0382 | 0.5401 |
| 2022 5d seed 43 joint λ=.10, best new accuracy | 55.80% | 51.16% | 0.1165 | 0.5378 |
| 2022 5d seed 42 magnitude, best new AUC | 55.44% | 51.16% | 0.1105 | 0.5437 |
| 2023 5d seed 43 joint λ=.10 | 49.31% | 52.93% | −0.0597 | 0.4404 |

Validation selected λ=.10 in six of eight cells, and λ=.30 for the two 1d seed-42 cells. Joint loss did not improve the primary AUC consistently. The 2022 1d seed-42 gain versus original was +0.0022 with a paired interval [−0.0055, +0.0116]. The analogous 2023 result was −0.0178 [−0.0328, −0.0033]. The 2022 seed-43 joint fit gained 1.44 percentage points in accuracy over the original but lost 0.0026 AUC and did not transfer to 2023. The same 2023 seed-43 model fell from 0.5429 validation to 0.4404 assessment AUC, even though its fit AUC was 0.5285. This is closer to temporal instability than evidence that adding a ranking term solved the ranking problem. Differences use descriptive 10-session date blocks. They do not account for all prior research comparisons on these windows.

Mild magnitude weighting changed AUC by under 0.0015 in every cell versus the original. It improved 2022 five-day seed-43 accuracy to 55.70% from 54.36%, but 2023 seed-43 five-day AUC remained 0.4349 with negative MCC. The near-null ranking changes do not rule out all forms of small-move label noise. This predeclared weighting rule offered no stable gain.

**No new model reached 63% accuracy or 0.63 within-day AUC.** No checkpoint was promoted. The 32-unit 2022 seed-42 model still holds the best individual historical 5d AUC, and the highest archived accuracy of 61.46% remains a separate 2023 news fit below its own majority. The strongest defensible reproducible demonstration fallback is the unchanged original 2022 seed-42 GAT. The experiments support retaining the original baseline rather than tuning λ, changing cutoff policy, or testing more weight schedules on these same assessment labels. Future work needs validated new historically available information or an unused qualified assessment. The 2026 evaluation remains locked and no independent generalization was established. [Updated best-model registry](best_model_registry.json)

## Verification, files and commands

**91 tests passed and Ruff passed.** Tests include within-date pair grouping, tie gradients, single-class and masked labels, the fit-weight boundary, original modelling and graph behavior, and chronological splits. The verifier checks both candidate checkpoints and earlier prediction hashes, recalculates lambda choices, verifies each of the 16 outer prediction artifacts and replays each selected and magnitude checkpoint within 1e-7. The original fallback was replayed and again matched every archived probability exactly. Both notebooks and shared production source were preserved. No network purchases or Git operations. Free disk stayed around 8.8 GiB.

New files in `gat_joint_v1` are `protocol.md`, `EXPERIMENT_STATUS.md`, this report, `loss.py`, `trainer.py`, `study.py`, `verify.py`, `test_loss.py`, `execution.log`, `fallback_recheck.json`, `tables.md`, `best_model_registry.json`, plus its `runs/` manifests, ledgers, source snapshot, 16 checkpoint states, 16 validation prediction Parquets and eight outer Parquets. The sibling `gat_margin_weight_v1` contains its protocol, status, study, test, execution log, and `runs/` with manifest, fit ledger, source snapshot, eight checkpoint states and eight outer prediction Parquets.

```sh
.venv/bin/python -B -m experiments.gat_joint_v1.verify
.venv/bin/python -B -m experiments.historical_demo_v1.replay development
```

To reproduce training, use fresh isolated sibling copies without existing `runs/` folders and run their `study` modules from the repository root. The original state dict and threshold for any selected run are captured in the ledgers. [Best new accuracy checkpoint](runs/w0_h5_s43_lambda0.10.pt) · [Best new accuracy predictions](runs/w0_h5_s43_lambda0.10_assessment.parquet) · [Best new AUC checkpoint](../gat_margin_weight_v1/runs/w0_h5_s42_magnitude.pt) · [Best new AUC predictions](../gat_margin_weight_v1/runs/w0_h5_s42_magnitude_assessment.parquet)
