# Fit-subperiod graph persistence and seed averaging

## Decision

Keep the verified **2022 5-day original-attention model, seed 42**, as the demonstration fallback. Twelve genuinely new GAT fits and four no-fit ensemble evaluations completed. The strongest old fit still provides the best audited 5-day AUC and MCC. Neither intervention improved ranking reliably across windows or delivered stable classification accuracy above majority. The 63% accuracy and 0.63 within-day AUC goals were not reached. All windows here are retrospective development. [All matched model metrics and intervals](tables.md) · [Individual leaders and fallback](best_model_registry.json)

## Hypotheses and execution

Previously examined news had unverified release timestamps and no repeatable benefit. Signed edge weighting and graph centering did not transfer. Price-based tabular, rank, residual and market-component models also lacked stable lift. Stronger fit than assessment AUC and original GAT seed sensitivity especially in 2023 motivated the two distinct interventions below. Local historical data lacked validated point-in-time earnings and fundamental fields. This cycle reused the pinned original price and label artifacts without adding proxies. The absence of a new information family, along with a looming submission, disfavors more tuning on these assessed dates. There is no demonstrated new production implementation defect.

1. **Persistently selected peers.** Fit original residual correlation graphs on the first and second halves of the 252-session weight-fit segment. A full-fit edge is kept only if present with positive weight in both halves. Keep both directions and original full-fit weights. A sparse comparator takes the strongest full-fit pairs at the same directed edge count. This holds density constant while allowing coverage and peer distribution to differ. This intervention changes topology. The learned attention and feature paths are unchanged.
2. **Equal seed averaging.** Existing original-attention models at seeds 42 and 43 had identical eligible rows and labels. Center each margin on its own earlier validation cutoff, average the two margins equally and classify at zero. No calibration model or ensemble weights were fitted. The resulting margins are not probabilities. The rule did not improve on seed-42 ranking in any of the four cells.

Reuse the exact earlier original and Price Only controls, checked against the saved assessment keys and labels. Price Only was available at seed 42. Window 0 is July 18–December 30, 2022. Window 1 is July 24–December 20, 2023. For both horizons the 252-fit/50-validation split and five-session gaps match the prior study, with exact dates in [the run ledger](runs/ledger.jsonl). Both screened variants initially use seed 42 at each horizon and window. Persistence is replicated with seed 43 against original attention. All fit labels are original strictly-positive adjusted close-to-close return labels. The optimizer and preprocessing use only the earlier inputs. The first 2022 5d persistence run served as the actual training feasibility check at 0.5377 AUC and was counted once.

The common model uses original attention, a 20-session lookback, one 64-unit LSTM, 0.1 dropout, four attention heads, six price features plus the full-window flag, Adam learning rate 0.001, weight decay 0.0001, clip 1.0, and up to 12 epochs. Selection checks once per epoch with four epochs patience. Decision thresholds are selected on validation only. The experimental graph adapter checks that its weights and edges were constructed on the exact declared fit dates before handing them to the unchanged fit-only training pipeline.

## Graph reliability and model results

| Window | Fit-half peer-set Jaccard | Original edges | Persistent / sparse edges | Assessment retention, original / persistent / sparse |
|---|---:|---:|---:|---:|
| 2022 | 0.257 | 1,492 | 582 | 42.4% / 59.5% / 71.1% |
| 2023 | 0.291 | 1,478 | 628 | 33.3% / 48.4% / 49.7% |

Later graphs were built for this diagnostic after all training. Assessment similarity confounds sampling variation, graph rank cutoff and historical membership changes. Screening improved the fraction of retained fit edges seen later, but the equal-size strongest-edge comparator had higher retention in both windows. Persistence covered more graph nodes than equal sparsity (99 vs 92 and 101 vs 95). It did not reliably add ranking value over the sparse comparator. [Edge audit](runs/graph_diagnostics.json)

Persistence improved 2023 1d ranking relative to original attention in both seeds, by +0.0073 and +0.0055. Only seed 43's paired interval excluded zero. In 2022 the corresponding differences were +0.0019 and −0.0009. At 5d persistence lost to original in 2022 at both seeds (−0.0061, −0.0071) and improved slightly in 2023 (+0.0073, +0.0063), with intervals including zero. At seed 43 in 2023 its 5d validation-to-assessment AUC fell from 0.5421 to 0.4426 and its MCC was −0.1041, so the original generalization failure was not fixed. Same seed-42 2023 5d sparse rose to 0.5208 AUC vs original 0.5035, with paired interval −0.0018 to +0.0394. Its 2022 direction reversed, and it never beat majority accuracy at seed 42. This single-window signal did not justify additional seed sweeps. The complete metric tables also report train-probe and validation AUC, matched Price Only, balanced accuracy, F1 and call balance. Intervals use whole-date 10-session circular blocks and are descriptive.

| New result | Accuracy | Majority | MCC | Within-day AUC |
|---|---:|---:|---:|---:|
| Highest new AUC, 2022 5d sparse seed 42 | 49.58% | 51.16% | −0.0263 | 0.5401 |
| Highest new fitted accuracy, 2022 5d persistent seed 43 | 53.24% | 51.16% | 0.1014 | 0.5333 |
| Verified original fallback, 2022 5d seed 42 | 55.48% | 51.16% | 0.1114 | 0.5438 |

The 2022 5d equal-weight ensemble scored 55.14% accuracy and 0.5435 AUC. Its 2023 5d AUC was 0.4398, compared with 0.5035 for seed 42 alone. Other ensemble results and all supporting full-population metrics are in [ensemble results](ensemble/results.json). These ensembles do not require fresh model fits, but they were formed from saved scores. The new single-model checkpoints were all replayed with a forward pass and matched their saved probabilities within 1e-7.

## Registry and demonstration

The registry covers persisted original-direction fits from the 2019–2023 research and the new candidates. Individual metrics can select different runs. Highest verified historical accuracy in this registry is the 2023 5d next-session news run at 61.46% against 61.69% majority, with AUC 0.4894. Highest verified 5d AUC, MCC and accuracy lift occur together in the 2022 original fallback. At 1d the archived highest AUC is 0.5316 from the 2022 boosted relative regressor evaluated on the original absolute label. No candidate meets the cross-period promotion bar. The demonstration recommendation remains seed-42 2022 five-day original attention. The official five-fold aggregate should be shown separately from this individual window.

The initial fallback replay reproduced all 11,925 archived probabilities exactly and reconstructed all 1,492 graph edges. Its split dates, selected step, threshold, checkpoint and input hashes are recorded in the untouched [fallback replay](../historical_demo_v1/development_replay.json). Use the unchanged historical runner, with no fresh training or notebook operation:

```sh
.venv/bin/python -B -m experiments.historical_demo_v1.replay development
.venv/bin/python -B -m experiments.historical_demo_v1.replay official-saved
```

## Files and verification

Only this experiment directory was written. It contains the two protocols, status, this report, `study.py`, `test_topology.py`, `verify.py`, `registry.py`, `ensemble.py`, execution log and metric tables, registry and model artifact hashes. `runs/` contains the input/source manifest, six graph snapshots, topology audit, 12 checkpoints, 12 prediction Parquets and complete fit ledger. `ensemble/` contains four score Parquets with actual call decisions and the fixed averaging rule provenance in `results.json`. Total recorded training time was about 80 seconds across 12 fits. Disk remained around 8.8 GiB free.

Run `.venv/bin/python -B -m experiments.persistent_graph_v1.verify` from the repository root to replay all fitted checkpoints and recalculate contrasts. For fresh replication use a clean sibling copy without `runs/` or `ensemble/`, then invoke the `study` and `ensemble` modules. Use the pinned inputs and verify with the same runner. **89 tests passed and Ruff passed.** The official predictions, datasets, prior checkpoints, notebooks and 2026 lock were preserved.

Stop this branch. The screened graph's small 2023 advantage is exploratory, and a third round of tuning on those same windows would not establish generalization. New research should be driven by independently timestamped sources or a qualified unused window. Neither demo notebook was created, edited or executed.
