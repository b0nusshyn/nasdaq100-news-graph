# Experiments: what we fixed, what we tried, what we found

This folder holds the research done **after** the official five-fold results. None of it
changes the official model (config T1, seed 42), its predictions in `data/predictions/`, or the
report's headline numbers. Every study has its own `report.md` or `EXPERIMENT_STATUS.md`.

## Bottom line

We tried many materially different ways to make the GAT-LSTM predict NASDAQ-100 up/down
direction better: hyperparameters, model size, regularization, loss functions, more training
history, graph variants, checkpoint rules, seed ensembles, new data (SEC fundamentals, FINRA
short volume) and classic price factors. **None produced a large or stable improvement.** The
best combination raised the worst case a lot but the average only a little:

| | Mean within-day AUC (2022 + 2023 windows, seeds 42/43) | Worst run |
|---|---:|---:|
| Original GAT-LSTM | 0.506 | 0.436 |
| Best combination: GAT (scale-free MACD) + walk-forward multi-factor model | 0.531 | 0.521 |

Official five-fold result for reference (unchanged): 1-day within-day AUC 0.509 (block-bootstrap
95% CI 0.500–0.520), 5-day 0.499 (CI 0.484–0.515). The evidence points to weak, regime-dependent
signal in the data rather than to an under-tuned model. Hyperparameter tuning (task 3.12,
`results/tables/tuning_gat_lstm.md`) moved validation AUC by at most 0.013 across six configs,
and the chosen config's validation 0.528 fell to 0.509 / 0.499 on test.

Caveat: the 2022 and 2023 development windows were inspected by many studies, so all numbers
below are development evidence, not untouched test results.

## Fixes

| Fix | Problem | Effect on official results |
|---|---|---|
| `notebooks/demo.ipynb` step 4 check | Exact-equality check passed only on the machine that wrote the official file; other CPUs differ by floating point (corr 0.9988, AUC 0.5009 vs 0.5010) | None; now checks correlation > 0.99 and AUC within 0.001 |
| `notebooks/demo.ipynb` step 8 (new) | Adds the 2022 checkpoint replay, labelled separately from the official result | None |
| `src/models/gat_lstm.py` | `torch_geometric` imported at module top, breaking the light CI set (CLAUDE.md rule) | None; retrained `original` and `signed` arms are bit-identical |
| `tests/test_market_direction_study.py` | Imported torch without skipping, so the light CI job failed at collection | None |
| SEC proxy feature store (`sec_fsds_v1`) | Old store let a newer filing without facts erase older fundamentals; ticker reuse mapped GEN, META, WTW, FISV, PCAR, FOX to the wrong company | New pipeline only |
| MACD scale (`macd_scale_v1`, study only) | MACD on split-adjusted closes carries each stock's later split history | Not applied to the official model |

Local data note: `data/labels.parquet` must be the renamed version (`fwd_ret_1d`, `fwd_ret_5d`)
or `read_table("labels")` raises `SchemaError`; `data/edges.parquet` comes from the shared Drive.

## Everything we tried

Within-day AUC is the mean of daily AUCs; 2022 = 18 Jul–30 Dec 2022, 2023 = 24 Jul–20 Dec 2023,
5-day labels unless noted. "Earlier" = studies run before 21 Nov; their numbers come from their
own reports.

### Model and training changes (earlier)

| Study | Idea | Result |
|---|---|---|
| `gat_lstm_tuning`, task 3.12 | 6 hyperparameter configs on validation | Spread 0.013 in validation AUC; T1 kept |
| `gat_regularization_v1` | 32-unit model, drop-edge | 32-unit 2022 seed 42: 0.548 / 55.96%, but reversed at seed 43 and in 2023 |
| `gat_data_scaling_v1` | 25–100% training history, recency weighting | More history did not help; full history 0.480–0.490 in 2022 |
| `gat_joint_v1` | BCE + same-day ranking loss | No stable improvement (2022 0.540 vs 0.544) |
| `offline_rank_v1` | Ranking objectives, boosted rankers | Best 1d 0.522, 5d 0.515; not stable |
| `gat_margin_weight_v1` | Down-weight near-zero returns | Changes under 0.0015 AUC |
| `gat_checkpoint_robust_v1` | Different checkpoint-selection rules | Fixed one weak cell (0.436 → 0.501), others tied or worse |
| `persistent_graph_v1` | Persistent-edge graph, 2-seed averaging | Small 2023 1d gain, not reproduced in 2022 |
| `gat_edge_weight_v1` | Signed / centred graph messages | No stable gain; source of the 2022 fallback checkpoint |
| `boosted_decision_v1`, `cohort_outperformance_v1` | Boosted trees, cohort-relative labels | ~0.527 AUC in 2024–2025 on a 30-stock cohort; no usable accuracy edge |
| `market_direction_v1`, `submission_readiness_v1` | Separate market and stock components | No candidate stably better |

### New studies (21 Nov)

| Study | Idea | Result (mean within-day AUC, 4 runs unless noted) |
|---|---|---|
| `macd_scale_v1` | MACD / close (scale-free) | 0.519 vs 0.506; removes the 2023 collapse, 2022 slightly lower |
| `sec_fsds_v1` | Point-in-time SEC fundamentals (coverage 0.8% → 93% of assessment rows) | Events 0.513, fundamentals 0.495, both 0.503: no help |
| selective prediction | Predict only the most confident 10–30% per day | 52–53% accuracy, below the full-population 55.5% |
| `factor_blend_v1` | GAT + low-volatility and MAX factors (50/50 rank blend) | 0.527, worst 0.515 |
| `factor_blend_v1` (input arm) | Same factors as GAT inputs | 0.488: overfits |
| `seed_ensemble_v1` | Average of 10 seeds | 2022 0.535, 2023 0.476: no help |
| `finra_short_v1` | FINRA daily short-sale volume | Screen failed (0.49–0.51 in 2019–2021) |
| `price_signal_screen_v1` | Overnight/intraday, industry-adjusted reversal, sector lead-lag, volume shocks | All failed the 2019–2021 screen |
| `multifactor_v1` | Walk-forward logistic model on 16 factors, blended with GAT | **0.531, worst 0.521** (best) |

Screening rule for new factors, fixed in advance: within-day AUC above 0.5 in each of 2019, 2020
and 2021 (not assessment years) and a mean of at least 0.51. Only low volatility and low MAX passed.

## Reproduce

```sh
.venv/bin/python -B -m experiments.historical_demo_v1.replay development   # 2022 checkpoint, AUC 0.5438
.venv/bin/python -B -m experiments.macd_scale_v1.study                      # needs runs/ absent
.venv/bin/python -B -m experiments.factor_blend_v1.study
.venv/bin/python -B -m experiments.multifactor_v1.study
```

The SEC and FINRA raw downloads are not in git (23 MB and 21 MB); rebuild them with
`experiments/sec_fsds_v1/fetch.py --fetch` (needs `SEC_USER_AGENT`) and
`experiments/finra_short_v1/fetch.py`, as described in each report. Model checkpoints (`*.pt`)
are git-ignored except the one the demo replays. `fnspid_candidate_audit/` (1.1 GB) stays local;
only its report is committed. `final_2026_reconciliation/` documents the 2026 evaluation, which
remains locked and unscored.
