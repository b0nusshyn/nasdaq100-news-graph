# MACD scale study (2026-11-21)

**Question.** `macd`/`macd_signal` (`src/data/indicators.py:47`) are EMA differences on *adjusted*
closes, z-scored with one global mean/std over all tickers (`src/models/panel.py:196`). Their
cross-sectional size follows each ticker's adjusted price level, which is set by splits made
after the sample (e.g. 2024 10:1 splits), i.e. a look-ahead-contaminated, non-stationary factor.

**Evidence.** Mean daily Spearman(log|macd|, log adjusted close), 2021–23 members: **0.66**.
Adjusted log price level alone as a ranker: within-day AUC 0.4665 (2022 window), 0.5179 (2023):
the sign flips between windows. The archived original model's daily Spearman with log price
level: −0.06 (2022), −0.20 (2023).

**Arms.** Pre-registered, one comparison, same protocol as `gat_edge_weight_v1` (252 fit / 50
validation sessions, 5-session gaps, validation-selected checkpoint and threshold, corr graph,
12 epochs). `macd_norm` divides both MACD features by the same-day adjusted close (adjustment
factor cancels; data up to that close only). 8 fits, 70 s.

| Window | Seed | Arm | Within-day AUC | Accuracy | Majority | MCC |
|---|---|---|---:|---:|---:|---:|
| 2022 | 42 | original | 0.5438 | 0.5548 | 0.5116 | 0.111 |
| 2022 | 42 | macd_norm | 0.5323 | 0.5466 | 0.5116 | 0.093 |
| 2022 | 43 | original | 0.5404 | 0.5436 | 0.5116 | 0.085 |
| 2022 | 43 | macd_norm | 0.5341 | 0.5464 | 0.5116 | 0.092 |
| 2023 | 42 | original | 0.5035 | 0.5222 | 0.5293 | −0.023 |
| 2023 | 42 | macd_norm | 0.5066 | 0.5293 | 0.5293 | 0.004 |
| 2023 | 43 | original | 0.4363 | 0.4882 | 0.5293 | −0.070 |
| 2023 | 43 | macd_norm | 0.5019 | 0.5297 | 0.5293 | 0.009 |

Seed-averaged paired daily AUC difference (norm − original): 2022 −0.009, 2023 +0.034. Daily
AUCs of overlapping 5-day labels are autocorrelated, so naive t-test p-values are optimistic.

**Reproducibility.** Re-training the `original` arm reproduced the archived
`gat_edge_weight_v1` predictions **bit-exactly (max |Δp| = 0.0) for all four window/seed pairs**:
training itself, not only checkpoint inference, is deterministic on this machine.

**Conclusion.** Scale-free MACD removes a dependency on future-adjusted price level and removes
the seed-43 2023 collapse (worst case 0.436 → 0.502), but gives no reliable mean gain; 2022
is slightly lower. It is a validity correction, not a performance breakthrough, and no arm
approaches 0.63. The fallback demo keeps the original features (it is the verified artifact);
its MACD caveat should be stated.

Reproduce: `.venv/bin/python -B -m experiments.macd_scale_v1.study` (needs `runs/` absent).
Outputs: `runs/` (8 checkpoints, 8 prediction files, `ledger.jsonl`, `manifest.json` with
source hashes), `execution.log`.
