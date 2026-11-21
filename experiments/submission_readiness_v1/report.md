# Submission-ready NASDAQ-100 results

## What to demonstrate

Use the **official seed-42 T1 GAT-LSTM saved evaluation** as the primary project demonstration. Its official fold means were replayed from the prediction artifacts using the saved fold thresholds. The benchmark compares each model on its original eligible rows. The majority shown here is the mean of the per-fold observed class majority, a benchmark that has access to fold prevalence.

| Official horizon | Accuracy | Same-population majority | MCC | Within-day AUC |
|---|---:|---:|---:|---:|
| 1 day | 49.94% | 51.65% | 0.0024 | 0.5094 |
| 5 days | 50.40% | 54.03% | 0.0047 | 0.4992 |

The original 20-session GAT-LSTM uses 64 hidden units, dropout 0.1, four heads, price inputs plus news scalars and 64 PCA components, and sector, correlation and co-mention edges. The saved T1 config and fold dates, thresholds and states are in the [registry](best_model_registry.json). All ten local checkpoint state dicts loaded under the recorded architecture. Their manifest agreed with the official run's selected steps and thresholds. The dedicated archived attention probabilities reproduced all 128,152 official predictions exactly on identical horizon, fold, ticker and date keys. This is archive parity and checkpoint load verification. The complete input graph is absent locally, so live reconstruction of the original official feature/graph inference was not verified. The [official prediction Parquet](../../data/predictions/gat_lstm.parquet) and [attention artifact](../../data/attention/gat_lstm_contribution.parquet) provide the reproducible fallback. The demo notebook was untouched and its fit cells were not run.

For an optional checkpoint inference walkthrough, use the **2022 5-day original-attention price-plus-correlation research model**, seed 42, assessment July 18 through December 30. Its checkpoint probabilities were replayed from actual inputs by the edge-study verifier. It scored **55.48% accuracy**, **51.16% majority**, **0.1114 MCC**, and **0.5438 AUC** on that one reviewed window. The 2022 result held across seed 43 (0.5404 AUC), but the 2023 result fell to 0.5035 at seed 42 and 0.4363 at seed 43. Present it as a reproducible experimental example. [Checkpoint](../gat_edge_weight_v1/runs/w0_h5_s42_original.pt) · [Predictions](../gat_edge_weight_v1/runs/w0_h5_s42_original.parquet)

## New controlled experiment

The preceding ranking study showed much higher fit than assessment ranking AUC. Earlier diagnostics found substantial shared daily variation in official model scores but weak correspondence with actual direction. We tested whether separating daily market direction from security-specific direction improves decisions without introducing more high-capacity fits.

The preregistered additive system fits two deterministic Ridge estimators (alpha 50) in each of four window/horizon cells. One predicts the daily fraction Up from the six existing market features. The other fits the stock's binary direction minus its own fit day's observed Up fraction, on 12 eligible-stock-centered features. Within-date centering uses only that day's features. The realized daily fraction is a fitting target and never enters assessment inference. Median imputation and scaling fit only on the earlier weight-fit partitions, and all decision cutoffs come from separate earlier 50-session windows using the fixed balanced-accuracy search. The fit windows have 252 sessions and at least five unused sessions between weight-fit, cutoff and assessment. No seed or hyperparameter search occurred.

**Eight estimators were actually fitted** in the current cycle. Combined and stock-only are scores from those same fits. No baseline was refit. The previous 16-fit offline ranker study was reused as a matched control and is separate from the eight new fits. The verified assessment populations contain 3,390 rows over 113 dates in July–December 2024 and 3,360 rows over 112 dates in July–December 2025. The pinned 30-stock cohort and exact-session positive-return targets were unchanged. The 101-security 2026 readiness flag stayed false. [Configuration, source hashes and model settings](runs/manifest.json) · [All matched scores and paired intervals](matched_results.md)

| Window | Horizon | Combined accuracy | Majority | MCC | Within-day AUC |
|---|---|---:|---:|---:|---:|
| 2024 | 1d | 52.80% | 52.60% | 0.0355 | 0.5108 |
| 2025 | 1d | 49.85% | 52.77% | 0.0180 | 0.4962 |
| 2024 | 5d | 51.77% | 52.18% | 0.0112 | 0.4686 |
| 2025 | 5d | 53.51% | 54.46% | −0.0071 | 0.4534 |

Adding the market component raised accuracy relative to stock-only in all four cells (0.27 to 1.55 percentage points) and left within-day AUC unchanged, as expected when adding one constant per date. It beat the majority only in 2024 1d, by 0.21 points. The 2024 1d result led the matched logistic, boosted and pairwise arms on AUC, but the date-block contrast against logistic included zero. The strongest 2025 1d comparator was the pairwise model at 0.5224. In both 5d windows the combined system trailed the strongest matched ranking comparator. In 2024 5d the paired AUC difference versus logistic was −0.0463 [−0.0836, −0.0096]. The ablation supports a distinction between classifying overall direction and ranking stocks. It does not establish reliable predictive improvement. Both chronological windows were previously inspected, and cross-seed robustness is inapplicable to these deterministic Ridge fits.

## Registry and best-model decision

The replay registry covers 109 persisted original-target development run/score records from the price, graph, news, rank and restricted-cohort studies, plus the ten official GAT folds. It excludes different labels, selective accuracy, and unverified summary-only records without prediction files. Each recorded candidate has model, horizon, assessment dates, seed where applicable, metrics, exact prediction path and hash, configuration provenance and checkpoint path and hash where one is locally available.

| Verified development maximum | 1d | 5d |
|---|---|---|
| Full-population accuracy | 54.94%, delayed-news LSTM, majority 55.17% | 61.46%, next-session-news LSTM, majority 61.69% |
| Within-day AUC | 0.5316, boosted relative-return regression evaluated on absolute direction | 0.5438, original-attention price-plus-correlation GAT |
| Accuracy lift vs same-population majority | +0.50 pp, centered graph, AUC 0.5153 | +4.32 pp, original-attention price-plus-correlation GAT, AUC 0.5438 |

The maximin 2024–2025 ranking candidates in the registered comparison are the relative regressor at 1d (0.5183 and 0.5136) and stock logistic at 5d (0.4893 and 0.5411). Their saved calibrated decisions were all positive and tied the matched majority in both years. Neither qualifies as a stable full-task winner. Single-window accuracy maxima are similarly driven by prevalence. **No verified run in this registry reached 0.63 AUC or 63% full-population accuracy.**

## Reproduction, limitations and stopping decision

- From the repository root, run `.venv/bin/python -B -m experiments.submission_readiness_v1.registry` to regenerate the registry and verify archived official inference, state-dict loading and official metrics. It does not train anything.
- Run `.venv/bin/python -B -m experiments.submission_readiness_v1.verify` to replay all Ridge estimators on saved rows, verify calls and render matched statistics. To redo training, use the `decompose` module in a fresh isolated experiment copy without a `runs/` directory. The original executed source is retained in `runs/executed_source.py.snapshot`; the current source removes one unused import.
- Run `.venv/bin/python -B -m experiments.gat_edge_weight_v1.verify` for the experimental GAT checkpoint replay. Do not rerun the official training pipeline to demonstrate the archive.
- **52 tests passed and Ruff passed.** Files created here include `protocol.md`, `decompose.py`, `test_decompose.py`, `verify.py`, `registry.py`, the machine-readable registry, matched tables, this report and the status file, plus the new manifest, ledger, four paired-estimator files and four prediction files in `runs/`. Official artifacts, notebooks and prior study outputs were preserved.

The available restricted cohort does not resolve market membership, original adjustment vintages or point-in-time event information. No newly qualified data or untouched assessment was available. This experiment gave no repeatable improvement on the primary metric, and further search across the same archival inputs and observed windows risks overfitting while reducing submission readiness. Stop this cycle and preserve the official fallback. A later new experiment should start from vetted accession-linked fundamentals or timestamped earnings events and a qualified chronological assessment, then compare on the same original labels.
