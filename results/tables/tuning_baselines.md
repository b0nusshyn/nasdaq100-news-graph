# Retrospective tuning study: LSTM baselines (Task 4.13, Issue #59)

Study fingerprint: `b019995a684e8130f208193ff5871290997f0f007665d342325c45272078bf2f`. Full configurations, data/source hashes, runtime and fold dates:
[`manifest.json`](tuning_baselines_runs/manifest.json). Per-fit JSON records retain thresholds and stopping diagnostics.

Selection and early stopping use validation within-day AUC; MCC uses each validation-selected threshold.
Five folds and both horizons receive equal weight; one configuration per baseline, shared across horizons.
Exact ties use T0–T5 order. All six candidates need ten valid records before a winner is selected.
Undefined metrics are N/A, never zero; an undefined AUC prevents selection for that baseline.

**Retrospective limitation:** later validation windows overlap earlier test periods. These are not
prospectively out-of-sample validated configurations. No test predictions or performance scores enter selection.
Shared preprocessing/PCA is fitted on the outer training fold, including internal validation; it is unchanged.
Validation scores are optimistic because checkpoints, thresholds and configurations reuse validation observations.

Historical T1 sidecars lack verifiable original data/source/runtime fingerprints and were not imported.
All rows use measured Task 4.13 records; resumption reuses them only after exact fingerprint verification.

| Baseline | Candidate | Batch days | Learning rate | Hidden | Heads | Dropout | Val MCC 1d | Val MCC 5d | Val within-day AUC 1d | Val within-day AUC 5d | Selection score | Outcome | Provenance |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| lstm_price | T0 | 32 | 1e-03 | 64 | N/A | 0.1 | 0.033714 | 0.021790 | 0.517311 | 0.492780 | 0.505046 | not selected | 10 measured, fingerprint-verified records |
| lstm_price | T1 | 8 | 1e-03 | 64 | N/A | 0.1 | 0.033521 | 0.009902 | 0.522691 | 0.500926 | 0.511809 | selected | 10 measured, fingerprint-verified records |
| lstm_price | T2 | 8 | 3e-04 | 64 | N/A | 0.1 | 0.030287 | 0.032873 | 0.518852 | 0.498420 | 0.508636 | not selected | 10 measured, fingerprint-verified records |
| lstm_price | T3 | 8 | 1e-04 | 64 | N/A | 0.1 | 0.038112 | 0.040912 | 0.515344 | 0.494065 | 0.504704 | not selected | 10 measured, fingerprint-verified records |
| lstm_price | T4 | 16 | 3e-04 | 64 | N/A | 0.1 | 0.036274 | 0.038414 | 0.516950 | 0.496425 | 0.506688 | not selected | 10 measured, fingerprint-verified records |
| lstm_price | T5 | 8 | 3e-04 | 64 | N/A | 0.3 | 0.034580 | 0.033019 | 0.518668 | 0.498309 | 0.508489 | not selected | 10 measured, fingerprint-verified records |
| lstm_news | T0 | 32 | 1e-03 | 64 | N/A | 0.1 | 0.039124 | 0.034750 | 0.516049 | 0.498619 | 0.507334 | not selected | 10 measured, fingerprint-verified records |
| lstm_news | T1 | 8 | 1e-03 | 64 | N/A | 0.1 | 0.038280 | 0.027626 | 0.522456 | 0.519442 | 0.520949 | selected | 10 measured, fingerprint-verified records |
| lstm_news | T2 | 8 | 3e-04 | 64 | N/A | 0.1 | 0.037796 | 0.035427 | 0.515880 | 0.516864 | 0.516372 | not selected | 10 measured, fingerprint-verified records |
| lstm_news | T3 | 8 | 1e-04 | 64 | N/A | 0.1 | 0.027106 | 0.034565 | 0.508991 | 0.518816 | 0.513903 | not selected | 10 measured, fingerprint-verified records |
| lstm_news | T4 | 16 | 3e-04 | 64 | N/A | 0.1 | 0.036322 | 0.035084 | 0.513324 | 0.516067 | 0.514695 | not selected | 10 measured, fingerprint-verified records |
| lstm_news | T5 | 8 | 3e-04 | 64 | N/A | 0.3 | 0.037257 | 0.035088 | 0.516103 | 0.516981 | 0.516542 | not selected | 10 measured, fingerprint-verified records |
