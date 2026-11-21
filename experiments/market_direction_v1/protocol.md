# Direction study protocol

Frozen before opening any 2026 labels. Target is strictly positive adjusted close-to-close return on the exact first or fifth future NASDAQ session, with the original in-index and observed-target eligibility rules for the historical panel.

External research, if available, uses a fixed cohort of all securities in the historical membership table on 2023-12-29, with a single public provider and no adjustment-vintage splice. Request 2023 warm-up plus 2024–2026 sessions for all cohort identities, record failures and raw hashes. Require verified symbol continuity and complete source coverage or documented exit valuations before treating it as a final comparison. Do not use a surviving subset as a final NASDAQ-100 test. With qualified data use 2024–2025 development, reserve 2026 labels and metrics until the model and threshold are frozen, and purge at least five sessions between partitions. Archive quote adjustment vintage is an acknowledged limitation.

Fallback is retrospective only. Walk forward on historical data with assessments 2021-07-12 to 2021-12-31 and 2022-07-18 to 2022-12-30, each with 252 earlier fit sessions, 50 chronological calibration sessions and five unused sessions at each boundary. No previously published predictions enter model selection.

Compare market-only direction logistic, enriched direction logistic, regularized histogram gradient-boosted binary classification, and histogram gradient-boosted regression of same-day cross-sectional demeaned forward return. Score all against the original binary target. Run the frozen Price Only LSTM as a matched reference if feasible. Market-only is a diagnostic for market direction versus stock ranking. Select on mean assessment within-day AUC and cross-window consistency, with MCC/accuracy and simplicity as supporting evidence.

For each fit calibrate scores with a one-dimensional logistic fit on the separate calibration period. Use calibrated distance from 0.5 and calibration quantiles corresponding to 5%, 10%, 20%, 50% as fixed thresholds applied to assessment. Keep threshold ties and report actual coverage. Primary confidence target is predeclared at 20%; report all levels and same-subset majority accuracy. Report chronological/date-block uncertainty and class-specific selected performance. Never choose a threshold based on assessment accuracy. Regime is rising/falling by trailing 20-session equal-weight member return at signal close.

No executable-price trading claim from the historical archive. No official artifact edits.
