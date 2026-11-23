# Joint objective progress

Prior neural ranking replaced BCE entirely, and prior boosted rankers used a different architecture. The joint direction objective with same-session opposite-label pairs is a new controlled test. Lambda choices and outer assessment policy are in [protocol](protocol.md). The guarantee remains the original [2022 five-day fallback](../historical_demo_v1/development_replay.json). No notebook work.

The first cells verified training, validation-only commitment and matching outer keys against both controls. The initial 2022 5d joint result was 0.5401 AUC versus the original 0.5438. Seed-42 2022 1d reached 0.5159, but 2023 seed-42 1d fell to 0.4859. Continued predeclared replication without changing weights or rerunning failed outer variants. A materially distinct branch tests mild fit-only weighting by return magnitude in the sibling experiment.

Completed 16 joint fits, selected eight outer models based on earlier AUC alone, and completed eight distinct fit-only weighting fits. All outer artifact scores replayed. No stable improvement, no candidate promoted. [All scores](tables.md) · [Report](report.md) · [Registry](best_model_registry.json). Stop rather than add tuning degrees of freedom to the same chronological windows. The next defensible cycle needs timestamped inputs or a genuinely unused qualified assessment. 2026 and notebooks remain untouched.
