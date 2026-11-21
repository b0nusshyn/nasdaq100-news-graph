# Walk-forward multi-factor study (2026-11-21)

Design fixed in `study.py` docstring before results: 16 within-day-ranked factors (price,
volume, sector-relative, SEC store_v3, FINRA lagged), logistic regression C=0.1, trained on
2019-01-02 .. assessment start − 11 sessions (label + 5-session embargo), scored on the exact
archived 2022/2023 5-day assessment keys. Blend = 50/50 within-day z with the scale-free-MACD
GAT (seeds 42/43), GAT level and validation threshold kept.

| Arm | wdAUC 2022 | wdAUC 2023 | Acc 2022 (maj 0.5116) | Acc 2023 (maj 0.5293) |
|---|---|---|---|---|
| multi-factor alone | 0.5313 | 0.5236 | 0.4920 (thr 0.5) | 0.5269 |
| **multi-factor + GAT, s42 / s43** | **0.5381 / 0.5389** | **0.5257 / 0.5214** | 0.5464 / 0.5467 | 0.5291 / 0.5293 |
| previous best: 2-factor + GAT (factor_blend_v1) | 0.5375 (mean) | 0.5166 (mean) | 0.5506 | 0.5285 |
| original GAT (fallback arm) | 0.5421 (mean) | 0.4699 (mean) | 0.5492 | 0.5052 |

Multi-factor + GAT: mean wdAUC 0.5310, worst 0.5214 (2-factor blend: 0.5270 / 0.5151; original
GAT: 0.5060 / 0.4363). Accuracy beats majority in 2022 by ~3.5 points and equals it in 2023.

Learned weights are dominated by low volatility (0.22 / 0.19); most others are small or offset
(5d reversal vs. industry-adjusted reversal). The gain over the 2-factor blend is ~+0.004 mean
wdAUC — within seed/period noise. Development evidence only (2022/2023 inspected before).
