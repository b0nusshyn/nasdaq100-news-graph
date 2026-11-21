# Factor blend study (2026-11-21)

Same task (5-day up/down), same 2022/2023 assessment keys and labels, same validation-selected
thresholds. Design fixed before any 2022/2023 combination result was seen (see `study.py`
docstring). Factors screened on 2019–2021 only: low 60-day volatility and low 20-day MAX return,
both with within-day AUC > 0.5 in every year 2019–2023 (~0.52).

| Arm | Mean wdAUC | Worst wdAUC | wdAUC 2022 / 2023 | Acc 2022 / 2023 | Mean MCC |
|---|---:|---:|---|---|---:|
| factor only (no NN) | 0.5295 | 0.5246 | 0.5344 / 0.5246 | 0.5173 / 0.5147 | 0.027 |
| **blend (GAT macd_norm + factor)** | **0.5270** | **0.5151** | 0.5375 / 0.5166 | **0.5506 / 0.5285** | 0.048 |
| GAT macd_norm | 0.5187 | 0.5019 | 0.5332 / 0.5042 | 0.5465 / 0.5295 | 0.050 |
| blend (GAT original + factor) | 0.5157 | 0.4669 | 0.5400 / 0.4914 | 0.5507 / 0.5035 | 0.023 |
| GAT original (fallback arm) | 0.5060 | 0.4363 | 0.5421 / 0.4699 | 0.5492 / 0.5052 | 0.026 |
| GAT + factors as inputs | 0.4882 | 0.4299 | 0.5361 / 0.4404 | 0.5309 / 0.4557 | −0.015 |

Majority: 2022 0.5116, 2023 0.5293. Seeds 42/43 for every GAT arm (factor arm is seed-free).

Findings
- The blend of the scale-free-MACD GAT with the factor composite is the most robust model:
  every one of its 4 runs has wdAUC ≥ 0.515 (original GAT: 0.436–0.544), and 2022 accuracy
  55.3 % / 54.9 % vs 51.2 % majority. In 2023 accuracy equals majority (52.8 % vs 52.9 %).
- The two simple factors alone rank as well as or better than any GAT arm on average; the
  GAT adds 2022 accuracy (timing via its threshold) but not stable ranking.
- Feeding the factors to the GAT as inputs overfits (2023 wdAUC 0.43–0.45).
- Gains are ~+0.02 mean wdAUC over the original GAT; nothing approaches 0.6.

Caveat: 2022 and 2023 were inspected by earlier studies; this is development evidence, not an
untouched test. Six arms are reported in full; none was dropped.

Reproduce: `.venv/bin/python -B -m experiments.factor_blend_v1.study` (needs `runs/` absent;
blend arms read archived GAT predictions from gat_edge_weight_v1 and macd_scale_v1).
