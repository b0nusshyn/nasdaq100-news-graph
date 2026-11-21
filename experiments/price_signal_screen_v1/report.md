# Price-signal screen (2026-11-21)

Local data only (adjusted OHLCV, GICS sectors). Pre-registered rule: a feature advances only if
its within-day AUC for y_5d is > 0.5 in each of 2019, 2020, 2021 and averages >= 0.51 there.

| feature (direction as hypothesised) | 2019 | 2020 | 2021 | 2022 | 2023 | pass |
|---|---:|---:|---:|---:|---:|---|
| overnight return, 20d (Lou-Polk-Skouras) | 0.4968 | 0.4999 | 0.4987 | 0.5139 | 0.5073 | no |
| overnight return, 5d | 0.4936 | 0.4888 | 0.4902 | 0.5152 | 0.5004 | no |
| −intraday return, 20d | 0.5117 | 0.5159 | 0.4846 | 0.5071 | 0.5147 | no |
| −intraday return, 5d | 0.5027 | 0.5246 | 0.4850 | 0.4976 | 0.5059 | no |
| industry-adjusted 5d reversal (Da-Liu-Schaumburg) | 0.5006 | 0.5231 | 0.4896 | 0.4834 | 0.5045 | no |
| industry-adjusted 1d reversal | 0.5065 | 0.5211 | 0.4898 | 0.4945 | 0.5045 | no |
| lagged sector 5d return (industry momentum / lead-lag) | 0.5026 | 0.4860 | 0.4987 | 0.4850 | 0.5011 | no |

Earlier screens on the same rule: low 60d volatility and low 20d MAX pass (used in
factor_blend_v1); 5d/20d raw reversal, 12-1 momentum, liquidity, abnormal-volume returns, SEC
fundamentals and FINRA short volume fail. No model fitted from this screen.
