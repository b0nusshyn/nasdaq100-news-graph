# Seed ensemble study (2026-11-21)

Pre-registered: scale-free-MACD GAT-LSTM, seeds 42–51 (all kept), 5-day, same windows,
partitions and selection as gat_edge_weight_v1. Ensemble = mean p_up, threshold = mean of the
ten validation-selected thresholds. 20 fits, 163 s. Seeds 42/43 reproduce macd_scale_v1 exactly.

| | 2022 wdAUC | 2023 wdAUC | 2022 acc (maj 0.5116) | 2023 acc (maj 0.5293) |
|---|---|---|---|---|
| single seeds, range | 0.5223–0.5389 | 0.4211–0.5084 | 0.5116–0.5506 | 0.4706–0.5306 |
| 10-seed mean-probability ensemble | 0.5347 | 0.4758 | 0.5440 | 0.5280 |
| ensemble + factor blend | 0.5370 | 0.4963 | 0.5466 | 0.5287 |
| *post-hoc* rank-mean ensemble | 0.5340 | 0.4945 | – | – |

Why 2023 drops: seed 47's daily p_up dispersion is 0.055 vs 0.010–0.022 for the other seeds,
so it dominates the probability mean; a post-hoc rank mean (labelled as such, not adopted)
still gives 0.4945. In 2023 most seeds output nearly constant cross-sections: the network finds
no ranking signal there, and averaging cannot create one.

Conclusion: seed ensembling does not improve the model. The single-seed spread (2023:
0.421–0.508) shows how much a reported number could be inflated by picking a seed; that was
not done. Best robust result remains experiments/factor_blend_v1 `blend_macd_norm`.
