# GAT regularization status

Predeclared capacity and stochastic graph regularization against the matched historical graph and price models. The verified fallback remains [the 2022 five-session checkpoint](../historical_demo_v1/development_replay.json). No notebook work.

The initial drop-edge feasibility fit completed at 0.5412 AUC, versus the matched original 0.5438. The scheduled seed-42 width reduction reached 0.5476 in that same 2022 5d cell. Those are single-window observations. Both regularization arms are running their frozen matrix without changing settings.

Both frozen regularization arms completed all 16 fits. The 32-unit 2022 5d seed-42 fit was the individual leader at 0.547553 AUC and 55.96% accuracy, but the gain reversed at seed 43 and failed to transfer to 2023. Drop-edge remained unconvincing. The distinct worst-half selection branch completed four more fits and improved a weak 2023 seed-43 comparison only. All 20 checkpoints replayed within tolerance. No configuration was promoted. [Final report](report.md) · [Registry](best_model_registry.json). Next research requires qualified independent data. Notebook work remains out of scope.
