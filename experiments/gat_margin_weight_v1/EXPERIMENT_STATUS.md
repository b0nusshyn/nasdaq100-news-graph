# Fit-only weighting progress

The [protocol](protocol.md) fixes the risk of near-zero-label noise as the hypothesis. It uses original labels and all eligible rows, with mildly different gradient weights determined strictly inside the fitting dates. The first 2022 5d seed-42 fit reached 0.5437 versus 0.5438 for original attention, and 2023 seed-42 5d reached 0.5033 versus 0.5035. The fixed matrix continues across both horizons and seeds. The [verified original fallback](../historical_demo_v1/development_replay.json) remains protected.

All eight fits completed. Within-day AUC shifts versus original were each under 0.0015. The best raw AUC was 0.5437 (2022 5d seed 42), and the strongest accuracy was 55.70% (2022 5d seed 43). The latter did not transfer to 2023. All eight prediction files were replayed from checkpoints. [Combined report](../gat_joint_v1/report.md) · [Tables](../gat_joint_v1/tables.md) · [Updated registry](../gat_joint_v1/best_model_registry.json). No candidate promoted.
