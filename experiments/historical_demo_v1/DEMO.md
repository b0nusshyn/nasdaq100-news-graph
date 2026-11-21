# Historical GAT-LSTM demonstration

## Use the 2022 experimental checkpoint for live computation in the room

Run from the repository root with the existing virtual environment:

```sh
.venv/bin/python -B -m experiments.historical_demo_v1.replay development
```

This performs **genuine inference using the existing checkpoint**. It prepares historical features, reconstructs the fit-only graph, compares it to the saved graph, loads weights, executes a forward pass for each assessment row, and checks every emitted probability by exact date, ticker and label. It recalculates metrics on the newly computed scores using the validation-selected threshold. It fits no weights and needs no network or new market data.

| 2022 development assessment | Result |
|---|---:|
| Accuracy | 0.5548008385744235 |
| Matched majority accuracy | 0.5116142557651991 |
| Within-day AUC | 0.5438356674216742 |
| MCC | 0.11137160864722019 |
| F1 | 0.5638708617432022 |
| Rows / sessions | 11,925 / 117 |
| Largest difference from archived probabilities | 0 |

Model `w0_h5_s42_original` is the original attention implementation with price inputs and correlation edges only. It has one 64-unit LSTM layer, four attention heads, 0.1 dropout and a 20-session lookback. No FinBERT or sector/co-mention graph is used during this replay. The seven inputs are one-day return, 20-day volatility, RSI 14, MACD, MACD signal, 20-day volume z-score and the complete-window flag. Feature scaling is fit on the 252 weight-fit sessions. Correlation is from market-residual adjusted-price returns on the same fit sessions. The 1,492 directed edges and weights rebuild exactly from the saved price panel. Assessment graph nodes use each historical day's eligible membership recorded in that panel. The existing label compatibility reader verifies the original stored labels against exact future-session closes in memory.

| Stage | Dates |
|---|---|
| Weight fitting and graph/statistic estimation | 2021-04-21 to 2022-04-19 |
| Validation | 2022-04-27 to 2022-07-08 |
| Development assessment | 2022-07-18 to 2022-12-30 |

Seed 42, selected checkpoint step 160 and threshold 0.4202089816331863. The date gaps are five sessions. This is an individual **retrospective development window**, not the aggregate performance of the official model. Prior seed 43 reached 0.5404 AUC on the same 2022 window, but the same 5-day architecture weakened in 2023 (0.5035 and 0.4363 at seeds 42 and 43). [2022 replay metadata and exact hashes](development_replay.json) · [Study ledger](../gat_edge_weight_v1/runs/ledger.jsonl) · [Checkpoint](../gat_edge_weight_v1/runs/w0_h5_s42_original.pt) · [Saved scores](../gat_edge_weight_v1/runs/w0_h5_s42_original.parquet) · [Fit-only graph](../gat_edge_weight_v1/runs/graph_0.parquet)

## Show the official five-fold results separately

```sh
.venv/bin/python -B -m experiments.historical_demo_v1.replay official-saved
```

This command recalculates metrics on genuine archived predictions with the original fold-specific thresholds. There is no forward pass. Dates cover the official 2021–2023 assessment folds. Majority is the mean of the observed same-population majority accuracies per fold.

| Official horizon | Accuracy | Matched majority | Within-day AUC | MCC |
|---|---:|---:|---:|---:|
| 1d | 49.93998599% | 51.64755197% | 0.50942247 | 0.00244779 |
| 5d | 50.39849613% | 54.02811734% | 0.49918314 | 0.00471403 |

[Official replay output](official_saved.json) · [Official prediction Parquet](../../data/predictions/gat_lstm.parquet) · [Original run record and thresholds](../../data/predictions/gat_lstm_run.json)

## Official graph reconstruction and inference audit

The original global `data/edges.parquet` is absent. The local per-fold graph builder inputs are present: historical prices, embedded membership flags and membership intervals, static GICS mapping, and aligned article links. Existing label and daily news artifacts are present. The price artifact contains 238,195 rows for 174 tickers from January 2018 through December 2023, including historical warm-up. No download is needed.

```sh
.venv/bin/python -B -m experiments.historical_demo_v1.replay official-rebuild
```

The command reconstructs the sector, correlation and co-mention graph in memory using the project builders. It never writes a global graph. The reconstruction has 18,816 rows and matches all 15,634 observed distinct archived fold edge/weight rows within 0.000001. These observed rows are a subset of the full graph. The graph's usable fold edge counts also match the official run record. Genuine checkpoint reconstruction matches archived probabilities for both horizons in folds 2–4, with discrepancies no greater than 0.00000012. Folds 0–1 for both horizons differ, with maximum absolute probability discrepancies from 0.0246 to 0.0767. Those mismatches persist when the graph contribution is removed in both the replay and the archive (up to 0.0368). The missing global edge artifact therefore does not explain the whole discrepancy. Inputs, preprocessing state, or checkpoint provenance for early folds require further reconciliation. This audit does not establish a specific software defect. Full official five-fold checkpoint parity remains unverified. [Reconstruction and branch diagnostics](official_rebuild_diagnostics.json)

For the professor, first execute the 2022 checkpoint command to demonstrate actual model computation. Then run the official saved-prediction command to show the historical five-fold metrics. If the compute environment is unavailable during the demonstration, the already-verified [development output](development_replay.json) and [official replay output](official_saved.json) are evidence of prior execution. Identify those files as saved results if shown. No fresh training is required. The official artifacts, checkpoints, notebook and locked 2026 experiment were preserved.
