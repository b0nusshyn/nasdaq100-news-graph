# GAT-LSTM edge reliability and daily representation centering

## What the implementation actually did

The original model encodes each ticker's feature window with a shared LSTM. Each edge type has its own PyG GATConv. Per head, the learned source, destination, and scalar edge terms are summed, passed through LeakyReLU and normalized with softmax over incoming neighbors of the same type. Head outputs are averaged. Type-specific messages are summed and passed through ELU before a linear head combines that signal with the untouched ticker state. Consequently, weight can already affect neighbor choice through a freely learned logit term. It does not directly bound the amount of signal a neighbor can contribute. With one incoming neighbor, its coefficient is 1 regardless of weight. Negative edge values can enter logits but do not force an opposite-signed message. Even an isolated node can receive the original convolution's learned type bias. That bias is shared across nodes. There was no evidence of a broken gradient path or a mislabeled prediction.

The builder assigns sector edges a constant 1, co-mention edges a Jaccard coefficient, and correlation edges a signed training-window return correlation after fitted equal-weight market residualization. These are different quantities. The strongest-peer filter orders raw signed correlation from largest down, so it tends to select positive edges. The experiment graph for 2022 had 1,492 directed edges with weights 0.0495 to 0.8245 (median 0.2378). The 2023 graph had 1,478 with weights 0.0734 to 0.8999 (median 0.2316). Neither contained negative or zero edges. Negative and zero handling was verified in unit tests. Training results do not test the benefit of adding anticorrelated peers.

The training batches have separate graphs per daily membership list. Price windows use data through that day's close. Labels are used only for loss and evaluation masking. BCE trains direction over eligible ticker-days, including market timing, whereas the headline averages stock-level AUC within dates. Validation uses that headline to select and restore the checkpoint, and then sets a direction threshold by validation MCC. The original published configuration uses multiple checks within an epoch. All arms here used one check per epoch. Existing fit-only preprocessing and graph construction were used. Fitted correlation, market beta and scaling used only the weight-fit partition. The original legacy label artifact uses `ret_*d` names. A first launch failed before training at direct schema validation, and the rerun used the existing compatibility reader, which verifies keys, original returns and binary direction against exact exchange-session closes. No label file was edited.

## Implementation tested

- **Price** uses the original LSTM encoder and classification head without edges.
- **Original** uses the unchanged attention and message rule with fit-only correlation edges.
- **Signed** retains learned logits and per-destination normalized attention. Its message coefficient is the normalized attention times the original signed edge weight. This makes a lone 0.2 edge deliver one-fifth the strength of a lone 1.0 edge with the same transformed state. No extra normalization cancels the magnitude. Zero-weight edges are excluded before softmax, and the experimental convolution has no output bias so isolates have exactly zero graph contribution.
- **Centered** adds subtraction of the same-session mean LSTM state from each graph input before the graph projections, attention and nonlinear head. The ticker's uncentered state remains in the self branch. This alters relative logits and can change ranks. Batch grouping uses the observed membership graph node IDs and is independent of future label availability.

The original behavior stays as the default. The exposed attention values still describe normalized neighbor shares. For signed messages the effective coefficient additionally uses the original edge weight. This comparison uses correlation-only graph topology and common price features in every arm. It isolates the new graph rule while avoiding date-uncertain news and static sector classifications. It does not reproduce the original three-edge-type news model.

## Design and reproducibility

Both development periods were previously available to the project. No 2026 input, label, or metric was opened. Each period used 252 fit sessions, a five-session embargo, 50 validation sessions, then another five-session embargo before the assessment. There was no post-assessment tuning. The price feature set is the baseline's six indicators plus the full-window flag. Both horizons use the same original direction target. Training used a 20-session lookback, one LSTM layer with hidden size 64 and dropout 0.1, four graph heads, Adam with learning rate 0.001 and weight decay 0.0001, gradient clip 1.0, eight dates per batch, a 12-epoch cap, and patience of four epochs. No learning-rate schedule was used.

The first 16 fits compare all four arms across both periods and horizons using seed 42. Eight seed-43 fits replicate the original-versus-signed comparison without centering. Train AUC is the within-day AUC on every fifth fit date at the selected checkpoint. Test entries below are developmental assessments. The paired intervals use 2,500 circular 10-session date blocks, preserve common dates, and are descriptive without multiple-comparison correction.

## Results

Seed-42 means weight both assessment windows equally.

| Configuration | 1d accuracy | 1d within-day AUC | 5d accuracy | 5d within-day AUC |
|---|---:|---:|---:|---:|
| Price Only | 49.46% | 0.4986 | 49.15% | 0.5099 |
| Original attention | 50.63% | 0.5087 | 53.85% | 0.5237 |
| Signed messages | 51.42% | 0.5049 | 50.77% | 0.5202 |
| Signed messages and centered inputs | 52.00% | 0.5041 | 48.29% | 0.5055 |
| Same-population majority accuracy | 51.97% | — | 52.05% | — |

The original attention led Price Only in all four seed-42 AUC comparisons. Only the 2023 1d paired contrast excluded zero, at +0.0192 [0.0012, 0.0379]. The Price Only control was not repeated at seed 43 within budget, so graph-versus-no-graph replication remains unmeasured. Direct reliability scaling lost to original attention in five of eight matched cells, including all four 2022 comparisons, and every paired interval included zero. Centering gave small gains over signed-only messages in 2022 and lost in both 2023 cells. The 2023 5d centered arm lost 0.0305 AUC [−0.0607, −0.0025] against signed-only. A second seed was reserved for the edge intervention, so centering has no second-seed result.

The strongest individual fit was 2022 5d original attention at 55.48% accuracy versus 51.16% majority, 0.1114 MCC and 0.5438 AUC. In seed 43 for that window the same original attention configuration reached 54.36% accuracy and 0.5404 AUC. In 2023 its 5d AUC was 0.5035 at seed 42 and 0.4363 at seed 43. The latter came from a checkpoint with 0.5452 validation AUC, a 0.1089 drop to assessment. The signed model in that seed had a similar gap, from 0.5494 to 0.4406. Thirteen of 24 fits selected the first epoch's checkpoint. The recorded histories show unstable transfer across windows and optimization seeds. They do not establish whether common market timing is the primary cause.

Best individual 1d accuracy and AUC came from the 2022 centered arm at 52.38% against 51.87% majority and 0.5153 AUC. Its 2023 1d AUC fell to 0.4930. The results can exceed the nominal 50% level in some cells. The published 1d accuracy/AUC of 49.94%/0.5094 and 5d of 50.40%/0.4992 use a different evaluation and are descriptive context only. No development fit reached 60% accuracy or 0.60 AUC, and neither 70% goal was reached. No independent scoring occurred.

Retain original attention as the graph research control and Price Only as the simple reference. The optional code paths are available for reproducibility, with no evidence to justify enabling them in the default configuration. There was no hyperparameter search after this comparison. All 24 fits were used.

### Complete run records

Up share is at the validation-selected threshold. Down share is its complement and is stored explicitly in the ledger. Checkpoint cells give selected versus completed optimizer steps.


| Run | Rows | Fit AUC | Val AUC | Assess AUC | Fit–assess gap | Acc. | Majority | MCC | F1 | Up % | Step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w0_h1_s42_price | 11925 | 0.5311 | 0.5166 | 0.5128 | 0.0183 | 0.5060 | 0.5187 | 0.0282 | 0.5776 | 68.8% | 224/352 |
| w0_h1_s42_original | 11925 | 0.5265 | 0.5202 | 0.5137 | 0.0128 | 0.4976 | 0.5187 | 0.0229 | 0.6096 | 80.6% | 224/352 |
| w0_h1_s42_signed | 11925 | 0.5278 | 0.5170 | 0.5107 | 0.0172 | 0.5105 | 0.5187 | 0.0241 | 0.5204 | 53.9% | 64/192 |
| w0_h1_s42_centered | 11925 | 0.5290 | 0.5176 | 0.5153 | 0.0138 | 0.5238 | 0.5187 | 0.0339 | 0.3605 | 26.3% | 32/160 |
| w0_h5_s42_price | 11925 | 0.5394 | 0.4930 | 0.5260 | 0.0134 | 0.4870 | 0.5116 | -0.0231 | 0.6545 | 99.6% | 32/160 |
| w0_h5_s42_original | 11925 | 0.5291 | 0.5073 | 0.5438 | -0.0147 | 0.5548 | 0.5116 | 0.1114 | 0.5639 | 53.2% | 160/288 |
| w0_h5_s42_signed | 11925 | 0.5354 | 0.5018 | 0.5359 | -0.0004 | 0.4956 | 0.5116 | -0.0376 | 0.1919 | 13.6% | 32/160 |
| w0_h5_s42_centered | 11925 | 0.5374 | 0.5034 | 0.5370 | 0.0004 | 0.5011 | 0.5116 | -0.0249 | 0.1702 | 11.3% | 32/160 |
| w1_h1_s42_price | 10705 | 0.5358 | 0.5155 | 0.4844 | 0.0513 | 0.4832 | 0.5206 | -0.0307 | 0.4755 | 46.5% | 96/224 |
| w1_h1_s42_original | 10705 | 0.5077 | 0.4807 | 0.5037 | 0.0041 | 0.5150 | 0.5206 | -0.0220 | 0.6733 | 96.4% | 32/160 |
| w1_h1_s42_signed | 10705 | 0.5074 | 0.4838 | 0.4991 | 0.0083 | 0.5178 | 0.5206 | -0.0163 | 0.6793 | 98.3% | 32/160 |
| w1_h1_s42_centered | 10705 | 0.5272 | 0.4820 | 0.4930 | 0.0342 | 0.5163 | 0.5206 | -0.0092 | 0.6702 | 94.6% | 128/256 |
| w1_h5_s42_price | 10701 | 0.4959 | 0.5185 | 0.4938 | 0.0021 | 0.4959 | 0.5293 | -0.0168 | 0.5423 | 57.2% | 32/160 |
| w1_h5_s42_original | 10701 | 0.4995 | 0.4969 | 0.5035 | -0.0040 | 0.5222 | 0.5293 | -0.0226 | 0.6785 | 95.7% | 32/160 |
| w1_h5_s42_signed | 10701 | 0.4937 | 0.5000 | 0.5045 | -0.0108 | 0.5198 | 0.5293 | -0.0199 | 0.6700 | 92.6% | 32/160 |
| w1_h5_s42_centered | 10701 | 0.5232 | 0.5145 | 0.4741 | 0.0491 | 0.4647 | 0.5293 | -0.0512 | 0.3513 | 29.6% | 384/384 |
| w0_h1_s43_original | 11925 | 0.5268 | 0.5156 | 0.5135 | 0.0133 | 0.5052 | 0.5187 | 0.0216 | 0.5578 | 63.8% | 32/160 |
| w0_h1_s43_signed | 11925 | 0.5290 | 0.5127 | 0.5107 | 0.0182 | 0.4912 | 0.5187 | 0.0076 | 0.6068 | 81.3% | 32/160 |
| w0_h5_s43_original | 11925 | 0.5371 | 0.5197 | 0.5404 | -0.0034 | 0.5436 | 0.5116 | 0.0846 | 0.4987 | 42.2% | 224/352 |
| w0_h5_s43_signed | 11925 | 0.5450 | 0.5256 | 0.5377 | 0.0074 | 0.5407 | 0.5116 | 0.0810 | 0.5302 | 48.9% | 256/384 |
| w1_h1_s43_original | 10705 | 0.5113 | 0.4956 | 0.4932 | 0.0181 | 0.4897 | 0.5206 | 0.0190 | 0.1867 | 10.7% | 32/160 |
| w1_h1_s43_signed | 10705 | 0.5133 | 0.4948 | 0.4941 | 0.0191 | 0.4883 | 0.5206 | 0.0171 | 0.1675 | 9.4% | 32/160 |
| w1_h5_s43_original | 10701 | 0.5155 | 0.5452 | 0.4363 | 0.0792 | 0.4882 | 0.5293 | -0.0702 | 0.6113 | 78.7% | 320/384 |
| w1_h5_s43_signed | 10701 | 0.5231 | 0.5494 | 0.4406 | 0.0825 | 0.4672 | 0.5293 | -0.0908 | 0.5550 | 66.8% | 352/384 |

| Run | Comparator | Paired AUC difference [95% block interval] |
|---|---|---|
| w0_h1_s42_original | price | +0.0009 [-0.0066, +0.0077] |
| w0_h1_s42_signed | original | -0.0030 [-0.0100, +0.0036] |
| w0_h1_s42_centered | signed | +0.0046 [-0.0035, +0.0134] |
| w0_h5_s42_original | price | +0.0178 [-0.0194, +0.0623] |
| w0_h5_s42_signed | original | -0.0079 [-0.0305, +0.0117] |
| w0_h5_s42_centered | signed | +0.0011 [-0.0030, +0.0046] |
| w1_h1_s42_original | price | +0.0192 [+0.0012, +0.0379] |
| w1_h1_s42_signed | original | -0.0045 [-0.0148, +0.0054] |
| w1_h1_s42_centered | signed | -0.0061 [-0.0129, +0.0001] |
| w1_h5_s42_original | price | +0.0097 [-0.0169, +0.0387] |
| w1_h5_s42_signed | original | +0.0010 [-0.0122, +0.0136] |
| w1_h5_s42_centered | signed | -0.0305 [-0.0607, -0.0025] |
| w0_h1_s43_signed | original | -0.0027 [-0.0066, +0.0017] |
| w0_h5_s43_signed | original | -0.0028 [-0.0278, +0.0268] |
| w1_h1_s43_signed | original | +0.0009 [-0.0028, +0.0041] |
| w1_h5_s43_signed | original | +0.0043 [-0.0151, +0.0214] |

| Window | Fit | Validation | Assessment |
|---|---|---|---|
| 0 | 2021-04-21 to 2022-04-19 | 2022-04-27 to 2022-07-08 | 2022-07-18 to 2022-12-30 |
| 1 | 2022-04-25 to 2023-04-25 | 2023-05-03 to 2023-07-14 | 2023-07-24 to 2023-12-20 |

## Files and verification

- Shared modelling updates are `src/models/gat_lstm.py` (reliability and centering), `src/models/train.py` (session IDs and configuration) and `src/models/attention.py` (session-aware explanations).
- The regression tests in `tests/test_signed_graph_messages.py` cover signed magnitude, zero-edge exclusion, isolated nodes, gradient flow, session-local centering, relative logits, and explanation parity. The model, attention, graph builder and splitter suites passed with them. **100 tests passed.** Ruff passed.
- This directory contains `study.py`, `summarize.py`, `verify.py`, `execution.log`, and this report. The `runs/` folder contains the manifest with source hashes, the exact executed source snapshot (the current runner has only import sorting changes), a 24-fit JSONL ledger, two fitted graph Parquets, 24 assessment Parquets and 24 checkpoint state dicts.
- Two CPU threads completed model fitting in about 162 seconds in total. Disk remained at about 10 GiB free. The verification script checked finite predictions and identical labels and keys within every window and horizon. It replayed the strongest checkpoint and matched the saved probabilities.
- To reproduce, run `.venv/bin/python -B -m experiments.gat_edge_weight_v1.study` in a fresh copy of the experiment directory with no pre-existing `runs/`. Run the corresponding `summarize` and `verify` modules against the saved outputs. Reproduce the selected checkpoint using its configuration in the ledger and the original input artifacts hashed in the manifest.

The component evidence is limited to two historical periods, the top-positive-peer correlation graphs that the original builder selected, and price-only signals. It supports neither a claim about full news and graph performance nor an investment claim. Official data, predictions, checkpoints, and published results were preserved. No Git operations were performed.
