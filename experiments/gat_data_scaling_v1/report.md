# Does more historical data improve the five-day GAT?

## Data utilization

The historical loader uses the official price and label files and verifies their labels and close-to-close return magnitudes against exact future Nasdaq closes in memory. Only 2019–2023 signal dates contribute labelled examples. The available 2018 prices supply earlier indicator warm-up. The 2019–2023 price panel covers all 1,258 exchange sessions and contains **199,591 observed ticker-days from 172 securities**. Of those, **128,519 are historical members across 148 securities** and **71,072 are nonmembers**, which are excluded as supervised targets. An independent interval join found no observed member flag mismatches and five expected-but-absent member prices (XLNX, February 14–18, 2022). Those five are never fabricated. There are 128,411 eligible 1d labels and 127,979 eligible 5d labels. The 108 and 540 missing-label member counts include the final horizon boundary and unavailable exact target closes. [Audit](runs/audit.json) · [Missing keys](runs/missing_member_keys.json)

| Five-day utilization | 2022 development window | 2023 development window |
|---|---:|---:|
| Previous rolling weight-fit sessions | 252 | 252 |
| Previous rolling weight-fit rows | 25,654 | 25,617 |
| Available pre-cutoff fit rows | 85,194 | 111,117 |
| Additional eligible fit rows | 59,540 | 85,500 |
| Fixed validation rows | 5,100 | 5,050 |
| Fixed assessment rows | 11,925 | 10,701 |

The trainer keeps incomplete price windows and encodes their scaled missing values as zero with a completeness flag. Graph construction restricts available neighbors using fit-period returns, and missing peers do not remove a supervised row. Isolated nodes retain their own encoder path. The ledger tracks incomplete rows retained and labelled rows whose ticker has no fit-graph edge. A ticker can also be isolated on individual sessions after membership masking. Some out-of-index price observations can contribute causal indicator warm-up and observed returns for a ticker that was a fit-period member. They do not become training targets. [Utilization audit](runs/utilization.json)

## Design

The four windows nest backward from each original fit cutoff, retaining the most recent 25%, 50%, 75% or 100% of the available sessions from 2019 onward. The 2022 fit cutoff is April 19, with validation April 27–July 8 and assessment July 18–December 30. The 2023 fit cutoff and exact ranges are in the run ledger. Validation and assessment, their original labelled rows, and five-session embargoes stayed fixed. The two seeds and four fractions were frozen in advance. All tested fractions met the unchanged minimum overlap for a usable correlation graph. The five-day horizon was the primary focus, and no new one-day models were fit.

Each cell refits scaler and residual correlation graph on that cell's fit days, with the same original-attention configuration (64 units, four heads, 20-day price sequence, Adam .001, weight decay .0001, eight dates per batch, BCE, epoch AUC checkpoint and earlier MCC threshold). Per epoch computation grows with the available rows. This tests the end-to-end effect of adding older history under consistent training rules. Sample age, observed universes, fitted graph/scaler and optimizer updates cannot be disentangled from this core curve. It is not an equal-update causal estimate of sample count. The previously inspected windows are retrospective development. No live or 2026 inputs were used.

## Completed learning curves

[Accuracy, AUC and MCC learning curves as SVG](learning_curves.svg) · [All core runs, dates, row counts, graph edges, update counts, train/validation metrics and date-block contrasts](tables.md)

| Assessment | Fit sessions, 25 / 50 / 75 / 100% | Training observations, 25 / 50 / 75 / 100% | AUC, seed 42 | AUC, seed 43 |
|---|---|---|---|---|
| 2022 | 208 / 416 / 624 / 831 | 21,166 / 42,464 / 63,883 / 85,194 | .5267 / .5312 / .4715 / .4800 | .5277 / .5433 / .4746 / .4902 |
| 2023 | 272 / 543 / 815 / 1,086 | 27,657 / 55,249 / 83,219 / 111,117 | .5030 / .4792 / .4833 / .4879 | .4515 / .4821 / .4880 / .4623 |

Validation was May 3–July 14 for the 2023 case. Assessment was July 24–December 20, 2023. Full fit ended April 25, 2023. Every scaling start date and graph size are in the linked table. No size was rejected. Actual 2022 full-fit graph edges rose to 1,884 versus 1,494 at 25%; 2023 rose to 1,966 from 1,490. The 2022 full history retained 137 incomplete windows and 147 fit observations for securities with no fit-graph peer. The 2023 full history also kept those categories where applicable. Membership changes were respected throughout.

**More data did not consistently help.** In 2022 the gain from 25% to 50% reversed at 75% and remained weak for 100%. Full history lost to the rolling 252-session controls by 0.0638 AUC at seed 42 and 0.0502 at seed 43. Their 10-session paired intervals excluded zero. The strongest scaling cell was 2022 seed 43 at 50%, with 54.26% accuracy, 51.16% majority, 0.0838 MCC and 0.5433 AUC. Its +0.0029 AUC contrast versus the same-seed 252-session original crossed zero, and it was below the seed-42 fallback. In 2023 the ranking shifted irregularly by seed. Longer histories improved the particularly weak seed-43 252-session comparator in some cells, while every 2023 scaling fit had accuracy below majority and negative MCC. The rolling controls did not establish stable 2023 skill either.

An earlier-validation-only window-choice diagnostic selected 25% in both 2022 seeds and 50% in both 2023 seeds. This yielded 0.5267/0.5277 and 0.4792/0.4821 assessment AUC respectively. The window rule is a diagnostic replay of a fixed selection criterion and was applied after a full retrospective curve. It is not an independent selection trial. The variation in supervision supports distribution shift as a plausible explanation. Annual five-day Up prevalence was 58.36% in 2019, 56.77% in 2020, 54.24% in 2021, 47.33% in 2022 and 55.42% in 2023. Mean absolute five-day return also rose from 2.93% in 2019 to 4.74% in 2022. Membership, marginal returns, graph structure, and training steps all change with history length. The curve cannot identify which caused the degradation. In 2023 validation often looked better while assessment remained weak. The train-to-assessment gaps are in the results table. [Year-by-year label audit](runs/annual_population.csv)

## Follow-up: retain history and downweight its gradients

The curve justified a distinct recency experiment. All full-history rows, transformations and graph estimation stayed fixed. The only intervention was weighting the training BCE by exponentially decreasing age with a 252-session half-life. This checks whether emphasizing recent updates addresses the long-history weakness. Earlier validation still selected the checkpoint and threshold using the existing unweighted criteria. The rule did not choose a new half-life based on an assessed window. [Follow-up protocol](recency_protocol.md)

| Assessment / seed | AUC, full unweighted | AUC, full recency | Recency accuracy | Majority | MCC |
|---|---:|---:|---:|---:|---:|
| 2022 / 42 | .4800 | .5195 | 48.89% | 51.16% | .0053 |
| 2022 / 43 | .4902 | .5114 | 49.05% | 51.16% | .0144 |
| 2023 / 42 | .4879 | .4995 | 48.37% | 52.93% | −.0490 |
| 2023 / 43 | .4623 | .4906 | 45.68% | 52.93% | −.0720 |

Recency weighting improved ranking in all four comparisons against unweighted full history. Only 2022 seed 42 had a paired date-block interval excluding zero (+0.0394, [0.0032, 0.0803]). All four accuracy results were under their matched majority. Recency still trailed both rolling controls in 2022, and both 2023 recency fits stayed below 0.50 AUC. AUC and class decision metrics therefore lead to different conclusions. Retaining older observations and reducing their gradient influence offered a partial ranking recovery, without a useful replacement for the current control. [Four complete recency runs with intervals](recency_tables.md)

## Decision and next evidence

**Twenty fits completed**, 16 for the curve and four for the weighting follow-up. Both used only the existing historical archive. Some eligible securities absent from the shorter fit windows naturally entered the expanded fit through earlier historical membership. No nonmember training targets, synthetic rows, new sources or new cohort rules were introduced. The results give no support to acquiring more of the same price-only data solely to improve these models. They also do not establish that older information always hurts. Further experiments with age windows or weighting on the same inspected dates would consume more selection degrees of freedom. A newly qualified assessment period or independently dated predictive inputs would be more informative. Examples include filing acceptance timestamps paired with as-filed fundamentals and verified earnings announcement and expectation timestamps. Existing reconstructed news dates do not supply such point-in-time proof.

Neither numerical target was reached. The highest new accuracy and AUC both came from 2022 416 sessions, seed 43: 54.26% and 0.5433, above its majority baseline but without an AUC gain distinguishable from zero on that same seed. The previous 32-unit 2022 seed-42 checkpoint retains the highest individual five-day AUC at 0.547553, with 55.96% accuracy and 0.1354 MCC. No new model merits promotion. Keep the separately verified original seed-42 2022 fallback, 55.4801% accuracy and 0.543836 AUC, for the demonstration. It again matched all archived output probabilities exactly. The unchanged one-day task remains outside the scope of the new fits. [Updated registry](best_model_registry.json) · [Best new scaling checkpoint](runs/w0_p050_s43.pt) · [Its predictions](runs/w0_p050_s43.parquet) · [Fallback provenance](fallback_recheck.json)

## Reproduction, files and tests

All changes are isolated here. New code includes `study.py`, `audit_details.py`, `recency.py`, `plot.py`, `verify.py`, `registry.py`, and `test_scaling.py`. New documentation includes the two protocols, experiment status, report, SVG and tables. The two execution logs and fallback replay capture execution evidence. `runs/` includes the price/membership audits, manifest, ledger, eight graph tables, sixteen assessment Parquets and sixteen checkpoints. `recency_runs/` has its own manifest, ledger, four predictions and four checkpoints. The ledgers record fitted configurations and hashes. Verification loaded and ran all 20 saved states and matched archived probabilities within 1e-7. **90 tests passed and Ruff passed.** Summed recorded fit time was about 446 seconds over 20 fits. Disk stayed around 8.8 GiB free. No Git operations, notebook work, official edits or 2026 scoring took place.

From the repository root, verify artifacts and regenerate the learning curve with:

```sh
.venv/bin/python -B -m experiments.gat_data_scaling_v1.audit_details
.venv/bin/python -B -m experiments.gat_data_scaling_v1.verify
.venv/bin/python -B -c 'from experiments.gat_data_scaling_v1.verify import main; main(decay=True)'
.venv/bin/python -B -m experiments.gat_data_scaling_v1.plot
.venv/bin/python -B -m experiments.historical_demo_v1.replay development
```

For fresh training use an isolated sibling copy without existing `runs/` or `recency_runs/` directories, and run the `study` module before the `recency` module. Neither command is needed for replay.
