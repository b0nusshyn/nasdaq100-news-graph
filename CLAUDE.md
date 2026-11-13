# CLAUDE.md

Academic project: predict NASDAQ-100 daily price direction (up/down, 1d and 5d) by propagating
FinBERT news signals through a company graph with a GAT on top of a per-company LSTM.
Hard deadline **Sat 21 Nov 2026**; feature freeze **Thu 19 Nov 2026**.

- `README.md` — research question, method, baselines, setup commands. Read before any modelling or evaluation work.
- `SCHEMA.md` — the data contract for every Parquet file in `data/`. Read before writing or reading any Parquet file; change it only with the whole team's agreement.
- `CHECKLIST.md` — tasks, owners, deadlines and hand-offs. Read before picking up or rescheduling work.

## Ownership

Each stage writes one artifact that the next stage consumes. Edit inside the owning directory
and keep its output matching `SCHEMA.md`.

| Role | Directory | Produces |
|---|---|---|
| Data | `src/data/` | `membership`, `prices`, `labels` |
| NLP | `src/nlp/` | `article_tickers`, `news_features` |
| Modelling | `src/graph/`, `src/models/` (except `baselines/`) | `edges`, `predictions/gat_lstm` |
| Evaluation | `src/eval/`, `src/models/baselines/` | splitter, metrics, backtest, baseline predictions, `results/` |
| Everyone | `src/core/` | shared paths, schema, I/O, fixtures; change with the team's agreement |

- Read and write artifacts only through `src.core.io` (`read_table` / `write_table`); it validates against `SCHEMA.md`.
- The data directory is `$NDX_DATA_DIR` (shared Google Drive; default `data/`). Set it to `tests/fixtures` to code against synthetic data until the real upstream artifact lands.
- Skeleton functions call `todo(<issue>, ...)`. Replacing one closes that part of the issue.
- Keep the light dependency set (`requirements-dev.txt`) enough to import `src.core` and run every CLI's `--help`: import torch, transformers and other heavy packages inside the modules or functions that need them. CI runs `ruff check .` and `pytest` on that set.
## Leakage invariants

The result is only worth reporting if none of these leak. Every change touching data, features
or training keeps all of them true, with a test where possible:

- **Survivorship**: a ticker is in the sample only on days its `in_index` flag is true (membership rebuilt from Wikipedia history, never today's list).
- **Look-ahead (news)**: articles timestamped after 16:00 ET belong to the next trading day.
- **Look-ahead (features)**: indicators, `corr` edge weights, `comention` weights and normalisation statistics are fitted on the training fold only.
- **Walk-forward**: splits are contiguous in time, with an embargo equal to the horizon between train and test. Random splits are never used.
- **Sparse news**: `has_news = 0` means "no coverage" and stays distinct from neutral sentiment.

Baseline 6 (LSTM + news, no graph) is the proposed model with the GAT removed, sharing the same
encoder code. It is the comparison the project's conclusion rests on; keep the two in lockstep.

## Issue tracker

GitHub Issues on `b0nusshyn/nasdaq100-news-graph`, via the `gh` CLI. One issue per `CHECKLIST.md` task.

- **Task ID** `person.task` starts every issue title (`[1.3] …`) and every checklist line: 0 = everyone, 1 = Data, 2 = NLP, 3 = Modelling, 4 = Evaluation. A new task takes the next free number for its person, in both places.
- **Milestone** = the deadline day (`Due Sat 14 Nov`, …). The due time is in the issue body.
- **Role label**: `data`, `nlp`, `modelling`, `evaluation`, or `team` for shared tasks.
- **`handoff`**: another role is waiting on this issue's output; it takes priority.
- **`blocked`**: at least one issue in the body's **Blocked by** list is still open. When closing an issue, check every issue listing it under **Blocked by** and remove `blocked` from any whose blockers are now all closed.

Pick work from open issues without `blocked`, earliest milestone first. When a date in an issue
changes, update `CHECKLIST.md` in the same change.
