# Checklist

Task breakdown and deadlines for the NASDAQ-100 news-propagation project.

**Final deadline: Saturday 21 November 2026**
**Start: Friday 13 November 2026** (9 days, inclusive)

**Required submission items** (course brief): the **presentation file** (0.8) and the **demo**
(1.9 → 1.12). Neither is optional or a stretch goal.

Every deadline is **23:59 on the date shown** unless a time is given. Tick a box only when the
output is pushed to `main` and the next person in the chain can use it.

---

## Team

Task IDs are **person.task**: `1.3` is Data's third task. `0.x` tasks belong to everyone.
GitHub issue titles start with the same ID.

| # | Role | Member | Owns |
|---|---|---|---|
| 1 | Data | Chotima | `src/data/` · `prices.parquet`, `labels.parquet` |
| 2 | NLP | Kornchawal | `src/nlp/` · `news_features.parquet` |
| 3 | Modelling | Shayanis | `src/graph/`, `src/models/` · `edges.parquet`, GAT + LSTM |
| 4 | Evaluation | Nonlada | `src/eval/` · baselines, walk-forward, backtest, results |

---

## Critical hand-offs

These are the dates the rest of the plan depends on. If one slips, tell the team the same day.

| Due | Task | Artifact | From → To |
|---|---|---|---|
| Fri 13 Nov | 0.2, 0.5 | `SCHEMA.md` frozen + synthetic fixtures for every Parquet file | All → All |
| Sat 14 Nov | 4.1 | Walk-forward splitter (with embargo) | Evaluation → Modelling |
| Sun 15 Nov | 1.3, 1.4 | `prices.parquet`, `labels.parquet` | Data → Modelling, Evaluation |
| Sun 15 Nov | 2.4 | Article → ticker matches (pre-FinBERT) | NLP → Modelling (co-mention edges) |
| Mon 16 Nov | 2.6 | `news_features.parquet` | NLP → Modelling, Evaluation |
| Tue 17 Nov | 3.5 | `edges.parquet` (all three edge types) | Modelling → Evaluation |
| Wed 18 Nov | 3.7 | Shared LSTM encoder used by both the proposed model and Baseline 6 | Modelling → Evaluation |
| Thu 19 Nov | — | **Feature freeze** — no new experiments after this | All |

The synthetic fixtures are what let everyone work in parallel from day one: each person codes
against a small fake Parquet file that matches the schema, then swaps in the real one when it
lands.

---

## 0 · Day 0 — everyone

- [x] **0.1** · **Fri 13 Nov** — Kickoff meeting: confirm roles, fill in names above and in `README.md` · [#1](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/1)
- [x] **0.2** · **Fri 13 Nov** — Freeze `SCHEMA.md` (column names, dtypes, keys, date convention, 16:00 ET cutoff rule) · [#2](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/2)
- [x] **0.3** · **Fri 13 Nov** — Create repo skeleton (`src/data`, `src/nlp`, `src/graph`, `src/models`, `src/eval`, `configs/`, `notebooks/`, `results/`), `.gitignore` for `data/` · [#3](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/3)
- [x] **0.4** · **Fri 13 Nov** — Add `requirements.txt`; confirm PyTorch Geometric installs locally or on Colab for every member · [#4](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/4)
- [x] **0.5** · **Fri 13 Nov** — Each owner commits a synthetic fixture for their Parquet file under `tests/fixtures/` · [#5](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/5)

---

## 1 · Data — Chotima

- [ ] **1.1** · **Sat 14 Nov** — `build_index_history`: scrape Wikipedia NASDAQ-100 changes table, reconstruct daily membership 2019–2023 · [#6](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/6)
- [ ] **1.2** · **Sat 14 Nov** — `fetch_prices`: download OHLCV via `yfinance` for every ticker that was ever a member (2018-01-01 → 2023-12-31, extra year for indicator warm-up) · [#7](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/7)
- [ ] **1.3** · **Sun 15 Nov** — Add `in_index` flag; write `prices.parquet` ✱ hand-off · [#14](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/14)
- [ ] **1.4** · **Sun 15 Nov** — `build_labels`: `y_1d`, `y_5d` on trading-day calendar; write `labels.parquet` ✱ hand-off · [#15](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/15)
- [ ] **1.5** · **Mon 16 Nov** — Technical indicators for the LSTM input (returns, volatility, RSI, MACD, volume z-score), computed without look-ahead · [#19](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/19)
- [ ] **1.6** · **Mon 16 Nov** — Data-quality checks: missing days, delisted tickers, splits/adjustments, label class balance per year · [#20](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/20)
- [ ] **1.7** · **Tue 17 Nov** — Bias tests: assert no ticker appears before it joined the index; assert no label uses future prices beyond its horizon · [#26](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/26)
- [ ] **1.8** · **Tue 17 Nov** — Investing.com earnings-date pull; report FNSPID coverage gaps to NLP · [#27](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/27)
- [ ] **1.9** · **Wed 18 Nov** — ⚑ **Required submission item.** Demo, first runnable version: `notebooks/demo.ipynb` end-to-end on a small slice (load → features → model → prediction), as soon as the model trains end to end (3.6) · [#34](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/34)
- [ ] **1.10** · **Thu 19 Nov** — Reproducibility run: fresh clone, follow `README.md` Setup exactly, fix anything that breaks · [#40](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/40)
- [ ] **1.12** · **Thu 19 Nov** — ⚑ **Required submission item.** Demo polished by the feature freeze: runs top to bottom on the real data, uses the final model, shows one attention trace (2.10); no edits after this except fixes · [#60](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/60)
- [ ] **1.11** · **Fri 20 Nov** — Write Dataset & bias-handling section of the report · [#46](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/46)

## 2 · NLP — Kornchawal

- [ ] **2.1** · **Sat 14 Nov** — Download FNSPID from HuggingFace; filter to NASDAQ-100 tickers (ever-members) and 2019–2023; save `data/raw/fnspid_filtered.csv` · [#10](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/10)
- [ ] **2.2** · **Sat 14 Nov** — Estimate FinBERT runtime on the filtered set; if > 6 GPU-hours, agree a reduction (headlines only, dedup, cap per ticker-day) with Modelling · [#11](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/11)
- [ ] **2.3** · **Sun 15 Nov** — Ticker / entity matching: map each article to one or more tickers (symbol + company-name aliases) · [#16](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/16)
- [ ] **2.4** · **Sun 15 Nov** — Timestamp alignment: convert to ET, articles after 16:00 → next trading day ✱ hand matches to Modelling · [#17](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/17)
- [ ] **2.5** · **Mon 16 Nov** — FinBERT encoding (Colab GPU): sentiment, confidence, 768-d embedding per article · [#23](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/23)
- [ ] **2.6** · **Mon 16 Nov** — Aggregate to (`date`, `ticker`), add `has_news`; write `news_features.parquet` ✱ hand-off · [#24](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/24)
- [ ] **2.7** · **Tue 17 Nov** — Coverage report: share of ticker-days with news, by year and by sector (feeds the "sparse news" argument) · [#30](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/30)
- [ ] **2.8** · **Tue 17 Nov** — Spot-check 50 random matches by hand; fix any systematic mismatch (e.g. `META` vs `FB`, `GOOG`/`GOOGL`) · [#31](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/31)
- [ ] **2.9** · **Wed 18 Nov** — Interpretability tooling: extract GAT attention weights and trace a prediction back to source article + edge type · [#37](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/37)
- [ ] **2.10** · **Thu 19 Nov** — Produce 2–3 worked interpretability examples (e.g. NVDA article → ARM prediction) as figures in `results/` · [#43](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/43)
- [ ] **2.11** · **Fri 20 Nov** — Write News encoding & Interpretability sections of the report · [#48](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/48)

## 3 · Modelling — Shayanis

- [ ] **3.1** · **Sat 14 Nov** — `sector` edges from GICS classification · [#8](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/8)
- [ ] **3.2** · **Sat 14 Nov** — Model skeleton on synthetic fixtures: LSTM encoder → GAT → dense → P(up); confirm tensor shapes end to end · [#9](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/9)
- [ ] **3.3** · **Mon 16 Nov** — `comention` edges from NLP's article → ticker matches (weight = co-mention count, training window only) · [#21](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/21)
- [ ] **3.4** · **Mon 16 Nov** — `corr` edges from return correlation, **recomputed per training fold** using Evaluation's splitter · [#22](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/22)
- [ ] **3.5** · **Tue 17 Nov** — Write `edges.parquet` ✱ hand-off · [#28](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/28)
- [ ] **3.6** · **Tue 17 Nov** — Training loop + `configs/gat_lstm.yaml`; normalisation fitted on training fold only · [#29](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/29)
- [ ] **3.7** · **Wed 18 Nov** — Factor the LSTM encoder so Baseline 6 is the same model with the GAT removed ✱ hand-off to Evaluation · [#35](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/35)
- [ ] **3.8** · **Wed 18 Nov** — Full walk-forward training run, both horizons (1d, 5d); save predictions per fold · [#36](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/36)
- [ ] **3.9** · **Thu 19 Nov** — Ablations: drop each edge type in turn; drop news features · [#41](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/41)
- [ ] **3.10** · **Thu 19 Nov** — Expose attention weights per prediction for NLP's interpretability work · [#42](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/42)
- [ ] **3.12** · **Thu 19 Nov** — Fine-tuning pass (course brief: "Re-run + fine-tune"): log at least 5–6 GAT + LSTM runs in `results/tables/tuning_gat_lstm.md` with columns learning rate, hidden dim, attention heads, dropout, validation MCC; score on the validation split inside each training fold, never the test windows; re-run 3.8 with the best config; include the table in the report · [#58](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/58)
- [ ] **3.11** · **Fri 20 Nov** — Write Method (graph + model) section of the report · [#47](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/47)

## 4 · Evaluation — Nonlada

- [ ] **4.1** · **Sat 14 Nov** — Walk-forward splitter with a 5-trading-day embargo (longest horizon), the same for every horizon ✱ hand-off to Modelling · [#12](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/12)
- [ ] **4.2** · **Sat 14 Nov** — Metrics module: accuracy, F1, MCC, AUC; unit-test on the synthetic fixtures · [#13](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/13)
- [ ] **4.3** · **Sun 15 Nov** — Backtest module: long/short from P(up), cumulative return, Sharpe, max drawdown, net of transaction costs · [#18](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/18)
- [ ] **4.4** · **Mon 16 Nov** — Baselines 1–3: buy-and-hold, ARIMA, logistic regression · [#25](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/25)
- [ ] **4.5** · **Tue 17 Nov** — Baseline 4: XGBoost · [#32](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/32)
- [ ] **4.6** · **Tue 17 Nov** — Baseline 5: LSTM, price only · [#33](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/33)
- [ ] **4.7** · **Wed 18 Nov** — Baseline 6: LSTM + news, no graph (using Modelling's shared encoder) · [#38](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/38)
- [ ] **4.8** · **Wed 18 Nov** — `run_baselines --all` runs every baseline on the same folds as the proposed model · [#39](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/39)
- [ ] **4.9** · **Thu 19 Nov** — Significance test: proposed model vs Baseline 6 (paired test across folds, e.g. bootstrap on MCC) · [#44](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/44)
- [ ] **4.10** · **Thu 19 Nov** — `compare.py`: results table + figures (equity curves, metric per fold) into `results/` · [#45](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/45)
- [ ] **4.13** · **Thu 19 Nov** — Fine-tuning pass for the LSTM baselines (5 and 6): log at least 5–6 runs in `results/tables/tuning_baselines.md` with columns learning rate, hidden dim, attention heads (n/a), dropout, validation MCC, using the same validation split and a similar budget to 3.12 so Baseline 6 is tuned as fairly as the proposed model; include the table in the report · [#59](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/59)
- [ ] **4.11** · **Fri 20 Nov** — Fill in the Results table and ablation table in `README.md` · [#49](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/49)
- [ ] **4.12** · **Fri 20 Nov** — Write Evaluation & Results sections of the report · [#50](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/50)

---

## 0 · Final stretch — everyone

- [ ] **0.6** · **Fri 20 Nov, 12:00** — All report sections drafted; merge into one document · [#51](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/51)
- [ ] **0.7** · **Fri 20 Nov** — Cross-review: each member reads one other member's section and code · [#52](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/52)
- [ ] **0.8** · **Fri 20 Nov** — ⚑ **Required submission item.** Presentation file and one rehearsal · [#53](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/53)
- [ ] **0.9** · **Sat 21 Nov, 12:00** — Final fixes; update `README.md` status from "in development" to final · [#54](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/54)
- [ ] **0.10** · **Sat 21 Nov, 18:00** — **Submit.** Tag the release (`git tag v1.0`) · [#55](https://github.com/b0nusshyn/nasdaq100-news-graph/issues/55)

The gap between 18:00 and midnight on 21 Nov is buffer — do not plan work into it.

---

## If something slips

- **News pipeline late** → Modelling and Evaluation keep working on synthetic `news_features`; Baselines 1–5 don't need news at all.
- **FinBERT too slow** → encode headlines only, or sample; record the reduction in the report.
- **GAT won't train by Wed 18 Nov** → drop to one horizon (1d) and one edge type (`comention`), then add the others back if time allows.
- **Proposed model doesn't beat Baseline 6** → that is a valid result. Report it honestly; the research question is answered either way.
