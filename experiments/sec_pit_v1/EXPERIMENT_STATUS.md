# SEC point-in-time fundamental-data study

**Status: acquisition gate failed.** Zero fits and zero model-ready SEC rows. Zero paid charges.

## Local audit

- Historical 2019–2023 prices contain 199,591 ticker-days. The historical-member, valid 5d label population contains 127,979 rows, 148 tickers, and 1,253 dates.
- The matched original-attention prediction sets contain 11,925 rows across 109 tickers from July 18 to December 30, 2022, and 10,701 across 108 tickers from July 24 to December 20, 2023.
- The local earnings handoff documents 13 manually selected date-only windows and an earlier 1,737-window, 114-ticker Quant500 exercise based on 8-K cover dates. Neither establishes intraday public availability or complete issuer coverage. The Quant500 source and prepared earnings CSV were absent from the local raw data directory, and no prior model consumer of these fundamental inputs was found. [Coverage handoff](../../report/earnings_coverage_handoff.md) · [Local preparation](../../src/data/prepare_sec_earnings.py)
- No cohort-wide dated ticker-to-CIK master or usable longitudinal filing and fact snapshot was found. A current SEC ticker list alone cannot resolve old ticker aliases, share classes, mergers, or issuer changes.

## Source pilot and decision

Five official filing indexes for four issuers were reviewed and recorded in [filing_pilot.csv](filing_pilot.csv). They establish availability of historical filing detail and acceptance timestamps in the sample. They do not establish 2019–2023 coverage. The sample spans 10-K, 10-Q, Item 2.02 8-K, and a foreign issuer's 20-F. Apple and Meta show why filing date cannot replace acceptance time. None of the five records supplies dated public-dissemination evidence sufficient for this study's gate. Meta's own filing documents the FB-to-META transition on June 9, 2022. [Meta 10-K](https://www.sec.gov/Archives/edgar/data/1326801/000132680123000013/meta-20221231.htm)

The four pilot securities account for 468 of 11,925 assessment keys in 2022 (3.92%) and 424 of 10,701 in 2023 (3.96%). This is potential identity overlap only. Verified feature coverage is **0** in both windows. Microsoft provides a quarterly revenue and operating-income example. Apple's 10-K contains annual values that cannot be treated as standalone quarterly values. The first implementation excludes 20-F and 6-K until their event scope and accounting contexts are validated. [Microsoft 10-Q](https://www.sec.gov/Archives/edgar/data/789019/000095017023054855/msft-20230930.htm) · [Apple 10-K](https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924.htm)

An identifying project User-Agent with a genuine contact was unavailable. Only ordinary public filing pages and documentation were reviewed. No bulk API requests, model fits, or paid charges. [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) · [SEC fair-access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)

## Implemented and tested

[features.py](features.py) accepts an audited historical CIK for each stock-date key and explicit accession-level acceptance and public-by evidence. It uses the first session strictly after the later timestamp's Eastern calendar date. This deliberately delays even pre-open filings, so early closes cannot admit same-day information. The join preserves all keys and returns nulls where evidence is missing. Null records cannot be interpreted as zero filing events. Filing form, sessions since the latest supported event, last periodic report end, and quarterly operating margin carry accession provenance. Margin uses validated, consolidated USD revenue and operating income from the same accession, issuer, and quarter context. Annual, YTD, conflicting contexts, and unavailable values are excluded. Amendments enter the history only when public and cannot overwrite earlier observations. Raw EDGAR ingestion, completeness certification, and model-runner integration remain pending.

Tests (3 passed):

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider experiments/sec_pit_v1/test_features.py
```

## Best-model registry and resume

The professor fallback is original attention, 5 sessions, seed 42, fit April 21, 2021–April 19, 2022, validation April 27–July 8, 2022, assessment July 18–December 30, 2022. Genuine checkpoint inference reproduced all 11,925 probabilities exactly and rebuilt the 1,492 graph rows exactly. Accuracy 55.4801%, majority 51.1614%, MCC 0.111372, within-day AUC 0.543836. The isolated 32-unit challenger remains preserved in the [prior registry](../gat_regularization_v1/best_model_registry.json). No new stability or 0.63 claim is warranted.

- [Fallback checkpoint](../gat_edge_weight_v1/runs/w0_h5_s42_original.pt), SHA-256 `708066a4bbec82a53f78ad35ffee8fb61c894efa715f58cb315ca79aaf749d1e`
- [Fallback predictions](../gat_edge_weight_v1/runs/w0_h5_s42_original.parquet), SHA-256 `7c9f050dff9933e2210415105e132984d2901bb571c4c01ec8572645e5ca4b16`

```sh
.venv/bin/python -B -m experiments.historical_demo_v1.replay development
```

Resume with a real institutional or project contact for the SEC User-Agent, or a provenance-preserved cache from the team. Collect a small representative set of issuer submissions and historical accession facts (including archive pages referenced by submissions), with dated dissemination evidence and verified security-to-issuer identity intervals. Establish complete filing coverage and warm-up history before using event absence as a feature. Report distinct issuer, accession and matched stock-date coverage for both windows before choosing whether to train. Fit any imputation and scaling on training rows alone. Keep the original labels, assessment rows, graph rules and embargoes. A filing timestamp is an EDGAR event, and the earliest timestamped earnings release is a separate research input.
