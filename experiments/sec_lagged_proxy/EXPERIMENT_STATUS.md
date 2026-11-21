# SEC_LAGGED_PROXY status

**Training readiness: FAIL.** This is a sourced offline pilot, with no training, no paid charges, and no changes to the strict policy.

## Source and identity audit

The 2019–2023 five-session historical-member dataset has 127,979 keys across 148 tickers. The full [issuer inventory](pilot_2/issuer_inventory.csv) marks 7 ticker keys (6 CIKs) with reviewed partial intervals and 141 as unresolved. No ticker is certified across the full 2019–2023 period. The seven keys have an issuer attached on 1,383 rows (1.08%). The inventory exposes repository rename links, share classes and corporate-action CIK leads as aids to future review. Unresolved includes cases not yet researched as well as cases needing ambiguity or corporate-action review. Confirmed ambiguous and confirmed non-SEC issuers could not be counted separately. No unresolved issuer was classified as a non-filer. Dates before a reviewed interval remain unresolved. The sample treats GOOG and GOOGL as distinct stock keys with shared issuer facts, uses the documented FB-to-META ticker change, preserves Splunk's issuer through the end of the study, and keeps ASML's 20-F separate. The mapping does not use current ticker/CIK matches to backfill old records.

The [filing ledger input](filings.csv) contains seven reviewed original accessions, including five supported periodic filings, one Item 2.02 8-K, and one 20-F. The latter two are excluded from the periodic filing arm. All seven have accession, CIK, official date, observed acceptance timestamp, and source link. Exact dissemination time is independently verified for zero. The SEC describes a variable and unguaranteed lag. [SEC timestamp guidance](https://www.sec.gov/about/webmaster-frequently-asked-questions#edgar-timestamps)

[Original facts](facts.csv) contains two comparative quarter contexts, six values total, from Microsoft's October 24, 2023 10-Q, in USD millions. Both contexts are used only when the 2023 accession is eligible. They are not retroactively assigned to 2022. The facts were transcribed from the original accession's consolidated statement, with that source, period, scope, and units retained. The 2023 quarter yields about 12.76% year-over-year revenue growth, 47.59% operating margin and a 4.41-percentage-point year-over-year increase in net margin. No alternate taxonomy, cumulative context or currency was substituted. [Microsoft filing](https://www.sec.gov/Archives/edgar/data/789019/000095017023054855/msft-20230930.htm)

| Source | Use |
|---|---|
| SEC submissions history, including referenced older history files | Efficient accession, form, official date and acceptance metadata. Current entity aliases require historical corroboration. |
| Original accession detail and original filing | Best source for timestamp, security class and as-filed statement verification. Fetch selectively after indexing. |
| Quarterly Financial Statement Data Sets | Efficient as-filed accession index and primary-statement candidates, subject to extraction errors and incomplete metadata. December 2024 reprocessing changed selection and introduced segments. Filter by accession, context, currency and scope, then verify against original filings. |
| Original accession XBRL instance and rendered statement | Best context-level tie-breaker for original values and dimensional ambiguity. Company Facts entries can help locate values when kept by accession. Taking the latest value for a fiscal period can import a revision. |

Twenty 2019–2023 quarterly ZIPs are listed at approximately 1.91 GB compressed in total (78–115 MB each), excluding pre-2019 warm-up and extraction. Prefer issuer submissions first and stream one quarterly ZIP if needed. The new local indexer reads the SUB member without unzipping to disk and reports the cached archive hash. It yields review candidates and does not certify facts. SEC_USER_AGENT was absent. No automated network requests or archive downloads were attempted. Only public SEC filings and documentation were accessed. [SEC API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) · [SEC statement data](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets) · [SEC fair access](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)

## Timing and coverage

Use the later of the official filing date and Eastern acceptance date, then wait for two full subsequent exchange sessions to finish. The feature is eligible on the third session. The five-complete-session run is sensitivity only. Neither lag demonstrates historical dissemination or was selected from model assessment scores. The staging policy caps propagation after 90 sessions, preserves missing features, and marks all filing histories incomplete. The cap limits carry-forward from a known sparse sample. It does not certify that an omitted filing would leave its values unchanged.

| Window | Original rows | Mapped issuer rows | Strict verified rows | Two-session filing rows | Two-session fundamentals rows | Five-session filing / fundamentals |
|---|---:|---:|---:|---:|---:|---:|
| 2022 5d | 11,925 | 161 | 0 | 41 (0.34%) | 0 | 38 / 0 |
| 2023 5d | 10,701 | 574 | 0 | 38 (0.36%) | 38 (0.36%) | 35 / 35 |

The 2022 covered rows represent 41 dates and one ticker, AAPL, all 10-K. The 2023 covered rows represent 38 dates and one ticker, MSFT, all 10-Q. All supported assessment rows are Information Technology using the repository GICS classification for audit only. Under the two-session lag the full panel has 403 filing-feature rows across 5 stock keys and 196 dates, including 39 fundamental-feature rows. Full-panel filing coverage by year was 0 of 25,941 in 2019, 0 of 26,051 in 2020, 0 of 25,685 in 2021, 41 of 25,562 in 2022, and 362 of 24,740 in 2023. The assessment missingness breakdown is 11,764 unresolved issuer-date rows and 120 mapped rows without a recent verified periodic accession for 2022, and 10,127 and 536 for 2023. All original assessment keys and labels were checked for equality with the official label file. Both policies are below the preferred 25% coverage threshold and concentrated in one sector. A neural or cheap predictive pilot would primarily measure a handful of filing windows or issuer identity. No fits, model scores, cross-seed conclusions or 0.63 claims.

## Reproduce

Run from repository root. The build expects a new output directory and preserves prior artifacts.

```sh
.venv/bin/python -B -m experiments.sec_lagged_proxy.build --output experiments/sec_lagged_proxy/reproduction_2
.venv/bin/python -B -m experiments.sec_lagged_proxy.build --lag 5 --output experiments/sec_lagged_proxy/reproduction_5
.venv/bin/python -B -m pytest -q -p no:cacheprovider experiments/sec_lagged_proxy/test_build.py experiments/sec_pit_v1/test_features.py
.venv/bin/python -B -m experiments.historical_demo_v1.replay development
```

Six tests passed. They cover both policies, market holidays and early closes, use of the later timestamp, missing timestamp and timezone handling, keeping amendments off earlier rows, accession and accounting-context checks, unknown issuers and key preservation, and streaming a local quarter-index fixture.

Original-attention five-session seed 42 GAT-LSTM remains the demonstration fallback. Genuine checkpoint inference reproduced all 11,925 archived probabilities exactly. Accuracy 55.4801%, matched majority 51.1614%, within-day AUC 0.543836, MCC 0.111372. [Checkpoint](../gat_edge_weight_v1/runs/w0_h5_s42_original.pt) · [Predictions](../gat_edge_weight_v1/runs/w0_h5_s42_original.parquet) · [32-unit prior checkpoint](../gat_regularization_v1/runs/w0_h5_s42_small.pt)

## Resume and artifact inventory

Provide a real project contact through SEC_USER_AGENT, or a provenance-preserved local SEC cache. Start with audited historical security/issuer spans for representative securities across sectors. Expand via submissions history and original accessions, including filing history before the first prediction date. Use the quarterly archive index only as needed. Recompute coverage on the full keyed panel and compare alternate lags on chronological validation only. Exact historic dissemination remains unavailable for the strict policy unless separately evidenced.

New files are the three input CSVs (`identities.csv`, `filings.csv`, `facts.csv`), `build.py`, `index_quarter.py`, `test_build.py`, this report, and six outputs in each of `pilot_2/` and `pilot_5/`: `features.parquet`, `accessions.csv`, `assessment_coverage.parquet`, `sector_coverage.csv`, `issuer_inventory.csv`, `manifest.json`. No existing files were edited. No notebooks or Git operations. The 2026 evaluation remains locked.
