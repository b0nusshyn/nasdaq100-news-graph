# SEC coverage completion checkpoint

**Decision: training gate failed.** No new fits. No paid charges. No automated SEC requests, archive downloads, or SEC API calls. SEC_USER_AGENT was absent. No cached submissions, Company Facts, or quarterly statement ZIPs were found in the workspace. Public search results indexing official SEC filing pages supplied a bounded manual evidence expansion.

Four historical issuer mappings were added for Costco, PepsiCo, Comcast Class A, and Gilead. Eleven of the 148 ticker keys now have reviewed partial intervals covering ten issuers. The other 137 remain unresolved. The interval starts are bounded by historical evidence and do not imply full-history verification. The inventory retains local historical aliases, share-class groups, and corporate-action leads separately from resolved CIKs. Missing records do not establish non-reporting status. [Identity additions](identity_additions.csv) · [Full inventory](run_02/identity_inventory.csv)

Seven original domestic periodic filings were added with official dates and Eastern acceptance stamps. The combined ledger has 14 accessions, 12 supported periodic accessions, and two earlier exclusions (one Item 2.02 8-K and one 20-F). Four comparable quarter contexts containing 12 consolidated numeric values were added from original PepsiCo 10-Qs, giving six contexts and 18 values in the combined facts ledger. The two comparative contexts enter only with their own source accession. Foreign forms remain unmodeled pending distinct accounting-context and completeness validation. [Filing additions](filing_additions.csv) · [Fact additions](fact_additions.csv)

The unchanged proxy builder uses the later of official filed date and Eastern acceptance date, lets two subsequent exchange sessions complete, then stages the filing from the following session. Source histories are incomplete and stale sample values expire after 90 sessions. Exact dissemination evidence remains absent. Strict-policy coverage is zero, and no strict-policy files were edited.

| Assessment | Total rows | Mapped issuer rows | Filing rows | Fundamental rows |
|---|---:|---:|---:|---:|
| 2022 five-day | 11,925 | 323 | 194 (1.63%) | 53 (0.44%) |
| 2023 five-day | 10,701 | 923 | 252 (2.36%) | 86 (0.80%) |

2022 events cover four tickers in three sectors (99 10-K rows, 95 10-Q rows). 2023 events cover five tickers in four sectors (47 10-K rows, 205 10-Q rows). Fundamentals cover PepsiCo in 2022 and PepsiCo plus Microsoft in 2023. All event-covered assessment rows contain issuer, accession, source, filed date, acceptance and eligible date. The remaining 2022 rows comprise 11,602 unresolved issuer dates and 129 mapped dates without recent qualifying filings. The respective 2023 counts are 9,778 and 671. Across all 127,979 historical rows there are 2,312 mapped issuer rows, 893 filing rows, and 179 fundamental rows. Both assessment key sets and labels were checked against existing archives and the label file. [Year](run_02/coverage_year.csv) · [Ticker](run_02/coverage_ticker.csv) · [Sector](run_02/coverage_sector.csv) · [Form](run_02/coverage_form.csv) · [Missingness](run_02/coverage_coverage_reason.csv)

No neural comparison is justified at this coverage. Acquisition requires a genuine project contact set in SEC_USER_AGENT, or a provenance-preserved local cache of issuer submissions, historical original accession records and facts. The absence of a configured contact is an operational prerequisite. The public SEC evidence demonstrates that historical filings exist. Once acquisition can proceed, start with verified historical issuer spans spanning multiple sectors and use submissions plus referenced older history pages. Cross-check accession facts and expand coverage before training. Indexed SEC pages were consulted through ordinary search without automated collection against SEC endpoints. [SEC fair access](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) · [SEC APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

Run 02 is authoritative. Run 01 is retained as an intermediate audit before adding the 2022 PepsiCo quarter facts. Each contains features and assessment Parquet, accession and identity CSVs, five coverage CSVs, and a source-hash manifest. New inputs, rebuild script, report, and these output directories are isolated under this experiment. No existing files, official artifacts, notebooks, or 2026 lock state were changed. No Git operations.

```sh
.venv/bin/python -B -m experiments.sec_coverage_resume.rebuild experiments/sec_coverage_resume/reproduce
.venv/bin/python -B -m pytest -q -p no:cacheprovider experiments/sec_lagged_proxy/test_build.py experiments/sec_pit_v1/test_features.py
.venv/bin/python -B -m experiments.historical_demo_v1.replay development
```

Six timing and accession tests passed. The rebuilt coverage audit also checked key equality, original labels and absence of future eligible dates on assigned assessment rows. Genuine fallback inference reproduced all 11,925 probabilities exactly and rebuilt all 1,492 graph rows. Original-attention GAT-LSTM seed 42, 2022 five-day retrospective assessment: Accuracy 55.4801%, majority 51.1614%, within-day AUC 0.543836, MCC 0.111372. The 32-unit checkpoint is preserved. No new metrics support either 0.63 target. [Fallback checkpoint](../gat_edge_weight_v1/runs/w0_h5_s42_original.pt) · [Saved predictions](../gat_edge_weight_v1/runs/w0_h5_s42_original.parquet) · [32-unit checkpoint](../gat_regularization_v1/runs/w0_h5_s42_small.pt)
