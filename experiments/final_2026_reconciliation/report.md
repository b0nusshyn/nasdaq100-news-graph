# Final data reconciliation and 2026 evaluation gate

**NOT READY — HISTORICAL DATA OR CORPORATE ACTIONS INCOMPLETE.** The Nasdaq/SEC exit-date reconciliation passed. The frozen 101-security data gate failed. No 2026 return values, binary labels, predictions or model metrics were computed in this cycle. No model was trained.

## Corporate actions

The [dated ledger](corporate_actions.csv) records CUSIP, issuer SEC CIK, observed exit cutoff, legal closing and Nasdaq suspension dates, transaction consideration and both authoritative URLs. Nasdaq and the corresponding SEC filing distinguish the last trading session from the subsequent legal and exchange dates.

| Security | Last trading session | Merger | Nasdaq suspension | Consideration per eligible share |
|---|---|---|---|---|
| SPLK | 2024-03-15 | 2024-03-18 | 2024-03-19 | $157 cash |
| ANSS | 2025-07-16 | 2025-07-17 | 2025-07-18 | $199.91 cash plus 0.3399 Synopsys shares |
| WBA | 2025-08-27 | 2025-08-28 | 2025-08-29 | $11.45 cash plus contingent nontransferable divested-asset proceeds right |
| EA | 2026-08-04 | 2026-08-04 after trading | 2026-08-05 | $210 cash |
| WBD | 2026-10-05 | 2026-10-06 | 2026-10-07 | $31.01666668 cash |

October 6 WBD in the pinned archive is a zero-volume row with four identical $30.95 OHLC values and the same close as the prior session. Nasdaq specifies October 5 as last trade, the SEC filing documents the pre-open halt request, and the independent ChartExchange page ends October 5. The row is isolated in [archive quarantine](archive_quarantine.csv). Stock Analysis additionally displayed one ANSS row after last trade with missing volume and four SPLK rows after last Nasdaq trade. These remain flagged in the raw audit. The source and venue of those SPLK prints were not validated for this target. All considered merger proceeds are settlement terms. No synthetic traded close, forward fill or acquirer price was used. [WBD Nasdaq notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2026-710) · [WBD SEC filing](https://www.sec.gov/Archives/edgar/data/1437107/000119312526415055/d200872d8k.htm) · [WBD independent history](https://chartexchange.com/symbol/nasdaq-wbd/historical/)

## Historical recovery

The date scope is the pinned archive through 2026-11-19. Source pages are retained under `raw/` with hashes, URLs and download metadata in [readiness.json](readiness.json). Public ChartExchange tables reached all four last trading dates. Stock Analysis provided a second check near exit and limited adjusted-close values from displayed history. Yahoo chart requests returned HTTP 429, so no retries or bypass were attempted. ChartExchange's full-history API requires an API key. A 2024–2026 waterfall of partly overlapping sources would require a material change to the single-provider protocol. [ChartExchange price history](https://chartexchange.com/docs/stocks/historical/) · [ChartExchange API access](https://chartexchange.com/docs/api/v1/) · [Stock Analysis example](https://stockanalysis.com/stocks/ea/history/)

| Ticker | Recovered alternate-source bars from 2023 through exit | Recovered range | Sessions still missing from audited external sources before exit |
|---|---:|---|---|
| SPLK | 302 | 2023-01-03 to 2024-03-15 | 0 |
| ANSS | 370 | 2024-01-24 to 2025-07-16 | 265, 2023-01-03 to 2024-01-23 |
| WBA | 370 | 2024-03-07 to 2025-08-27 | 295, 2023-01-03 to 2024-03-06 |
| EA | 370 | 2025-02-12 to 2026-08-04 | 529, 2023-01-03 to 2025-02-11 |

The original project prices cover 2023 for the missing names, but using them to fill external histories would splice adjustment vintages. Even with that change, 339 post-2023 sessions would remain missing across ANSS (15), WBA (45) and EA (279). The alternate prices remain isolated, with exact dates and provider status in the [session inventory](session_inventory.csv) and [recovered candidate table](recovered_candidates.csv).

## Adjustments and membership

On 318 overlapping SPLK observations in 2022–2023, closes agreed to the cent on all 318. The volumes agreed on 97. Across the four absent names, ChartExchange and Stock Analysis close prices agreed to the cent on all 195 pre-exit overlaps. AAPL gave a control against the pinned archive, with all 369 closes matching at cents precision and only 3 exact volume matches. WBD matched on 368 of 370 closes and 3 exact volumes. Close alignment supports price-level sanity but does not certify equivalent venue aggregation, volume, or adjustment timing. The alternate OHLCV values cannot be substituted into the frozen model's volume and adjusted-return features without reconciling those semantics. The prior 97-name 2023 matched-return check had zero 1d or 5d direction disagreements. Full point-in-time adjustment history and persistent vendor security identifiers remain unverified. [Cross-source comparisons](source_overlap.csv) · [Prior 2023 overlap](../post2023_validation_v1/overlap_2023.csv)

The inventory independently reproduces all 101 anchor members from the project's historical membership intervals. It preserves the historical index membership bit alongside the fixed cohort and verified exits. The earlier 6, 14, and 23 replacements by the ends of 2024 and 2025 and the 2026 snapshot illustrate why those two concepts remain distinct. The post-exit calendar rows in the inventory are structural exclusions. They are not provider gaps and generate no observable targets. Identity reconciliation is limited to the five exit securities' exchange and SEC records. It does not resolve every possible listing or issuer change across the archive.

## Endpoint and label inventory

The target definition remains adjusted close-to-close direction on the exact first or fifth future NASDAQ session. The audit records only endpoint dates and whether there are eligible input records, without opening direction labels or computing evaluation metrics. A trade followed by an endpoint beyond its verified final session cannot be labeled under that target. The crossing cutoff is the last session for 1d and the final five sessions for 5d. The [inventory](session_inventory.csv) lists every date and endpoint, including historical membership and fixed-cohort eligibility, missing endpoints and archive tail.

| 2026 status for the 101-name anchor | 1d | 5d |
|---|---:|---:|
| Pinned archive endpoints present, provisional | 21,501 | 21,113 |
| Alternate endpoints only, adjustment and source unqualified | 146 | 142 |
| Horizon crosses confirmed final trading | 2 | 10 |
| Endpoint after archive cutoff | 96 | 480 |
| Start date after confirmed final trading | 778 | 778 |

For the full 2023–2026 inventory, five 1d and 25 5d endpoints cross the five documented exits. Endpoint presence alone does not qualify a label for final scoring. The final model would also depend on complete lagged inputs and a closed-bar signal cutoff. The recovered source data does not establish a usable historical cutoff or matching volume and adjustment conventions across the 101-security data panel.

## Evaluation decision and reproducibility

[Candidate specifications and hashes](candidate_freeze.json) pin the carried-forward logistic reference and the exploratory 1-day relative-return regressor. The Price Only LSTM stays a conditional reference under the earlier protocol. The existing five-session purge and chronological probability calibration with frozen confidence quantiles remain the rules. No final fitted model or complete scoring manifest was produced because source and training eligibility failed first. No success against the matched baseline or 70% thresholds was observed in this cycle. [Restricted feasibility](restricted_feasibility.md) summarizes the strongest previous developmental evidence and its limitations.

All changes are in this directory. The new files are this report, `restricted_feasibility.md`, `corporate_actions.csv`, `reconcile.py`, `test_reconcile.py`, `recovered_candidates.csv`, `archive_quarantine.csv`, `source_overlap.csv`, `session_inventory.csv`, `readiness.json`, `candidate_freeze.json`, and the 14 raw responses listed in the manifest. The official sources, metrics and prior experiments were preserved.

To regenerate the tables, make a new isolated sibling copy of this directory containing only `reconcile.py`, `corporate_actions.csv` and `raw/`. Run its reconciliation module from the repository root with the existing virtual environment. The runner uses exclusive creation and leaves previous outputs intact. To check the implementation, run `.venv/bin/python -B -m pytest -o addopts='' -q -p no:cacheprovider experiments/final_2026_reconciliation/test_reconcile.py tests/test_market_direction_study.py tests/test_splitter.py`. All 43 tests passed. Ruff passed for this directory. No Git operations were used.

Keep the 2026 lock. A consistent historical export must include the four missing identities through last trade and the 97 existing names with documented venue, adjustment and volume policies. Validate corporate-action identities and endpoint coverage on that export before constructing the held-out returns.
