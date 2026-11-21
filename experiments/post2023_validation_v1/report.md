# Post-2023 market-data validation and restricted development study

## Decision

A pinned public Yahoo Finance archive supplied useful bars for development. The full frozen cohort failed the acquisition and corporate-action gates, so 2026 was left unscored. The restricted study found no reliable advantage over the matched reference and did not attain 70% on either accuracy task. The official results and earlier research were preserved.

## Source and coverage

Stooq's one CSV probe returned a browser-verification HTML page. No authorized Alpaca credentials were found in the local environment or standard project locations checked. Alpaca's free IEX feed would require its own venue compatibility assessment. It is a single exchange. [Alpaca market data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq)

The public AmirTrader export supplied 97 symbol Parquets at revision `2f8763bb9ed0c09fca6e871c8d660bdd8e4ce0ea`. The retrieved release credits Yahoo Finance via yfinance and provides quote OHLC, adjusted close, volume and date. The acquisition ledger records retrieval times, URLs and SHA-256 hashes. The archived dataset transfer totaled 8,078,161 bytes including its README. Vendor confirmation of consolidated coverage, stable security identifiers and a dated adjustment ledger were unavailable. [Pinned archive](https://huggingface.co/datasets/AmirTrader/YahooFinance/tree/2f8763bb9ed0c09fca6e871c8d660bdd8e4ce0ea)

| Year | Expected fixed-cohort ticker-days on exchange sessions | Usable archive bars |
|---|---:|---:|
| 2023 | 25,250 | 24,250 |
| 2024 | 25,452 | 24,444 |
| 2025 | 25,250 | 24,250 |
| 2026 through November 19 | 22,523 | 21,599 |

No duplicate sessions, unusable OHLCV values or non-session bars were found in the audited slice. ANSS, EA, SPLK and WBA were absent from the provider listing, including their actively traded years. WBD coverage stopped on October 6, 2026, omitting 32 nominal sessions through November 19. Missing ticker-days while still marked members in the local historical table, counting 2023 through the cutoff, were ANSS 635, EA 898, SPLK 61, WBA 388 and WBD 2. The full Cartesian gap counts include days outside membership and after merger or delisting. [Exact gap dates](missing_sessions.csv) · [Coverage by symbol and year](coverage.csv)

Issuer or SEC records document SPLK acquired March 18, 2024, ANSS July 17, 2025, WBA August 28, 2025, and EA August 4, 2026. The WBD merger closed October 6, 2026. WBD requested a pre-open halt that day, so its last dated bar requires venue and security-level reconciliation. The SIRI quote series crosses the September 10, 2024 split-off, replacement shares and 1-for-10 adjustment with the same ticker. Nominal price continuity alone cannot certify the underlying security or settlement history. [SPLK](https://newsroom.cisco.com/c/r/newsroom/en/us/a/y2024/m03/cisco-completes-acquisition-of-splunk.html) · [ANSS](https://news.synopsys.com/2025-07-17-Synopsys-Completes-Acquisition-of-Ansys) · [WBA](https://www.sec.gov/Archives/edgar/data/1618921/000119312525190603/d87240dex991.htm) · [EA](https://www.ea.com/ea-studios/ea-sports/news/ea-announces-completion-of-acquisition) · [WBD](https://www.sec.gov/Archives/edgar/data/1437107/000119312526415055/d200872d8k.htm) · [SIRI](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2024-421)

## Comparability and evaluation gates

The 2023 overlap had 24,250 matched bars from 97 securities. For each ticker, after removing the constant median price-scale difference, the largest ticker-level median absolute error across OHLC was 0.00116 basis points. Close scale ratios ranged from 0.998072 to 1.000000. Share volume matched exactly on 24,223 rows. Signs agreed on all 24,153 matched adjacent-session and 23,765 five-session returns. NVDA and AVGO quotes showed split-adjusted continuity across their 2024 10-for-1 splits. The archive supplies an adjusted-close series but lacks full dividend, split, ticker mapping and exit-payment event data. These spot checks and the 2023 overlap support restricted adjusted-return research. The provider and original bars were never spliced. [Overlap results](overlap_2023.csv) · [NVDA filing](https://www.sec.gov/Archives/edgar/data/1045810/000104581024000144/nvda-20240607.htm) · [AVGO filing](https://www.sec.gov/Archives/edgar/data/1730168/000114036124033072/ny20032136x6_8k.htm)

The project membership table reproduced the 101-security cohort at December 29, 2023. Compared with that fixed list, the point-in-time membership table showed 6 replacements by end-2024, 14 by end-2025, and 23 by November 19, 2026. That quantifies cohort drift. The four absent histories also show archive selection against acquired and delisted companies, which can bias scoring on the remaining securities. A fixed set of 30 liquid names chosen by original 2023 dollar volume had complete 2023–2025 archive bars and was evaluated as a limited development subset. The study uses the versioned archive's revised historical data, so exact as-of historical OHLCV values are unverified. The full locked protocol still requires delisted-name history, corporate actions and security/venue reconciliation. No 2026 direction labels or model metrics were computed. [Historical membership comparison](audit_summary.json) · [Frozen full protocol](../market_direction_v1/protocol.md)

## Development design and fit accounting

The [restricted protocol](development_protocol.md) fixed both windows, horizons, calibration and 20% nominal selective rule before scoring. The 2024 window fit April 17, 2023 through April 16, 2024, calibrated April 24 through July 5, 2024, and assessed July 15 through December 20, 2024 (113 dates and 3,390 rows per horizon). The 2025 window fit April 15, 2024 through April 15, 2025, calibrated April 24 through July 7, 2025, and assessed July 15 through December 19, 2025 (112 dates and 3,360 rows). Five exchange sessions between partitions purged crossing 5-day labels. The LSTM also used the calibration partition for checkpoint selection. The same stock-date keys were used within each comparison.

Stock-and-market logistic used C=0.25, weight-fit-only median imputation and scaling. Features include trailing stock returns, normalized momentum, rolling volatility and market beta, relative strength, range, volume change, trend distance, cross-sectional ranks, and equal-weight market breadth and dispersion from the fixed 30. The relative-return histogram boosted regressor fit each stock's return less its day's cohort mean, with 90 rounds, learning rate 0.05, 7 leaves, minimum 150 rows per leaf and L2=8. Direction classification and within-day AUC used the original strictly-positive adjusted close-to-close label. The Price Only LSTM used the shared configuration with weight-fit preprocessing, 20 sessions, 64 hidden units, dropout 0.1, learning rate 0.001, up to 16 epochs, validation every 14 steps and patience of four epochs. Explicit stochastic seed was 42 where supported. Each arm used standardized L2 logistic score calibration and a 0.5 direction threshold.

Eleven fits returned metrics. One initial LSTM attempt exited after the first two model records during the prediction-key check. The runner had compared pandas key dtypes as well as values. Resume checked the ordered values and reused the completed models. The last 2025 5d LSTM was skipped to reserve one fit against the 12-fit cap for the failed attempt. No matched neural result exists for that cell. LSTM selected steps were 28 (2024 1d), 42 (2024 5d) and 14 (2025 1d), with selection AUC 0.5029, 0.5220 and 0.5104 respectively. The 2024 5d LSTM declined to 0.4394 on assessment. LSTM completion took about 1.5–2.6 CPU wall-clock seconds per successful fit as reported by the training code.

## Matched development results

Within-day AUC excludes one-class dates and averages the remaining dates. Rank IC is mean per-date Spearman correlation with realized forward return. Paired AUC intervals are descriptive 10-session circular block bootstrap intervals on the same dates, with no multiple-comparison adjustment. An undefined MCC indicates single-class predictions. Accuracy and ranking can diverge because a score can order stocks differently without ever crossing the Up/Down threshold.

| Window | Horizon | Model | Accuracy | Majority | MCC | Within-day AUC | Rank IC | AUC difference vs logistic [95% interval] |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 2024 | 1d | stock_logistic | 0.5251 | 0.5260 | 0.0110 | 0.5071 | -0.0230 | Reference |
| 2024 | 1d | relative_regressor | 0.5260 | 0.5260 | Undefined | 0.5183 | 0.0318 | +0.0112 [-0.0179, +0.0386] |
| 2024 | 1d | price_lstm | 0.5260 | 0.5260 | Undefined | 0.5185 | 0.0288 | +0.0114 [-0.0336, +0.0528] |
| 2024 | 5d | stock_logistic | 0.5218 | 0.5218 | Undefined | 0.4893 | 0.0199 | Reference |
| 2024 | 5d | relative_regressor | 0.5221 | 0.5218 | 0.0085 | 0.4892 | -0.0368 | -0.0000 [-0.0697, +0.0694] |
| 2024 | 5d | price_lstm | 0.5218 | 0.5218 | Undefined | 0.4394 | -0.0966 | -0.0498 [-0.1302, +0.0311] |
| 2025 | 1d | stock_logistic | 0.5277 | 0.5277 | Undefined | 0.5078 | 0.0239 | Reference |
| 2025 | 1d | relative_regressor | 0.5277 | 0.5277 | Undefined | 0.5136 | -0.0060 | +0.0058 [-0.0200, +0.0288] |
| 2025 | 1d | price_lstm | 0.5277 | 0.5277 | Undefined | 0.4895 | -0.0163 | -0.0183 [-0.0507, +0.0135] |
| 2025 | 5d | stock_logistic | 0.5446 | 0.5446 | Undefined | 0.5411 | 0.0746 | Reference |
| 2025 | 5d | relative_regressor | 0.5446 | 0.5446 | Undefined | 0.5241 | 0.0191 | -0.0170 [-0.0610, +0.0303] |

### Selective accuracy

| Window | Horizon | Model | Accuracy [95% interval] | Coverage | Rows | Dates | Up share | Subset majority |
|---|---:|---|---|---:|---:|---:|---:|---:|
| 2024 | 1d | stock_logistic | 47.92% [35.56%, 62.22%] | 1.42% | 48 | 31 | 100.00% | 52.08% |
| 2024 | 1d | relative_regressor | 53.00% [48.23%, 58.69%] | 26.55% | 900 | 111 | 100.00% | 53.00% |
| 2024 | 1d | price_lstm | 54.83% [48.37%, 60.49%] | 16.19% | 549 | 97 | 100.00% | 54.83% |
| 2024 | 5d | stock_logistic | 52.80% [39.03%, 66.22%] | 52.68% | 1786 | 86 | 100.00% | 52.80% |
| 2024 | 5d | relative_regressor | 52.69% [41.12%, 64.73%] | 25.25% | 856 | 112 | 100.00% | 52.69% |
| 2024 | 5d | price_lstm | 50.00% [36.68%, 64.11%] | 31.39% | 1064 | 113 | 100.00% | 50.00% |
| 2025 | 1d | stock_logistic | 50.00% [0.00%, 75.00%] | 0.18% | 6 | 4 | 100.00% | 50.00% |
| 2025 | 1d | relative_regressor | 53.64% [47.79%, 60.24%] | 17.59% | 591 | 111 | 100.00% | 53.64% |
| 2025 | 1d | price_lstm | 50.17% [46.44%, 53.88%] | 42.95% | 1443 | 112 | 100.00% | 50.17% |
| 2025 | 5d | stock_logistic | 63.64% [34.62%, 100.00%] | 0.65% | 22 | 14 | 100.00% | 63.64% |
| 2025 | 5d | relative_regressor | 58.31% [46.27%, 68.38%] | 10.57% | 355 | 103 | 100.00% | 58.31% |

The fixed-30 cohort is TSLA, NVDA, AAPL, MSFT, AMZN, AMD, META, GOOGL, GOOG, NFLX, AVGO, ADBE, INTC, PYPL, COST, MU, QCOM, CSCO, PANW, BKNG, PDD, PEP, TXN, AMAT, CMCSA, INTU, TMUS, LRCX, ABNB, ASML.

## Interpretation

The relative regressor led logistic at 1d in both windows by 0.0112 and 0.0058 AUC, with both intervals crossing zero and inconsistent rank IC. At 5d, logistic was the stronger simple reference in the later window, but fell below 0.50 in the first. The neural reference did not improve reliably on either horizon. Keep logistic as the simple reference, with the relative regressor retained as an exploratory 1d candidate. None merits designation as a validated winner.

The highest full-population accuracy was 54.46%, tied with the majority baseline, and the highest within-day AUC was 0.5411, both for 2025 5d logistic. The best selective score was 63.64% (14 of 22 rows, 14 dates, 0.65% coverage), also 2025 5d logistic. It exactly matched its same-subset majority baseline. All selective predictions across the study were Up. The wide selective intervals and coverage drift are evidence against a stable operating point. There was no 70% result or reliably useful bidirectional signal.

Historical official 1d and 5d GAT-LSTM accuracy was 0.4994 and 0.5040, AUC 0.5094 and 0.4992, and MCC 0.0024 and 0.0047. These came from a different population, period and model family. The study used matched developmental references to assess any gain. No trading backtest was run on historical adjusted closes. The study leaves venue, executable entry prices, liquidity, costs and exit settlements unresolved.

## Reproducibility and next action

Only this isolated experiment directory was written. It contains the acquisition, audit, development and summary scripts, the source manifest and raw response files, the compressed panel, coverage and overlap tables, the development protocol, prediction Parquets, the run ledger and resume log. The run ledger records the data, configuration and protocol hashes, full metrics, daily AUC and LSTM histories. Forty-six existing feature, orientation, split and study tests passed. The workspace retained about 9.8 GiB free after acquisition.

Obtain the four absent securities' historical bars through last trading, stable security identifiers and a dated corporate-action and delisting ledger, and reconcile the WBD October 6 bar and volume/venue semantics. Preserve the 2026 scoring lock until the complete frozen cohort passes these checks. Further historical tuning is unwarranted on this evidence.
