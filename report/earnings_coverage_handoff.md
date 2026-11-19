# FNSPID coverage hand-off for NLP

The matched-news coverage table shows the following ticker-day counts across observed
in-index prices. The uncovered column is calculated as ticker-days minus days with news.

| Year | Ticker-days | With matched news | Uncovered | Coverage | Tickers with zero covered days |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2019 | 25,956 | 16,149 | 9,807 | 62.22% | 2 |
| 2020 | 26,051 | 12,484 | 13,567 | 47.92% | 0 |
| 2021 | 25,695 | 9,345 | 16,350 | 36.37% | 10 |
| 2022 | 25,567 | 12,439 | 13,128 | 48.65% | 2 |
| 2023 | 25,250 | 13,366 | 11,884 | 52.93% | 4 |

These counts were reproduced using the supplied real article match file and agree with the
[tracked NLP coverage table](../results/tables/news_coverage_by_year.csv).
Start the review with 2021, which has the lowest coverage and the most tickers
with zero covered days (ABNB, ALXN, CTAS, HON, IDXX, KLAC, LCID, MDLZ, MXIM, SGEN).
For gaps, check the raw FNSPID extract and then the
ticker aliases and ET session alignment. The aggregate measures matched
articles on observed in-index price days.

## Earnings sample

[Reviewed dates](investing_earnings_sample.csv) contains 13 releases across all five
years from contemporaneous reporting hosted by Investing.com. The sample intentionally
includes three tickers with zero matched days in 2021 to verify specific gaps. It cannot
estimate an index-wide earnings coverage rate. The historical calendar request for
October 24, 2023 returned HTTP 403 on November 19, 2026, and the public Apple page
displayed only a recent window. A complete calendar export is still needed.
[Investing.com calendar](https://www.investing.com/earnings-calendar/)

Use the real matched-article Parquet supplied by NLP to check these events.

```bash
python -m src.data.coverage --articles /path/to/article_tickers.parquet \
  --earnings report/investing_earnings_sample.csv
```

Set `NDX_DATA_DIR` to the directory containing real `prices.parquet`.
Review rows with `coverage_gap=True` in `raw/earnings_coverage.csv`, alongside
`observed_in_index` and `window_within_range`. The window spans the release
date through the next NASDAQ session. A candidate gap means zero matched
articles for the ticker on observed in-index days in that window. A matching
article may cover a different story.

All 13 windows were in range and had observed in-index prices. The run identified
three windows with zero matches. For the NLP review, start with ALXN on April 30,
CTAS on September 29, and KLAC on October 27, all in 2021.

| Release date | Ticker | Window end | Distinct matched articles |
| --- | --- | --- | ---: |
| 2019-10-30 | AAPL | 2019-10-31 | 8 |
| 2020-10-29 | AAPL | 2020-10-30 | 1 |
| 2021-04-30 | ALXN | 2021-05-03 | 0 |
| 2021-09-29 | CTAS | 2021-09-30 | 0 |
| 2021-10-27 | KLAC | 2021-10-28 | 0 |
| 2021-10-28 | AAPL | 2021-10-29 | 5 |
| 2022-10-26 | KLAC | 2022-10-27 | 1 |
| 2022-10-27 | AAPL | 2022-10-28 | 58 |
| 2023-10-24 | GOOGL | 2023-10-25 | 33 |
| 2023-10-24 | MSFT | 2023-10-25 | 99 |
| 2023-10-26 | AMZN | 2023-10-27 | 23 |
| 2023-11-02 | AAPL | 2023-11-03 | 74 |
| 2023-11-21 | NVDA | 2023-11-22 | 96 |

The supplied article match file has SHA-256
`e75932d31ba0aa6a4c8daf240984cbe1aa238d9725d27ccb0127c41b4bef53a0`.

## Broader SEC filing analysis, 2019–2023

[Quant500's open data](https://quant500.com/data) provides dates and original SEC
links for Item 2.02 filings. The local preparation command keeps original 8-Ks
with a reported date and no uncertainty flag, removes duplicate ticker-dates,
and applies the project's inclusive historical membership intervals. Alphabet
and Fox issuer announcements are checked against each included share class.
A cover-page report date is a release-date proxy. Amendments and filing-date
fallbacks are excluded. The SEC filing acceptance time is unused.

| Year | Ticker-date windows | Tickers | Zero matched articles | Share with zero matches |
| --- | ---: | ---: | ---: | ---: |
| 2019 | 359 | 88 | 23 | 6.41% |
| 2020 | 362 | 89 | 56 | 15.47% |
| 2021 | 342 | 83 | 120 | 35.09% |
| 2022 | 333 | 82 | 73 | 21.92% |
| 2023 | 341 | 87 | 74 | 21.70% |
| Total | 1,737 | 114 | 346 | 19.92% |

All 1,737 windows had in-index prices and ended within the study period. For
triage, ISRG (24), ILMN (18), INTU (16), REGN (16), and CTAS (15) had the most
zero-match windows. Check the raw news extract and then ticker matching and
session alignment for those windows. The yearly rates describe the observed
sample. The provider tracks an S&P 500-based issuer universe, and Form 8-K
misses releases published by other routes. Even covered tickers can have
missing quarters. The denominator counts each share class separately.

Of 148 historical ticker symbols, 34 had no eligible SEC-source event in the
run. These are ASML, AZN, BIDU, BKNG, BMRN, CCEP, CHKP, CMCSA, DASH, DOCU, EA,
GEN, GFS, JD, LBTYA, LBTYK, LCID, MDB, MELI, NTES, OKTA, PDD, PTON, RIVN, ROP,
SGEN, SIRI, SPLK, TCOM, TEAM, TFCF, TFCFA, ZM, and ZS. Company investor relations
pages and applicable SEC filings can supply dates for the missing issuers.

Reproduce with the [source CSV](https://quant500.com/api/descarga/anuncios.csv)
and the same matched-news file used above.

```bash
python -m src.data.prepare_sec_earnings /path/to/quant500.csv data/raw/sec_earnings.csv
python -m src.data.coverage --articles /path/to/article_tickers.parquet \
  --earnings data/raw/sec_earnings.csv
```

The downloaded Quant500 snapshot has SHA-256
`4681db12d825009a0ca66221f2d832049e9335c874104a0`. The run writes
`data/raw/earnings_coverage.csv` when using the default data directory.
