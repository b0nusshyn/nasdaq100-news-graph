# Dataset and bias handling

Current Task 1.6 snapshot: 238,195 price/label rows across 174 tickers, with zero unexplained
in-index quote gaps across 2018–2023 and six documented nontrading sessions. Quality is
PASS_WITH_FLAGS; [the detailed review](../data_quality_1_6.md) documents retained price/volume
flags and annual class balance. Acquisition counts below describe earlier recovery stages.

Owner: Data · Issue #46. Measured acquisition/recovery audit below; unresolved data and model inputs remain pending.

The intended experiment covers NASDAQ-100 securities during 2019–2023. Prices are requested
from 2018 to provide indicator warm-up. A separate persistence check requests prices through
November 2026; Yahoo's end date is exclusive, and sessions after the download date must be
collected later. README specifies an end of 2026-11-30 while CHECKLIST 1.2 specifies
2023-12-31; the downloader exposes this choice through `--end`.

Historical membership is reconstructed by reversing Wikipedia's dated additions/removals
from the current constituent list. Multiple non-overlapping spells are retained for securities
that leave and rejoin. Spell starts and ends are inclusive NASDAQ session dates; pre-window
membership is left-censored at the explicitly requested 2018-01-01 sentinel. A removal on d
ends the preceding trading session. The builder rejects inconsistent events, canonical ticker
collisions, duplicate keys and overlapping spells, and checks for 100–103 members on every
2019–2023 trading session. Counts alone do not establish the completeness of Wikipedia's
retrospective records. Current and historical component tables now reside on separate
Wikipedia pages; source classification fields may use ICB rather than GICS.

Historical symbols map to current symbols through a shared rename map. GOOG and GOOGL remain
separate securities. Acquired and delisted securities are not replaced automatically by an
acquirer's price series. Yahoo OHLCV is downloaded per canonical ever-member with
`auto_adjust=True`, requesting split/dividend-adjusted OHLC and float64 volume. Empty,
malformed or failed histories are recorded separately. Yahoo data is retrospectively adjusted;
these data do not establish availability of a particular adjusted data vintage at past dates.
The missing/delisted-price audit must be completed before empirical results can support an
absence-of-survivorship-bias claim.

All price rows are retained, including out-of-index rows needed for indicator warm-up.
`in_index` is determined by historical spells; only rows with `in_index=True` enter the
training/evaluation sample. No ticker is considered a member before its join or during a gap
between spells. A target for horizon h is `close[t+h]/close[t]-1` at the exact h-th future
NASDAQ session. Positive returns map to 1; nonpositive returns to 0. Missing target closes,
non-session sources and unavailable trailing targets produce null labels, and missing
sessions never extend the target horizon. Prices and labels share exactly the same keys.

Indicators use information through the close of t, implying an after-close forecast origin.
Features are backward returns, 20-observation return volatility, Wilder RSI14, MACD12/26,
EMA9 MACD signal, and trailing volume z-scores. Early warm-up values remain null; no future
backfill is applied. Feature windows count observed rows, so missing-session findings need
review before training. Global feature scaling is not performed by the Data stage; downstream
scalers and graph statistics must be fitted within training folds. The shared evaluation
protocol uses chronological folds with a five-session embargo.

Automated checks cover missing sessions and entire member histories, failed downloads,
non-session rows, zero volume, large adjusted-close moves requiring corporate-action review,
label consistency and annual per-ticker/horizon in-index class balance. News coverage is
measured from matched `article_tickers` over observed in-index price days; this is not a
measurement of raw FNSPID completeness. A reviewed Investing.com earnings CSV can identify
candidate coverage gaps in an earnings-day/next-session diagnostic window. These dates are
not used as point-in-time model features. Historical earnings acquisition and real coverage
findings remain pending, as does delivery of the NLP matching artifact.

Regression tests check historical membership flags, independence of indicators from future
price changes, and independence of labels from prices beyond their stated horizons. Test
fixtures are synthetic; some were generated with generic business days and include US market
holidays. Production-calendar rebuilds can therefore differ from their original synthetic
labels. These fixtures validate implementation behavior and are not empirical observations.

Before finalising this section, insert measured ticker/spell counts, data vintage and download
dates, unresolved failed histories, yearly missing-session/class-balance summaries, and actual
FNSPID/earnings coverage findings. No such numbers have been inferred from synthetic data.

## First real-data download audit

Recorded 2026-11-15T01:26:13.638257+00:00. Yahoo was requested for 2018-01-01 through the exclusive
end 2024-01-01. The observed range is 2018-01-02–2023-12-29.
The Wikipedia reconstruction contains 187 spells for 180 canonical tickers;
daily 2019–2023 membership ranges from 101 to 103.

Prices were obtained for 152 tickers (216,662 rows). Labels have the same
216,662 keys, with 216,510 non-null 1-session labels and
215,902 non-null 5-session labels. These are acquisition counts, not model results.
There are 28 unresolved download failures, including 14 tickers
with membership intersecting the main 2019–2023 experiment. The exact failures are in
`raw/failed_tickers.csv`; window-specific classification is in `raw/failed_tickers_audit.csv`.

The expected 2019–2023 universe contains 128,524 in-index ticker-days. The downloaded
prices cover 117,921, leaving 10,603 in-index rows
missing. Completely absent main-window members are:
ALXN, ANSS, ATVI, CELG, CERN, CTXS, EA, MXIM, MYL, SGEN, SPLK, WBA, WBD, XLNX.
FOX and FOXA also have missing early-2019 history requiring corporate-action/identity review.
The quality audit found 16 adjusted
moves greater than 40% and 4 ticker-level
zero-volume findings. These require review and are not automatically classified as bad prices.
No internal missing-session, label-formula or membership-flag mismatches were reported.

The initial OHLC guard was too strict about floating-point rounding in adjusted prices;
review found an AZN discrepancy of approximately 7e-15. The revised guard uses 1e-12 relative
and absolute tolerances without changing the prices. Seven ticker histories were recovered
on retry, and one additional history was recovered after mapping NLOK to GEN based on its
[issuer announcement](https://www.sec.gov/Archives/edgar/data/849399/000110465922115277/tm2229813d1_ex99-01.htm).

**This dataset remains incomplete and must not be used to claim unbiased evaluation.**
Missing acquired/delisted histories need an independent historical source; no acquiring
company prices were substituted. Failed tickers must not be silently dropped from the
research universe. News/earnings coverage cannot be measured until the real NLP matching
artifact and historical earnings input are available. Download versions, bounds and file
checksums are recorded in `raw/download_manifest.json`; annual label balance is in
`raw/annual_label_balance.csv`. Both are local raw diagnostics rather than frozen artifacts.

## Recovery update: archived FNSPID prices

A subsequent recovery read selected CSVs from the FNSPID authors' price archive using
bounded HTTP Range requests. Only 3,493,618 bytes were read from the 589,525,596-byte ZIP.
Dataset revision `bf9189c41527198897d1af3e17b1a0095279fc45`, per-file checksums and issuer-symbol
filing URLs are recorded in the reviewed import manifest and recovery provenance. Eleven
previously absent ticker histories were added: ALXN, ANSS, ATVI, CERN, CTXS, EA, MXIM, MYL,
SPLK, WBA and XLNX. These supply 10,442 price rows. Their raw OHLC was dividend-adjusted using
archived adjusted-close/raw-close ratios; source volume was retained. Existing ticker price
histories were not overwritten or spliced across adjustment vintages, and pre-recovery
artifacts were backed up.

The current artifacts contain **163 tickers and 227,104 price/label rows**, including
124,727 of the 128,524 expected main-window in-index ticker-days. Missing in-index rows
fell from 10,603 to **3,797**. CELG, SGEN and WBD remain completely absent; other gaps are
listed by exact date in `raw/missing_prices.csv`. Several recovered delisted series stop
at 2020-07-02, and ANSS/EA/SPLK/WBA stop at 2023-12-28, so provider archive availability must
not be confused with continuous coverage through the required end date.

Identity review also rejected the archive's GEN file: around the 2019 transition its prices
and volumes materially differ from the NLOK issuer history, so filename equality would have
introduced an unrelated or misidentified series. NLOK's archived data repeat Yahoo's zero
volume on the 2019-11-05 ticker-transition date; no replacement was invented. SEC filings
establish that new Fox Corporation became standalone on March 19, 2019. Consequently old
Twenty-First Century Fox and new Fox share histories cannot be interchanged just because
FOXA/FOX symbols were reused. The reconstructed early-2019 membership needs an issuer-level
review before any change to those spells; the frozen artifact schema was not changed.

A review register distinguishes 16 large-move findings: six occur on in-index observations,
while ten are outside membership in this download. Of four zero-volume ticker findings,
only GEN's transition-day row is in-index; BKR, FER and XEL zero-volume rows are out-of-index.
These classifications describe current membership, not proof that warm-up values can never
matter. Large moves are not automatically bad quotes or unadjusted splits. Independent
price/corporate-action confirmation remains required, and the dataset remains incomplete.
