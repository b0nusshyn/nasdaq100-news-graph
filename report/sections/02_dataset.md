# Dataset and bias handling

The main experiment covers NASDAQ-100 constituents during 2019–2023. Prices from 2018
provide warm-up for price indicators. The separate longer price history described in the
README supports the persistence check and falls outside the main evaluation window.

## Universe and prices

We reconstructed historical index membership from dated additions and removals and
retained separate spells when a security left and rejoined. A removal takes effect at
the next spell boundary, with the old spell ending on the preceding exchange session.
Models use `in_index` rows for training and evaluation. Earlier prices can supply
indicator history for an entrant. Historical ticker renames are resolved explicitly,
GOOG and GOOGL remain distinct securities, and acquired companies retain their own price
histories. The reconstruction depends on the accuracy of the historical constituent
records. [Data acquisition and recovery audit](../../data/README.md)

The November 15 quality snapshot contains **238,195 price and label rows across 174
tickers** in the audit window, including warm-up. In the main window, **128,519 of 128,524**
expected member-session keys have prices. The five missing keys are XLNX's documented
February 2022 trading halt. The audit found zero unexplained in-index quote gaps and no
entirely absent in-window members. Historical gaps were recovered from reviewed archives.
The final audit retains advisory flags for large adjusted-price movements and unverified
zero volume. Prices use retrospective corporate-action adjustments, so their exact
historical data vintage is uncertain. [Quality review and recovery results](../data_quality_1_6.md)

## Targets and timing

The forecast is made after the close on session *t*. Price features use observations
through that close. They include past returns, trailing volatility, RSI, MACD, and volume
z-scores. Warm-up values remain null until enough observations exist. For horizons of one
and five sessions, the target is 1 when the adjusted close on the exact future NASDAQ
session exceeds the close on *t*, and 0 otherwise. If either required close is unavailable,
the target is null. The label builder never moves the endpoint to a later available quote.

Article timestamps are converted to US Eastern time. News after 16:00 ET rolls to the
next trading session, and news dated on a non-trading day is aligned to the next
session. The panel keeps an explicit `has_news` feature so sparse coverage is observable.
Feature normalisation and learned graph statistics are fitted within each training fold.
Validation and held-out testing are chronological, with a five-session embargo to
protect the longest target horizon. [News encoding](03_news_encoding.md) · [Walk-forward method](04_method.md)

| Year | Observed in-index rows | 1-session up share | 5-session up share | Days with matched news |
| --- | ---: | ---: | ---: | ---: |
| 2019 | 25,956 | 54.76% | 58.36% | 62.22% |
| 2020 | 26,051 | 53.05% | 56.77% | 47.92% |
| 2021 | 25,695 | 52.20% | 54.24% | 36.37% |
| 2022 | 25,567 | 48.04% | 47.33% | 48.65% |
| 2023 | 25,250 | 52.68% | 55.42% | 52.93% |

Up shares exclude null targets for the respective horizon. News shares use observed
in-index ticker-days and measure articles that successfully matched and aligned.
A separate diagnostic using SEC-derived dates found **346 windows with zero matches
out of 1,737** checked ticker-date windows across **114 of 148** historical ticker
symbols. The share describes that source's covered sample. Earnings dates serve only
as a coverage check. [Matched-news and earnings coverage](../earnings_coverage_handoff.md)

Regression tests exercise historical membership, exact session horizons and invariance
of price features to future changes. The project also tests the embargo and checks
the shared held-out prediction row set.
