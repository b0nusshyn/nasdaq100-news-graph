# data/

## Current review snapshot — Task 1.6

As of 15 November 2026: 238,195 price/label rows, 174 tickers; membership has
189 spells / 182 tickers. The 2018–2023 quality run is **PASS_WITH_FLAGS**:
zero unexplained member-session gaps, six documented nontrading sessions,
17 large-return flags and four zero-volume ticker findings. Eight absent histories
have membership only after the audit window. Earlier acquisition counts below
are historical checkpoints, superseded by this snapshot.

```bash
python -m src.data.quality --start 2018-01-01 --end 2023-12-31 --strict
python -m src.data.audit --start 2019-01-01 --end 2023-12-31
```

See [Task 1.6 review](../report/data_quality_1_6.md) for exact limitations and output files.
Strict quality fails structural errors, unexplained gaps and unreviewed/stale anomaly
records in `configs/quality_review.csv`. Warnings are retained, not silently repaired.


Everything here except this file is gitignored. The team's copy lives in the shared Google
Drive folder **`nasdaq100-news-graph/data`**
([open on Google Drive](https://drive.google.com/drive/folders/1WFo7veigQ6npmJ0bk4YW7kGcpOJnrQ-p)),
laid out exactly like this directory. Access is limited to the team; ask Shayanis (Modelling)
to be added.

```
data/
├── raw/                      # downloads: fnspid_filtered.csv, scraped tables
├── membership.parquet
├── prices.parquet
├── labels.parquet
├── article_tickers.parquet
├── news_features.parquet
├── edges.parquet
├── edges_manifest.json       # what built each (fold, edge type) of edges.parquet
└── predictions/<model>.parquet
```

Formats are fixed by [`SCHEMA.md`](../SCHEMA.md). Code finds the folder through the
`NDX_DATA_DIR` environment variable (default: this directory).

## Colab

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["NDX_DATA_DIR"] = "/content/drive/MyDrive/nasdaq100-news-graph/data"
```

If the folder was shared with you rather than created by you, first add a shortcut to it in
"My Drive" (right-click → Organize → Add shortcut) so it appears under `MyDrive`.

## Local

Either sync the Drive folder with Google Drive for desktop and point at it:

```bash
export NDX_DATA_DIR="/path/to/GoogleDrive/My Drive/nasdaq100-news-graph/data"   # PowerShell: $env:NDX_DATA_DIR = "..."
```

or download the files you need into this directory.

## Publishing a new artifact

1. Write it with `src.core.io.write_table`, which validates it against the contract.
2. Upload it to the Drive folder, replacing the old file.
3. Comment on the issue that produced it, then close the issue so blocked work can start.

To work before an upstream artifact exists, use the synthetic copies:
`NDX_DATA_DIR=tests/fixtures`.

## Data tasks 1.1 and 1.2

Install the light dependencies for tests (`pip install -r requirements-dev.txt`), and
`pip install yfinance` before a real price download. Both CLIs support `--help` without
importing yfinance. Run the offline checks first:

```bash
NDX_DATA_DIR=tests/fixtures pytest tests/test_index_history.py tests/test_fetch_prices.py
ruff check .
NDX_DATA_DIR=tests/fixtures pytest
```

The tests mock HTTP/Yahoo and write to temporary folders; committed fixtures are not replaced.
Do **not** run the real builders with `NDX_DATA_DIR=tests/fixtures`: the membership builder
would replace the synthetic membership and Yahoo cannot resolve the synthetic company symbols.

For real data, set `NDX_DATA_DIR` to the team's data folder as above, then run:

```bash
python -m src.data.build_index_history
python -m src.data.fetch_prices --start 2018-01-01 --end 2026-11-30
```

The first command downloads small Wikipedia pages; the second makes one Yahoo request per
canonical ever-member and may take several minutes. No download is performed by tests.
Wikipedia's tables moved from the main Nasdaq-100 article to
[List of NASDAQ-100 companies](https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies) and
[Historical components of the Nasdaq-100](https://en.wikipedia.org/wiki/Historical_components_of_the_Nasdaq-100).
The scraper tries the original article first and the corresponding moved page second. It
selects tables by column headers rather than position, and stops on an unknown layout.
Source classification columns are preserved: the current ICB classifications are not GICS.

Check the membership summary for spell count, unique ticker count, and **100–103 members on
all NASDAQ trading sessions in 2019–2023**. Inconsistent adds/removals, duplicate events,
canonical collisions and overlapping spells stop the build before it writes an artifact.
Counts are a sanity check, not proof that Wikipedia's history is complete or accurate.
Effective dates roll forward to the next NASDAQ session if on a holiday/weekend. Removal ends
on the preceding session. The explicitly requested `2018-01-01` start is a left-censoring
sentinel for pre-window membership, even though it is a market holiday. Changes before this
window are ignored; changes after 2023 must be reversed to reconstruct the earlier members.

Price outputs under `NDX_DATA_DIR`:

- `raw/prices_raw.csv`: adjusted OHLCV, one row per session/ticker, including out-of-index dates.
- `raw/failed_tickers.csv`: ticker and reason for empty, invalid or failed downloads. It always
  has headers, including when there are no failures. Inspect this file before any hand-off.

Raw downloads are outside the frozen contract (`src.core.paths.raw_dir`). They use CSV rather
than inventing a new Parquet schema. Every final contract artifact uses `write_table`;
`fetch_prices` now also adds `in_index` and writes validated `prices.parquet` (task 1.3).
All raw price rows are retained, including warm-up and out-of-index dates.
No missing prices are filled and no failed ticker is silently replaced by an acquiring company.
A ticker with malformed OHLCV is reported as failed in full for data-quality review.
An all-failed run writes the reports and exits with an error; partial success prints a summary.
Raw files are replaced on each run, so keep a copy if comparing downloads.

`--end` is **exclusive**, as in yfinance. The default follows README (`2026-11-30`), while
CHECKLIST 1.2 still says `2023-12-31`; this discrepancy needs team review. Choose the latter
explicitly for the shorter download. To include November 30 itself, use `--end 2026-12-01`.
As of November 14, 2026, future sessions through November 30 are not available yet: rerun
after those sessions for the persistence check. Historical membership and current adjusted
Yahoo prices are retrospective sources; these tasks do not assert point-in-time data-vintage
availability. Missing delisted history still needs task 1.6 before model training.

Additional symbol mappings were checked against issuer/exchange announcements:
[PCLN → BKNG](https://ir.bookingholdings.com/news/news-details/2018/The-Priceline-Group-Inc--NASDAQ-PCLN-Announces-Name-Change-to-Booking-Holdings-Inc--02-21-2018/default.aspx),
[CTRP → TCOM](https://www.sec.gov/Archives/edgar/data/1269238/000119312519274119/d814130dex991.htm),
[WLTW → WTW](https://investors.wtwco.com/node/24111/pdf), and
[FI → FISV](https://www.nasdaqtrader.com/TraderNews.aspx?id=DTN2025-32).
These identify the same continuing company; acquisitions and spin-offs require separate review
and are not automatically mapped to the acquirer. `GOOG` and `GOOGL` remain separate.


## Data tasks 1.3–1.7 and the NLP hand-off

If raw prices already exist, finalise without downloading again:

```bash
python -m src.data.fetch_prices --finalize-only
python -m src.data.build_labels --horizons 1 5
python -m src.data.quality --start 2019-01-01 --end 2023-12-31
python -m src.data.coverage --start 2019-01-01 --end 2023-12-31
```

`--finalize-only` consumes the entire existing raw CSV; `--start/--end` apply to downloads,
not to this offline mode. Both prices and labels use `write_table`. Prices keep original
row order and inclusive membership boundaries; overlapping spells and unknown tickers fail.
Labels preserve exactly the same keys/order as prices, use nullable `Int8`, and give flat
returns label 0. Unrequested horizons remain null to keep the frozen columns unchanged.

Labels look up the close on the exact h-th future NASDAQ session. A missing target price
produces a null label rather than extending the horizon to the next available row.
Non-session input rows also have null labels. Committed fixtures use generic business days,
including some US holidays; their pre-existing synthetic labels use row shifts. Regenerating
labels with the production calendar can therefore differ from those synthetic values.
The quality report identifies this difference and non-session rows; it does not rewrite
committed fixtures. Tests use actual NASDAQ sessions for calendar and horizon checks.

`compute_indicators(prices)` is an in-memory API: it returns keys plus six feature columns,
preserves order, uses each ticker's full history (including warm-up), and leaves early values
null. Volatility is 20-return sample SD, RSI uses Wilder smoothing seeded by a 14-period
arithmetic mean, MACD uses EMA12–EMA26 with EMA9 signal, and volume z-score uses a trailing
20-observation sample SD. Flat RSI is 50 and constant volume z-score is 0 after warm-up.
Missing sessions need review before using observed-row indicator windows. No backfill or
whole-sample scaling is performed. Model normalisation must be fitted on training folds only.
Features use closes at t, so predictions are made **after that session's close** for t+h.

`raw/quality_report.csv` lists internal missing sessions, missing in-index days and entire
member histories (when membership/bounds are supplied), non-session rows, zero volume,
adjusted close jumps over 40%, membership-flag mismatches, label formula mismatches and
per-ticker/year/horizon in-index class balance. A jump is a review candidate, not proof of an
unadjusted split. The CLI also includes `failed_tickers.csv` if available. Quality findings
are diagnostic and do not automatically delete data; review them before training/hand-off.
Explicit bounds are needed to detect absent prices at the edges of the intended experiment.

`raw/news_coverage.csv` measures matched `article_tickers` coverage per ticker/year on observed
in-index price rows. It requires the NLP artifact. It cannot prove raw FNSPID completeness,
and incomplete prices reduce its denominator. Resolve missing prices before interpreting it.
For task 1.8, optionally provide reviewed historical Investing.com earnings as a CSV with
`date` (ISO US/Eastern calendar day), `ticker`, `source_url`:

```bash
python -m src.data.coverage --earnings "$NDX_DATA_DIR/raw/investing_earnings.csv"
```

This adds `raw/earnings_coverage.csv`: candidate gaps from the earnings date through the next
NASDAQ session, restricted to observed in-index price rows. This wider diagnostic window
handles unknown announcement times; it is not a model feature or proof an article discusses
earnings. Events whose next session falls after `--end` are flagged by
`window_within_range=False` and excluded from candidate gaps. The
[coverage hand-off](../report/earnings_coverage_handoff.md) includes the reviewed Investing.com
sample and the findings from NLP's aggregate coverage table. These CSV reports are raw
diagnostics outside SCHEMA.md.

Run `notebooks/demo.ipynb` from the repo root or notebooks folder for the data-stage demo.
It does not write artifacts or download data. Model inference and attention tracing remain
pending the Modelling/NLP implementation, trained checkpoint and agreed inference interface.
The dataset report draft is `report/sections/02_dataset.md`; measured real-data statistics
are intentionally pending. A fresh-clone/full-environment reproducibility run (1.10) and
final trained-model demo (1.12) also remain pending those inputs. CHECKLIST boxes remain
unticked until changes are pushed to main and usable by the next owner, per repo convention.

## Download audit and recovery (15 November 2026)

The first real 2018–2023 download is now available locally under `data/` (gitignored).
It contains 187 membership spells / 180 ever-members, and 216,662 price/label rows across
152 downloaded tickers. The dataset is **incomplete**: the 2019–2023 audit identifies
14 entirely absent members and 10,603 missing in-index ticker-days, including partial
FOX/FOXA history. Do not treat these files as a complete unbiased research sample.
See `report/sections/02_dataset.md` for the measured audit and unresolved issues.

Additional local diagnostics are `raw/download_manifest.json` (bounds, versions, checksums),
`raw/failed_tickers_audit.csv` (which failed downloads intersect each research window), and
`raw/annual_label_balance.csv`. Acquisition logs are in `raw/fetch_prices*.log`.
Large-return/zero-volume flags are candidates for review, not automatic deletions.

A wholly blank separator in the live Wikipedia changes table is ignored; actual undated
events still fail. Parsed snapshots can be replayed without network:

```bash
python -m src.data.build_index_history --changes-csv "$NDX_DATA_DIR/raw/wikipedia_changes.csv" --current-csv "$NDX_DATA_DIR/raw/wikipedia_current.csv"
```

Both snapshots must correspond to the same download. To retry failed price downloads while
retaining successful histories, use the **same start/end as the original run**:

```bash
python -m src.data.fetch_prices --start 2018-01-01 --end 2024-01-01 --retry-failed
```

Retries still contact Yahoo; unsuccessful acquired/delisted histories require another
historical source. Successful retried tickers replace their entire old raw history. The
OHLC range guard allows only 1e-12 relative/absolute floating-point adjustment noise and
does not modify prices. NLOK is mapped to GEN using its
[issuer announcement](https://www.sec.gov/Archives/edgar/data/849399/000110465922115277/tm2229813d1_ex99-01.htm).
No acquiring-company or post-bankruptcy new-share prices were substituted for old securities.
Reports and the manifest describe this run; regenerate/review them after changing downloads.

## Archived-history recovery

A second pass recovered 11 additional ticker series from the FNSPID authors' archived
Yahoo prices, downloading only 3,493,618 bytes from their 589,525,596-byte ZIP via bounded
HTTP Range requests. The files are pinned to dataset revision
`bf9189c41527198897d1af3e17b1a0095279fc45`. Some stop at 2020-07-02; others at 2023-12-28.
They are partial recovery, not a complete new dataset.

The current combined dataset has **163 tickers / 227,104 rows**. Main-window missing
in-index rows decreased from 10,603 to **3,797**; CELG, SGEN and WBD are still wholly absent.
`raw/missing_prices.csv` lists exact absent keys; `raw/recovery_requests.csv` summarises
required intervals, including partially recovered histories and FOX/FOXA identity review.
`failed_tickers.csv` is deliberately retained as the Yahoo provider failure log: it is no
longer a list of wholly absent combined histories. Use the audit outputs for current gaps.

Download selected raw archive symbols, with a strict byte budget and no full-ZIP fallback:

```bash
python -m src.data.fetch_archive --tickers ALXN ANSS ATVI CERN CTXS EA MXIM MYL SPLK WBA XLNX --max-bytes 8000000
```

This command contacts Hugging Face. It does not import prices. A server ignoring Range,
incorrect ranges, oversized ZIP members or a byte-budget overrun fail instead of downloading
the whole archive. The downloaded raw files and provenance go to `raw/fnspid_recovery/`.

The import manifest is a CSV with `ticker,file,sha256,source_url,identity_url,identity_review`.
Every entry must have `identity_review=reviewed`, a source checksum, and issuer-symbol evidence.
Review security identity, not just the filename: the archive's GEN file has materially
different prices/volume from NortonLifeLock's history and **was not imported**. The archive's
NLOK file matches the provider's zero-volume transition day, so it cannot repair that finding.
Use the actual continuing company's history; never import the acquirer's security instead.

```bash
python -m src.data.recover_prices --manifest "$NDX_DATA_DIR/raw/fnspid_recovery/reviewed_sources.csv"
# After reviewing the preview/manifest:
python -m src.data.recover_prices --manifest "$NDX_DATA_DIR/raw/fnspid_recovery/reviewed_sources.csv" --apply
python -m src.data.quality --start 2019-01-01 --end 2023-12-31
python -m src.data.audit --start 2019-01-01 --end 2023-12-31
```

The importer accepts only entirely missing ticker series and refuses splicing archive/current
adjustment vintages for an existing ticker. Archived Yahoo raw OHLC is multiplied by
`adj close / close` and volume is preserved. It checks dates, keys, sessions, OHLC ranges,
finite values and membership, then writes final prices/labels through `write_table`.
Existing price values are preserved. Before applying, old prices/labels/raw prices are
copied into `raw/snapshots/<timestamp>/`. Reapplying the same manifest fails on ticker overlap,
rather than duplicating rows. `raw/recovery_provenance.json` records hashes, sources and backup.
Do not run a plain fresh Yahoo download over this recovery: that rebuilds Yahoo-only prices.

The audit command regenerates annual class balance, exact missing-date requests and current
file checksums. It preserves the original manifest as `raw/initial_download_manifest.json`.
Corporate identity problems, absent delisted history and genuine provider data issues remain
open; neither recovery nor passing tests certifies empirical completeness.
