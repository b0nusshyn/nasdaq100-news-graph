# Data hand-off status — 15 November 2026

Owner: Chotima. Local working tree; no push or issue closure has been performed.

| Item | Current result |
|---|---|
| Membership | 189 spells / 182 ever-members; daily count 101–103 in 2019–2023 |
| Prices and labels | 238,195 matching keys / 174 tickers; 2018-01-02 through 2023-12-29 |
| Main-window coverage | 128,519 observed of 128,524 expected member/session keys |
| Remaining main-window gaps | Five XLNX sessions, 2022-02-14 through 2022-02-18; documented trading halt before index removal |
| Unexplained main-window quote gaps | Zero; this does not certify quote accuracy or adjustment consistency |
| Historical issuer identity | Old FOX/FOXA separately represented as TFCF/TFCFA through 2019-03-18 |
| Research download | 185,296,627-byte checksum-pinned daily archive downloaded successfully |
| 2018 warm-up recovery | 5,239 rows for six reviewed series applied; ESRX 2018-12-21 documented halt |
| Code validation | 153 tests and ruff passed in the installed environment |
| Demo | Data-stage notebook prepared; NLP/model artifacts remain absent |

## Before training or evaluation

Task 1.6 checks and disposition review are complete locally: **PASS_WITH_FLAGS**.
Read [the quality report](data_quality_1_6.md) for retained price/volume limitations.
There are zero unexplained in-index quote gaps across 2018–2023. Six absent sessions
are documented nontrading days; no prices have been fabricated. Source adjustment
rounding and SIP/Yahoo volume differences remain limitations. Preserve flags in
volume-sensitive and warm-up analysis, and rerun strict quality checks after changes.
The main audit still covers 2019–2023, while the quality summary includes 2018.

## Other Data tasks

- 1.8: coverage code exists; real FNSPID article_tickers and reviewed historical Investing.com
  earnings are still absent. No empirical coverage report or historical earnings pull is complete.
- 1.9/1.12: no trained checkpoint, inference interface or attention trace exists in this checkout.
  The data-stage demo is prepared; a complete trained-model demo cannot yet run.
- 1.10: clean source-copy tests passed, but dependencies were reused. A fresh clone plus clean
  dependency installation and a full real-data end-to-end run have not been completed.
- 1.11: measured acquisition/recovery statistics are in sections/02_dataset.md. Completeness
  claims, final coverage findings and model results remain pending.

No suitable historical-stock-data connector was found in the available plugin directory.
Existing institutional WRDS/CRSP access or an already licensed delisted-history provider may
supply remaining data; no subscription, paid request, credential use or external team message
has been initiated. Public provider documentation confirms archives exist, not that every
required security/date has been checked in those archives.

## Provenance and recovery

- `raw/initial_download_manifest.json`: original Yahoo-only download audit.
- `raw/download_manifest.json`: current file hashes and main-window missing counts.
- `raw/fnspid_recovery/reviewed_sources.csv`: reviewed source hashes and issuer filing evidence.
- `raw/recovery_provenance.json`: applied archive imports and backup location.
- `raw/snapshots/`: pre-recovery price/label/raw-price backups.
- `raw/failed_tickers.csv`: original Yahoo failures, retained even when an archive partially recovered a ticker.
- `raw/reproducibility_source_snapshot.json`: clean source-copy check result and its limits.

Sources: [FNSPID author repository](https://github.com/Zdong104/FNSPID_Financial_News_Dataset),
[SEC FOX spin-off filing](https://www.sec.gov/Archives/edgar/data/1308161/000119312519079034/d721340d8k.htm),
[EODHD delisted-history documentation](https://eodhd.com/financial-apis/delisted-stock-companies-data-2),
[CRSP database documentation](https://www.crsp.org/crsp_pdf/crsp-us-stock-indexes-databases-data-descriptions-guide-crspaccess/).
