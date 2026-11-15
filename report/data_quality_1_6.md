# Task 1.6 — data-quality review

Local implementation and real-data review completed on 15 November 2026.
Status: **PASS_WITH_FLAGS**, not a claim that every vendor quote is correct.
The checklist remains unticked until the owner pushes the usable hand-off to main.

## Scope and results

The inclusive audit window is 2018-01-01–2023-12-31, including indicator warm-up.
There are 238,195 price rows across 174 tickers, with identical label keys;
membership has 189 spells across 182 historical tickers. The eight absent ticker
histories first enter the index after 2023 and are not missing members in this window.

- No malformed/nonfinite/nonpositive OHLC, negative volume, duplicate keys or non-session rows.
- No membership-flag or 1-/5-session label return/direction discrepancies.
- No unexplained missing in-index sessions or entirely absent in-window members.
- Six missing member/session keys are documented nontrading days: ESRX on
  2018-12-21 and XLNX on 2022-02-14–18. Membership is preserved and prices are not filled.
- Main 2019–2023 coverage is 128,519 observed of 128,524 expected member/session keys;
  the five remaining keys are the XLNX halt. This is a coverage result, not a quote certification.
- The original Yahoo failure log has 28 tickers; 20 now have recovered histories.
  The other eight have post-window membership only. Provider failures are retained as provenance.

## Adjustments and advisory flags

Six reviewed 2018 series (CA, DISH, ESRX, HOLX, QRTEA, SHPG) added 5,239 rows.
Recovered series use a single published split/dividend-adjustment basis and source SIP
volume convention per series. Checksum-pinned overlapping Yahoo quote OHLC is preserved
before applying published factors where available. No factor is fitted to returns.
The original adjusted archive was not casually spliced into a newer adjustment vintage.
Raw source, manifest hashes and pre-change backups are recorded in `raw/recovery_provenance.json`.

Seventeen adjusted-close moves exceed the diagnostic 40% threshold. Fifteen agree with
the pinned research snapshot within 1e-6 in daily return; both sources can share Yahoo
ancestry, so this is cross-archive corroboration, not independent verification. Two FER
moves lack a same-issuer comparison and remain flagged outside membership. Six flagged
moves are in-index (BIIB, DOCU, LCID, PDD, SIRI, ZM). No split or erroneous quote is inferred
merely from a large move; all are retained with their dispositions in `configs/quality_review.csv`.

Zero-volume findings cover BKR (1 row), FER (1,197), GEN (1), XEL (1).
Only GEN is in-index: 2019-11-05. Both archived NLOK and the research snapshot repeat the
zero, which does not establish that no trading occurred. Preserve the provider value and
flag it as unavailable/unverified volume in volume-sensitive analysis; test sensitivity
before using volume features. Outside-index anomalies can still affect warm-up features.
No replacement volume has been invented. Source rounding and SIP/Yahoo volume differences
remain documented limitations. This task implements and reviews checks; it cannot certify
every historical quote without an independent licensed archive.

## Annual label balance

Counts include observed in-index rows only. Null horizon targets are excluded from the up share.

| Year | In-index rows | 1-day up share | 5-day up share |
|---|---:|---:|---:|
| 2018 | 25,852 | 51.30% | 51.63% |
| 2019 | 25,956 | 54.76% | 58.36% |
| 2020 | 26,051 | 53.05% | 56.77% |
| 2021 | 25,695 | 52.20% | 54.24% |
| 2022 | 25,567 | 48.04% | 47.33% |
| 2023 | 25,250 | 52.68% | 55.42% |

Exact up/down/null counts are in `raw/quality_class_balance.csv`. No class resampling or
whole-sample normalization is performed by this review.

## Reproduce

```bash
python -m src.data.quality --start 2018-01-01 --end 2023-12-31 --strict
python -m src.data.audit --start 2019-01-01 --end 2023-12-31
ruff check .
pytest
```

Use the shared/extracted directory through `NDX_DATA_DIR`; do not run recovery against fixtures.
The quality CLI also runs offline on synthetic data with `NDX_DATA_DIR=tests/fixtures`;
`--strict` intentionally fails unreviewed synthetic anomalies, if any.
Quality outputs include `quality_report.csv`, `quality_summary.json`,
`quality_missing_sessions.csv`, `quality_download_status.csv`, `quality_class_balance.csv`
and `quality_review_register.csv` under the configured raw directory. Strict mode exits
nonzero for structural errors, unexplained member gaps or unreviewed/stale anomaly dispositions.
All final price/label artifacts were written through `write_table`; raw diagnostics remain CSV/JSON.

Validation: 153 tests and ruff passed using the installed environment. Tests cover holiday
boundaries, documented versus unexplained gaps, stale reviews, scope-specific findings,
prior-session jump calculation, recovered downloads and strict nonzero failure behavior.

Evidence: [ESRX trading suspension](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2018-152),
[December 2018 index removal](https://www.nasdaq.com/about/press-center/annual-changes-nasdaq-100-index-0),
[XLNX trading halt](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2022-23),
[XLNX index replacement](https://ir.nasdaq.com/news-releases/news-release-details/astrazeneca-plc-adr-join-nasdaq-100-index-beginning-february-22).
