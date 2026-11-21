# Prospective feed qualification

## Decision

**Live issuance remains locked.** A fresh public metadata query on November 21, 2026 UTC returned archive HEAD `2f8763bb9ed0c09fca6e871c8d660bdd8e4ce0ea`, last updated 2026-11-20 08:01:03 UTC. That is the same immutable revision already cached and precedes the November 20 signal close (21:00 UTC). The valid capture window ends at the November 23 open (14:30 UTC). Publication is still possible in that window, but this observed revision did not supply the current signal session. [Metadata response](archive_metadata.json) · [Coverage audit](audit_2026-11-20.json)

The 61-session audit verified the raw hashes for all 30 cohort files. There were 1,800 valid bars out of 1,830 required and no invalid present rows. Every ticker lacked November 20. Public archive metadata records revision publication. It does not certify instrument identity, redistribution rights for another application, consolidated venue coverage, dated split/dividend factors, or finalization of each bar.

## Alternatives checked

- The original Yahoo chart download probes received HTTP 429 and the original Stooq probe returned browser verification. The approved request in this cycle used a bounded Hugging Face metadata call. The full metadata response exceeded the 250 KB cap, so the successful request selected only the revision and modification fields.
- Alpaca's paid service is unnecessary for the first compatibility probe. Its documentation describes delayed historical SIP access for eligible free accounts when the query end is at least 15 minutes old. SIP is a more relevant candidate than single-venue IEX. No credentials were found in the current environment variable names or prior audit. No account was created. SIP bar eligibility, final regular-session semantics, adjusted-close continuity and stable identity still need to be verified. [Alpaca FAQ](https://docs.alpaca.markets/us/docs/market-data-faq) · [Historical bars contract](https://docs.alpaca.markets/us/reference/stockbars)
- Massive documents a midnight Eastern REST release for free historical daily bars. It needs an authorized key and documents reference and corporate-action endpoints. Compatibility with the frozen adjusted-close and volume features still requires comparison. No key or authenticated sample was available. [Release timing](https://massive.com/knowledge-base/article/on-the-free-basic-plan-when-can-i-pull-the-previous-trading-days-data) · [Instrument reference](https://massive.com/knowledge-base/article/does-massive-have-a-downloadable-security-master)

## Code and verification

Created root `AGENTS.md`, this status file, `archive.py`, `test_archive.py`, the bounded metadata response and the November 20 audit. The read-only price adapter verifies original acquisition SHA-256, preserves source OHLCV and adjusted values and distinguishes missing keys from present invalid rows. It emits a qualification audit and cannot approve a feed. The prior feature, target, frozen model and rank implementation are preserved. The real evaluator reported zero settled sessions. The adapter, prospective workflow and prior decision and target suites passed 11 tests. Ruff was rerun after a formatting fix. Disk retained 8.8 GiB. No prospective metrics, historical assessment scores, training fits, or 101-security 2026 outcomes were produced.

## Resume at the first valid input

Check the latest archive HEAD and capture relative to the live NASDAQ cutoff. A fresh revision can address the missing-date finding. To qualify that source, obtain verifiable per-release availability and a security-level and corporate-action mapping, and document its venue and adjustment semantics. Alternatively obtain an authorized free delayed-SIP credential or vetted provider export with immutable delivery provenance. Compare an overlapping 61-session sample of all 30 securities to the frozen archive, including corporate-action dates, adjusted return ratios, OHLC ranges, volume, and frozen feature differences. If compatibility and timing are established, use the existing [prospective runbook](../prospective_cohort_v1/report.md) to approve the source and issue once during the then-current valid window. A lapsed window becomes a skipped day. There is no qualified new development dataset and no justified extra modeling experiment before source validation.

## Overlap comparator and fresh recheck

A fresh bounded metadata read at approximately 2026-11-21 03:35 UTC returned bytes identical to the saved response. No new archive revision was available at that check. The November 20 signal remains blocked on the same missing all-cohort session and source qualification.

Added [overlap.py](overlap.py) and [test_overlap.py](test_overlap.py). They validate exact date/security keys and bar bounds, then report normalized adjusted OHLC differences, adjusted one-day return differences, volume differences, and differences in every frozen input feature. They neither generate labels nor evaluate a model. Run `.venv/bin/python -B -m experiments.prospective_feed_v1.overlap REFERENCE.parquet CANDIDATE.parquet NEW_REPORT.json --end YYYY-MM-DD`. Each input must contain precisely the 61 sessions ending on the supplied date and the full frozen cohort.

The [real-data positive control](overlap_control.json) compared two representations of the same pinned archive over 61 sessions through November 19. All 1,830 volumes matched, and 1,800 comparable one-day adjusted returns and every comparable frozen feature matched exactly. The negative test detected changed adjustment values, volume and missing keys. This verifies the comparison path. Alternate-provider compatibility is still unmeasured. Seven current feed and live tests and Ruff passed. Obtain an authorized vendor overlap or verifiable updated archive evidence before running a cross-provider comparison.

## Blocked audit

At 2026-11-21 03:36 UTC the bounded public metadata query again returned the exact saved revision. The environment and workspace checks found no authorized alternate feed sample or credentials. The live log is absent and the 101-security readiness record still disallows scoring. The same feed qualification dependency has persisted across three consecutive goal turns. The next meaningful step depends on a new qualified release or an authorized vendor sample with documented identity, adjustment, venue and release timing.
