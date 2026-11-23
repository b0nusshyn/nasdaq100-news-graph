# SEC FSDS pipeline — status (2026-11-21)

## Why this replaces the per-issuer workflow

Per-issuer submissions JSON gives filing *metadata only*; every financial value so far was
transcribed by hand (36 values, 6 tickers). At that rate the 25 % screening gate is out of reach:
after batch 03 fundamentals still cover 95/11,925 (0.80 %) of 2022 and 278/10,701 (2.60 %) of
2023 assessment rows. SEC's Financial Statement Data Sets carry, for **every** XBRL filer, each
submission's values **as filed under its own accession** with EDGAR `filed` date and
`accepted` timestamp; amendments/restatements are separate accessions. One sequential request
per quarter (22 for 2018q3–2023q4, ~2 GB streamed, ≤ ~115 MB on disk at a time) replaces
~140 issuer batches and the manual fact transcription.

## Verified state before this pipeline (independently recomputed)

| | 2022 (11,925 rows) | 2023 (10,701 rows) |
|---|---:|---:|
| Issuer mapped (batch 03 coverage) | 1,214 | 1,487 |
| Filing events | 1,200 (10.06 %) | 1,378 (12.88 %) |
| Fundamentals | 95 (0.80 %) | 278 (2.60 %) |
| Strict-policy rows | 0 | 0 |

Batch 03 (CSX, AMGN, SBUX, WBA) caches were present and parse; all 15 cached submissions files
are valid JSON with 19–21 2019–2023 periodic filings per issuer (root + history pages).

## Defect in the earlier proxy store (fixed here, old store left untouched)

`experiments/sec_lagged_proxy/build.py:93-100` writes every later supported filing's metric
columns onto its active rows, so a newer filing without extracted facts overwrites older
eligible fundamentals with NaN. `features.py` keeps two states: latest eligible filing (event)
and latest eligible filing **with values** (fundamentals, ≤ 130 sessions old). Tested.

## Pipeline (`experiments/sec_fsds_v1/`)

| File | Role | Network |
|---|---|---|
| `fetch.py` | resumable quarterly download; UA check; ≥2 s spacing; 5xx backoff; stop on 403/429/other; 2 GiB disk reserve; ZIP SHA-256 + size + URL in `extract/<q>.json` and `manifest.jsonl`; ZIP deleted after filtering unless `--keep-zips` | yes |
| `extract.py` | filter ZIP → `sub` (10-K/Q, 20-F, 40-F + amendments) and `num` (15 standard tags, quarter or annual, no co-registrant, no segment, standard taxonomy) | no |
| `features.py` | ledger with SEC_LAGGED_PROXY eligibility (imports `sec_lagged_proxy.build.eligible_session`, unchanged: two complete sessions after the later of filed date and Eastern acceptance date); backward-only identity from XBRL instance prefixes; two-state as-of store; PIT assertion; coverage on member keys and on the exact archived 2022/2023 assessment keys | no |
| `train_sec.py` | coverage-gated matched arms original / event / fund / both, 2 seeds × 2 windows | no |
| `test_fsds.py` | 8 tests: filtering, lag, same-filing comparatives, currency mismatch, amendments ignored, fact-less filing does not erase fundamentals, expiry, rename identity (FB→META), no future identity, prefix conflict/reuse | no |

Design choices: comparisons (YoY growth, margin change, EPS change) come from the same filing's
current vs prior-year comparative, never from a later filing; amendments are never used; no
price level enters any feature (adjusted closes embed later splits); missing stays NaN with
`sec_has_*` flags; fixed clip bounds in `train_sec.py`, no fitted imputation. Identity is a join
key, consulted only from filings dated on or before the row (≤ 400 days); short symbols (< 3
chars) need an exact prefix. SEC_LAGGED_PROXY remains an assumption, not proof of
dissemination time; filing (10-Q/10-K) usually lags the earnings press release (8-K) by days to
weeks, so these features measure post-filing information, not the announcement.

Controls already verified: with SEC channels present but empty, the `original` arm reproduces
the archived 2022 seed-42 predictions with max |Δp| = 0.0; the gate refuses an empty store.

## Blocker and next command

`SEC_USER_AGENT` is absent from the agent process; no identity is fabricated. From the
configured terminal (resumable; rerun after any stop):

```sh
cd /Users/nonladapream/nasdaq100-news-graph &&
.venv/bin/python -B -m experiments.sec_fsds_v1.fetch --fetch
```

Then, offline (agent can run these):

```sh
.venv/bin/python -B -m experiments.sec_fsds_v1.features --output experiments/sec_fsds_v1/store_v1
.venv/bin/python -B -m experiments.sec_fsds_v1.train_sec experiments/sec_fsds_v1/store_v1 experiments/sec_fsds_v1/runs_v1
```

Review `store_v1/unresolved_tickers.csv` and spot-check mapped CIKs against the 15 reviewed
intervals before trusting `train_sec` results. No SEC-augmented model has been fitted yet.

## Update 2026-11-21 (afternoon): data acquired, gate passed, experiment run

**Acquisition.** All 22 quarters (2018q3–2023q4) fetched with a user-supplied contact header
(not stored in any file), filtered and the ZIPs deleted; 23 MB of extracts with source SHA-256
in `extract/*.json` and `manifest.jsonl`. No 403/429.

**Fixes made after inspecting real data.**
- FSDS EPS `uom` is the currency (`USD`), not `USD/shares`: EPS now must match the revenue currency.
- Ticker reuse found and blocked in `identity_review.csv` (reason + evidence per row): GEN↔Genesis
  Healthcare, META↔Metaterra, WTW↔Weight Watchers, FISV↔Frank's International ('fi'),
  PCAR↔PACCAR Financial, FOX/FOXA↔21st Century Fox after the 2019-03-19 spin-off. Alias SYMC→GEN.
- Conflict rule: a symbol is ambiguous only if its filers *alternate* (A, B, A) within 800
  days; a clean succession (FOX, MRVL, MTCH reorganisations) maps to the newer filer.
- datetime unit / CIK dtype mismatches; ledger restricted to issuers whose filings carry one of
  our symbols (same result, 8.5 min → 12 s).

**Validation.** Mapping agrees with every hand-reviewed interval row (10,648 agree, 0 disagree,
172 unmapped). FSDS revenue and operating income equal all 32 hand-transcribed values exactly;
net income differs only by definition (parent-attributable `NetIncomeLoss` here vs. income
before NCI in the manual ledger). The point-in-time assertion passed. Stores: `store_v1`
(first build, ticker-reuse errors, kept for audit), `store_v2`, **`store_v3` (final)**.

**Coverage on the exact original assessment keys (store_v3).**

| | 2022 (11,925) | 2023 (10,701) |
|---|---:|---:|
| Issuer mapped | 98.0 % | 98.0 % |
| Filing events | 95.7 % | 95.9 % |
| Fundamentals | 93.0 % (102 tickers, 9 sectors) | 94.2 % (103 tickers, 9 sectors) |

Largest unmapped: JD, BIDU, TCOM, CHKP, LBTYA/K (foreign or prefix not matching early), INTU before 2020-11.

**Matched GAT-LSTM comparison (`runs_v1/`, 16 fits, 145 s, 5-day, gate passed).**

| Arm | 2022 s42 | 2022 s43 | 2023 s42 | 2023 s43 | Mean wdAUC |
|---|---:|---:|---:|---:|---:|
| original | 0.5438 | 0.5404 | 0.5035 | 0.4363 | 0.5060 |
| + events | 0.5326 | 0.5325 | 0.4856 | 0.5010 | 0.5129 |
| + fundamentals | 0.5331 | 0.5238 | 0.4757 | 0.4468 | 0.4949 |
| + both | 0.5001 | 0.5107 | 0.5007 | 0.4996 | 0.5028 |

Accuracy vs majority (2022 maj 0.5116, 2023 maj 0.5293): only `original` and `fund` s42 beat
majority in 2022 (0.5548, 0.5461); in 2023 no arm does. Full metrics in `runs_v1/ledger.jsonl`.
The `original` arm again reproduced the archive bit-exactly.

**Univariate check (no model, all member rows, within-day AUC for y_5d).** Every feature is
0.47–0.52 per year, with sign flips (revenue growth 0.474 in 2022, 0.507 in 2023). Operating
margin is the steadiest (> 0.5 in 4 of 5 years, ≤ 0.521). EPS change within 10 sessions of a
filing: 0.512.

**Conclusion.** Broad, validated, point-in-time SEC fundamentals do not improve 5-day direction
ranking for NASDAQ-100 stocks; the features carry at most ~0.01 AUC of unstable signal and the
GAT-LSTM overfits it. Branch stopped: the information is now available and was tested fairly,
so the negative result is the finding. No 0.6/0.63 evidence.
