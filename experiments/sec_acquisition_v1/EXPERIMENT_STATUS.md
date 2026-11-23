# SEC acquisition checkpoint

**Collection blocked and training deferred.** SEC_USER_AGENT is absent from the process, and no local SEC submissions, Company Facts or 2023 quarterly ZIP cache was found. No API probe, acquisition, spending or model fitting occurred. This does not establish provider unavailability.

The local submissions utility defaults to offline mode and reads cached root and historical pages. Network collection requires --fetch with a declared real project contact. The format check rejects common placeholders and cannot verify email ownership. The utility handles one reviewed CIK per invocation, checks the root issuer, narrows referenced history to 2019–2023, saves original JSON, records source URLs and SHA-256, and stops on 403/429. It waits at least one second before a request and retries select 5xx responses with bounded delays. Live HTTP behavior remains untested. Candidate metadata requires historical ticker and original accession review. The utility does not generate XBRL features from submissions metadata.

The existing proxy store was rebuilt separately from prior reviewed data. Eleven ticker keys have reviewed partial intervals across ten issuers. The remaining 137 are unresolved. Fourteen accessions and eighteen as-filed financial values across six contexts remain in the combined corpus. There are no new verified identities or observations. The unchanged strict policy still qualifies zero rows.

| Original 5d assessment | Rows | Filing events | Fundamentals |
|---|---:|---:|---:|
| 2022 | 11,925 | 194 (1.63%) | 53 (0.44%) |
| 2023 | 10,701 | 252 (2.36%) | 86 (0.80%) |

There was no change in coverage. Filing coverage spans three sectors in 2022 and four in 2023. Fundamental coverage remains PepsiCo in 2022 and PepsiCo plus Microsoft in 2023. The two-complete-session proxy is experimental and does not prove dissemination timing. All original assessment keys and labels were validated by the rebuild. Sector, ticker, form and missingness tables are in coverage_recheck.

Twelve offline tests passed. Genuine original-attention 2022 seed-42 GAT checkpoint inference again reproduced all 11,925 archived probabilities exactly. Accuracy was 55.4801%, matched majority 51.1614%, within-day AUC 0.543836, and MCC 0.111372. These remain retrospective results. No evidence supports either 0.63 target.

Once the real contact is exported to the process environment, run a single-root probe, followed by a batch of reviewed issuers across sectors. A provenance-preserved cache can instead be processed offline. Record original accession evidence before promoting data into the proxy store. Retain the training gate.

```sh
.venv/bin/python -B -m experiments.sec_acquisition_v1.submissions experiments/sec_acquisition_v1/cache 0000077476 --fetch --probe
.venv/bin/python -B -m experiments.sec_acquisition_v1.submissions experiments/sec_acquisition_v1/cache 0000077476
.venv/bin/python -B -m experiments.sec_coverage_resume.rebuild experiments/sec_acquisition_v1/reproduce
.venv/bin/python -B -m pytest -q -p no:cacheprovider experiments/sec_acquisition_v1/test_submissions.py experiments/sec_lagged_proxy/test_build.py experiments/sec_pit_v1/test_features.py
.venv/bin/python -B -m experiments.historical_demo_v1.replay development
```

Created submissions.py, test_submissions.py, EXPERIMENT_STATUS.md and coverage_recheck (ten audit files), all under this directory. Existing artifacts, notebooks, official datasets and metrics, and the 2026 lock were preserved. No Git operations.

## Terminal environment handoff

The collector's Python interpreter reported SEC_USER_AGENT absent. An explicit login-shell child also reported it absent. The process does not inherit variables exported only in an already running interactive terminal. No contact value was read or printed. No API request, new cache record, data acquisition, or training fit occurred. Twelve offline tests passed again. Coverage and the model registry remain unchanged.

From the configured terminal, run the command below. The first clause verifies that the shell exported a nonempty value to the same Python interpreter used by the collector. The second runs the one-issuer probe and caches the original root response on success.

```sh
cd /Users/nonladapream/nasdaq100-news-graph &&
.venv/bin/python -B -c 'import os, sys; sys.exit(0 if os.environ.get("SEC_USER_AGENT", "").strip() else "SEC_USER_AGENT is not exported to Python")' &&
.venv/bin/python -B -m experiments.sec_acquisition_v1.submissions experiments/sec_acquisition_v1/cache 0000077476 --fetch --probe
```

The cache path is experiments/sec_acquisition_v1/cache/CIK0000077476.json. A cached root can be validated by the assistant offline. Continuing network acquisition requires that the assistant collection process receive the same authorized environment value.

## PepsiCo cache validated and joined

The actual collector environment still lacks SEC_USER_AGENT. The user-run probe left a valid PepsiCo issuer response at cache/CIK0000077476.json, which was read offline. The recent array has 1,006 entries, includes all 20 2019–2023 domestic periodic candidates for this issuer, and needs no overlapping history pages for these dates. Two existing accessions matched filed date, period, form, issuer and Eastern acceptance time exactly. Eighteen new event-only accessions were staged. No financial facts were acquired from the submissions endpoint. The original SEC 2019 filing entity report names PEP and CIK 0000077476 and extends the dated reviewed interval back to February 15, 2019. [Dated SEC entity evidence](https://www.sec.gov/Archives/edgar/data/77476/000007747619000017/R1.htm)

The updated rebuild accepts a local cache and separately reviewed interval extensions. It uses only covered issuer intervals, requires timezone-aware acceptance, records original cache source and hash, and does not use submissions metadata to derive accounting values. The prior lag, original keys and labels, and strict qualification remain unchanged.

| Assessment | Rows | Mapped issuer rows | Filing-event rows | Fundamental rows |
|---|---:|---:|---:|---:|
| 2022 | 11,925 | 384 | 258 (2.16%) | 53 (0.44%) |
| 2023 | 10,701 | 923 | 310 (2.90%) | 86 (0.80%) |

The cohort still has 11 partially reviewed ticker keys across ten issuers and 137 unresolved ticker keys. Strict coverage remains zero. No automatic download or new training fit was performed. Twelve offline tests passed. [Updated coverage](coverage_pepsi/coverage_year.csv) · [Manifest](coverage_pepsi/manifest.json)

Run sequentially in the configured terminal to cache the first cross-sector batch using existing reviewed identities. Failure, including HTTP 403 or 429, ends the batch. Candidate lists printed by the collector include metadata only.

```sh
cd /Users/nonladapream/nasdaq100-news-graph
(
  set -e
  for cik in 0000909832 0001166691 0000882095 0000789019; do
    .venv/bin/python -B -m experiments.sec_acquisition_v1.submissions \
      experiments/sec_acquisition_v1/cache "$cik" --fetch
  done
)
```

The issuers are COST, CMCSA, GILD and MSFT. Their existing reviewed intervals bound what the rebuild can use. Additional periods need dated historical identity evidence. Fundamental coverage requires original financial documents or a provenance-preserved accession facts cache. Example local rebuild after collection:

```sh
.venv/bin/python -B -m experiments.sec_coverage_resume.rebuild \
  experiments/sec_acquisition_v1/coverage_batch_01 \
  --cache experiments/sec_acquisition_v1/cache \
  --identity-extensions experiments/sec_acquisition_v1/identity_extensions.csv
```

Files changed in this pass: experiments/sec_coverage_resume/rebuild.py and this status report. Created identity_extensions.csv and the ten distinct outputs in coverage_pepsi. The original strict module, cached SEC bytes, fallback workflow, checkpoints and official reports were preserved. No notebook or Git operations and no 2026 access.

## Cross-sector batch 01 verified

All four requested issuers cached successfully and parsed offline. Costco yielded 20 periodics, Comcast 20, Gilead 21, and Microsoft 20. Microsoft included a separate overlapping history page. PepsiCo's 20 remained available. Dated contemporaneous issuer filings allowed four interval starts to be extended while retaining the same issuer IDs. The reviewed key count remains 11 across ten issuers, with 137 unresolved. [Reviewed extensions](batch_01/identities.csv)

Added six quarterly contexts and 18 values from Comcast's 2022 September and 2023 June 10-Qs and Gilead's 2023 September 10-Q. These are as-filed consolidated US GAAP USD millions. Each current and comparison context covers one quarter, uses consolidated net income before allocation to noncontrolling interests, and becomes available together under its original accession. No prior quarter was backfilled. The rebuild checks additional facts against the issuer/accession and official source path and rejects duplicate contexts. [Financial evidence](batch_01/facts.csv)

| Assessment | Rows | Filing events | Fundamentals |
|---|---:|---:|---:|
| 2022 | 11,925 | 626 (5.25%) | 95 (0.80%) |
| 2023 | 10,701 | 530 (4.95%) | 178 (1.66%) |

The assessment keys and original labels matched archived inputs, and eligible dates were checked against stock-date keys. Strict dissemination coverage remains zero. Twelve offline tests passed. No training, notebook work, Git operations or 2026 scoring. Existing predictions, checkpoints, research artifacts and raw cache bytes were preserved. [Coverage tables and manifest](batch_01/coverage)

Next, collect three already reviewed issuers with domestic filing support. This will improve the filing-event inventory, while passing the screening gate also requires more verified issuer histories and original financial facts from a wider variety of sectors. Alphabet's two share classes share one issuer request. Run from the configured terminal:

```sh
(
  set -e
  cd /Users/nonladapream/nasdaq100-news-graph
  for cik in 0000320193 0001326801 0001652044; do
    .venv/bin/python -B -m experiments.sec_acquisition_v1.submissions \
      experiments/sec_acquisition_v1/cache "$cik" --fetch
  done
)
```

Reproduction with the current cache snapshot:

```sh
.venv/bin/python -B -m experiments.sec_coverage_resume.rebuild \
  experiments/sec_acquisition_v1/batch_01/reproduction \
  --cache experiments/sec_acquisition_v1/cache \
  --identity-extensions experiments/sec_acquisition_v1/batch_01/identities.csv \
  --reviewed-facts experiments/sec_acquisition_v1/batch_01/facts.csv
```

Later cache arrivals can intentionally expand results. The batch manifest identifies the exact inputs used here.

## Batch 02 verified

All Apple, Meta and Alphabet root responses validated. Meta and Alphabet also had overlapping cached history pages. The three issuers list 20, 20 and 21 2019–2023 periodic candidates respectively. Twelve accessions entered the proxy ledger within current reviewed ticker intervals, for 106 total accessions and 104 supported. Records outside reviewed intervals were held back. Alphabet share classes use a single CIK.

Dated SEC original filings supported new intervals for CSX, SBUX, AMGN and WBA. The registry now holds 15 reviewed partial ticker intervals across 14 issuers, with 133 unresolved. These four issuers await their own cache collection. Their filings span Industrials, Consumer Discretionary, Health Care and Consumer Staples in the local classification. See batch_02/new_identities.csv for the accession evidence.

Added 12 as-filed USD-million values from four current and comparative consolidated quarter contexts in Apple's July 2023 10-Q and Meta's September 2023 10-Q. The features first become eligible on August 9 and October 31, 2023 respectively. Prior-year comparisons are only used on or after those dates. No new Alphabet accounting facts were imported. Source and contextual details are in batch_02/facts.csv.

2022 filing-event coverage is 732/11,925 (6.14%), seven tickers and four sectors. Fundamentals are 95/11,925 (0.80%), two tickers and two sectors. 2023 event coverage is 954/10,701 (8.92%), nine tickers and four sectors. Fundamentals are 278/10,701 (2.60%), six tickers and four sectors. Strict eligibility remains zero. The build verified original labels and stock-date key equality. No training was justified. Fourteen tests passed, including terminal batch handling for simulated 403 and 429.

The reusable batch command accepts a dated issuer ledger and defaults to offline validation. --fetch requires a real SEC_USER_AGENT, which is still absent in this agent's environment. Fetch is sequential, resumes from cached files, deduplicates CIKs and stops on 403/429. The collector cannot verify new accounting facts from submissions alone. The rebuild accepts separately reviewed facts only.

Run the following from the configured terminal to collect and rebuild the next batch. Existing outputs are preserved by requiring a fresh destination.

```sh
cd /Users/nonladapream/nasdaq100-news-graph &&
.venv/bin/python -B -m experiments.sec_acquisition_v1.collect_batch \
  experiments/sec_acquisition_v1/cache \
  experiments/sec_acquisition_v1/batch_02/new_identities.csv --fetch &&
.venv/bin/python -B -m experiments.sec_coverage_resume.rebuild \
  experiments/sec_acquisition_v1/batch_03/coverage \
  --cache experiments/sec_acquisition_v1/cache \
  --identity-extensions experiments/sec_acquisition_v1/batch_01/identities.csv \
  --new-identities experiments/sec_acquisition_v1/batch_02/new_identities.csv \
  --reviewed-facts experiments/sec_acquisition_v1/batch_02/facts.csv
```

Created collect_batch.py, test_collect_batch.py and batch_02 (new issuer ledger, combined additional fact ledger and ten coverage outputs). Updated the rebuild script and this report. No notebooks, official artifacts, checkpoints, raw cache files or 2026 lock changed. No Git operations or purchases.
