# Data → Kornchawal (NLP), 15 November 2026

Prepared locally; delivery has not yet been confirmed. See `data_handoff.md` for Data QA.

## Ready inputs

- `membership.parquet`: 189 membership spells, 182 historical tickers.
- `prices.parquet` and `labels.parquet`: 238,195 matching date/ticker keys, 174 tickers.
- Main analysis window: 2019–2023. Filter prices by `in_index` and the date window.
- Five absent XLNX sessions (2022-02-14–18) are documented nontrading days;
  no prices have been invented. Other quote/adjustment QA remains open.
- Six recovered 2018 warm-up series are now included; ESRX 2018-12-21 is a documented halt.
- Task 1.6 passed with retained price/volume flags; read `data_quality_1_6.md` before training.

Use `src.core.io.read_table` and set `NDX_DATA_DIR` to the extracted data directory.
Use the full membership ticker universe to filter FNSPID, rather than today's constituents
or only tickers whose Yahoo downloads succeeded. Preserve the frozen `SCHEMA.md` contract.

## Matching and timing

- Apply `src.core.tickers.canonical_ticker`; `FB` maps to `META`.
- Keep `GOOG` and `GOOGL` separate. Do not map acquired companies to their buyers.
- Before 2019-03-19, old FOX/FOXA issuer histories use `TFCF`/`TFCFA`; the new
  `FOX`/`FOXA` issuer starts 2019-03-19. Entity matching must consider date and issuer,
  rather than applying a timeless FOX symbol alias.
- After 16:00 ET and nontrading-day articles belong to the next NASDAQ session.
- No news stays distinct from neutral sentiment through `has_news`.

## Needed back from NLP

1. Real `article_tickers.parquet` written with `write_table` after matching and alignment.
2. FNSPID source revision, extraction bounds, raw/filtered article counts and rejected matches.
3. `news_features.parquet` after encoding, with every required in-index observed price key.

Data can then run:

```bash
python -m src.data.coverage --start 2019-01-01 --end 2023-12-31
```

This writes `raw/news_coverage.csv` by ticker/year. It measures matched-news coverage,
not proof of raw FNSPID completeness. Investing.com earnings coverage remains pending
reviewed earnings input; no missing-news statistics have been fabricated.

The local delivery bundle is `raw/handoff_kornchawal.zip`; its `handoff_manifest.json`
lists checksums. The bundle is a review snapshot, not a final training-data release.
