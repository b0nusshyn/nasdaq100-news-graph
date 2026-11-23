---
license: mit
task_categories:
- tabular
tags:
- finance
- stock-market
- ohlcv
- yahoo-finance
size_categories:
- 10M<n<100M
---

# US Stock Market Daily OHLCV Dataset (Yahoo Finance)

Daily historical OHLCV data for US equities (retrieved via `yfinance`), stored in individual Parquet files per ticker.

## Data Layout
- `data/daily/{TICKER}.parquet`: Historical daily bars with columns `[date, open, high, low, close, adj_close, volume]`.
- `data/state/ingestion_state.parquet`: Pipeline metadata and tracking per ticker.

## Last Updated
2026-11-20 08:00 UTC
