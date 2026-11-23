# FINRA daily short-volume screen (2026-11-21)

New free data: FINRA Reg SHO consolidated daily short-sale volume, 2018-10 to 2023-12, 1,321
trading days requested, 23 returned 404 (no file), our tickers only (~153/day), SHA-256 of each
original file in `manifest.jsonl`, per-day Parquet in `days/`. Features lag one session
(day t-1's file is posted after its close). Coverage of member rows 99.3–100 %.

Pre-registered screen: hypothesis "more shorting → lower future return" (Diether, Lee & Werner
2009); a feature advances only if the negated feature's within-day AUC is > 0.5 in each of
2019, 2020 and 2021 (not assessment years).

| year | −short_ratio_1 (5d / 1d) | −short_ratio_5 (5d / 1d) | −short_abn (5d / 1d) |
|---|---|---|---|
| 2019 | 0.4937 / 0.4936 | 0.5039 / 0.4964 | 0.4993 / 0.4953 |
| 2020 | 0.5039 / 0.5005 | 0.4985 / 0.5027 | 0.5124 / 0.5083 |
| 2021 | 0.4939 / 0.4970 | 0.4950 / 0.5002 | 0.5007 / 0.4994 |
| 2022 | 0.5032 / 0.5000 | 0.5083 / 0.5056 | 0.5007 / 0.4933 |
| 2023 | 0.5095 / 0.5013 | 0.5126 / 0.4968 | 0.5028 / 0.4905 |

No feature passes (all within 0.49–0.51 in the screening years, signs inconsistent). The
slightly-above-0.5 2022/2023 values are not used: selecting on them would be choosing by the
assessment years. Branch stopped; no model fitted. Interpretation: in NASDAQ-100 mega-caps,
off-exchange short volume is dominated by market-maker activity and carries no stable
cross-sectional signal at 1–5 day horizons.

Also screened locally (no new data): sum of market-relative returns on abnormal-volume days in
the last 20 sessions and 5-day mean abnormal volume — within-day AUC 0.467–0.518 with sign flips
across 2019–2023. Not used.

Reproduce: `.venv/bin/python -B -m experiments.finra_short_v1.fetch` then
`.venv/bin/python -B -m experiments.finra_short_v1.features`.
