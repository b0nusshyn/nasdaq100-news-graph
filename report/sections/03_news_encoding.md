# News encoding

> Owner: NLP · Issue #48 · Due Fri 20 Nov

This section describes how raw financial news becomes the per-company, per-day features the
models consume: filtering FNSPID (task 2.1), matching articles to companies (2.3, 2.8),
assigning each article to a trading day (2.4), encoding it with FinBERT (2.2, 2.5),
aggregating to `news_features` (2.6), and measuring coverage (2.7).

## Source and filtering

We use FNSPID (Dong et al., 2024) from Hugging Face, pinned to revision
`bf9189c41527198897d1af3e17b1a0095279fc45`, file `Stock_news/nasdaq_exteral_data.csv`
(23 GB, 15,549,299 rows). The smaller `All_external.csv` was rejected: its news stops in 2020
and it has no rows for several large members (MSFT, COST, PYPL).

The company universe is every ticker with a NASDAQ-100 membership spell overlapping
2019–2023 (148 tickers), taken from the reconstructed membership history rather than today's
constituents, so delisted and removed firms keep their news (survivorship). Historical
symbols are mapped to one ticker per company (`FB` → `META`). Articles are kept by their
US/Eastern publication date, with a 7-day margin before 2019 so that late-December articles
can roll into the first trading days of the window. 239,274 rows remain.

## Matching articles to companies

An article is linked to a company in two ways: FNSPID's own symbol tag, and the company named
in the headline. Headline matching uses a reviewed alias table (`configs/ticker_aliases.csv`,
180 names for the 148 tickers), case-sensitive on word boundaries ("Intel" does not match
"Intelligence"), plus `$AAPL`, `NASDAQ: AAPL` and `Name (AAPL)` forms. Bare symbols never
match on their own, since many are ordinary words (`ON`, `TEAM`, `FAST`).

Headline matching raised the number of articles naming two or more companies — the input for
co-mention edges — from 34,432 to 45,512, and gave every one of the 148 tickers at least one
article (130 before).

Three corrections apply:

- **Share classes.** GOOG/GOOGL, FOX/FOXA and LBTYA/LBTYK are separate tickers but one
  company. FNSPID tagged Alphabet news GOOG-only 7,036 times and GOOGL-only 341 times, so a
  match on either class now links the article to both.
- **Symbol collision.** `AAL` is American Airlines on NASDAQ and Anglo American in London;
  442 of 5,078 AAL-tagged articles were mining and metals news. These tags are dropped unless
  the headline names the airline.
- **Old Fox issuer.** Before the 2019-03-19 spin-off, FOX/FOXA news refers to Twenty-First
  Century Fox and is assigned to TFCF/TFCFA.

**Validation (task 2.8).** In 50 randomly drawn article–company links, the company was the
headline subject in 20 and mentioned in the body in 20; 8 had no body to check and 2 were
wrong, i.e. 40 of 42 verifiable links (95%) were correct. Across all tagged articles with a
body, 2.7% never name the company; most of these are market-wide wraps. The sample with
verdicts is in `results/tables/match_spot_check.csv`.

The final `article_tickers` table has 263,760 article–company rows over 186,397 articles.

## Assigning articles to trading days

Features for day *t* may use only information available at that day's close (16:00 ET),
since predictions for *t + h* are made after the close. An article published at or after
16:00 ET, or on a non-trading day, therefore belongs to the next NASDAQ session.

FNSPID, however, gives only a date for 99.6% of articles (time stamped 00:00:00 UTC), so it is
unknown whether an article appeared before or after the close. To rule out look-ahead, an
article dated *D* is assigned to the first trading day **after** *D*, never to *D* itself.
Tests assert that no date-only article lands on or before its publication date.

This rule is conservative: news that the market absorbed during day *D* reaches the model one
day late. As a check, we compared it with the same-day assignment. On ticker-days with news,
the mean absolute next-day return is 1.09× that of ticker-days without news under our rule
and 1.13× under the same-day rule (5-day: 1.07× vs 1.09×). The conservative rule therefore
loses little signal, and part of the same-day rule's advantage is likely after-close news
leaking into the feature day.

## FinBERT encoding

Each article is encoded once with FinBERT (Araci, 2019; `ProsusAI/finbert`) from its **headline**: every
article has one while 23% have no body, FinBERT is trained on short financial sentences, and
a single text type keeps features comparable across articles. For each article we keep the
sentiment score *p*<sub>pos</sub> − *p*<sub>neg</sub> ∈ [−1, 1], the confidence (largest class
probability) and the 768-dimensional final-layer [CLS] embedding.

Encoding 186,834 headlines (before the 2.8 corrections removed 437 Anglo American articles) took about two minutes on an RTX 5060 (2,564 headlines/s in fp16,
2 GB of GPU memory), far below the 6 GPU-hour budget set in the plan.

Mean sentiment is +0.07 (19% of headlines above +0.5, 13% below −0.5), with the lowest yearly
mean in 2022 (+0.04), the year of the technology sell-off. As a raw predictor the signal is
weak: on ticker-days with clearly positive news the next-day return averages +0.09% against
−0.05% after clearly negative news, but the share of up-days barely moves (52.2% vs 52.0%,
base rate 52.0%) and the correlation between sentiment and next-day return is 0.01. This is
expected for large, liquid stocks and for news used with a one-day lag; the research question
compares two models that share these features, so it does not depend on the signal being
strong on its own.

## Aggregation to `news_features`

Articles are averaged per (date, company) with a plain mean (sentiment, confidence and
embedding). The table has one row for every in-index company-day in 2019–2023 (128,519
rows). Days without news have `has_news = false`, `n_articles = 0` and zeros elsewhere, so the
model can tell "no coverage" apart from neutral news.

## How the models use the features

Baseline 6 and the proposed model read the same news inputs per company-day, next to the price
indicators: `has_news`, `sentiment` and `confidence`, plus the FinBERT embedding reduced to 64
principal components. The PCA, like every normalisation statistic, is fitted on each training
fold only (`configs/gat_lstm.yaml`). News also shapes the graph: `comention` edges link
companies named in the same article, weighted on the training fold only.

The 3.9 ablations (`results/tables/ablations_3_9.md`, test within-day AUC, three seeds) show
what these inputs add at the 1-day horizon:

| Change | Within-day AUC | Seeds > 0 |
|---|---|---|
| News features, no graph (Baseline 6 − Baseline 5) | +0.0016 | 1 / 3 |
| News features, inside the graph model | +0.004 to +0.005 | 3 / 3 |
| `comention` edges, with news features | −0.0009 | 1 / 3 |

The effects are small, at the size of the seed-to-seed range (0.006):
- News helps only when the graph is present.
- The edges built from news add nothing at 1 day.
- At 5 days, `comention` edges lower within-day AUC in every seed (−0.005).

Section 7 asks what the graph actually passes on.

## Coverage

![News coverage by sector and year](../results/figures/news_coverage.png)

Half of all in-index company-days (49.6%) have at least one article. Coverage varies by year —
62% in 2019, 48% in 2020, 36% in 2021, 49% in 2022 and 53% in 2023 — and by sector, from 60%
in Communication Services and 52% in Information Technology down to 36% in Industrials and
35% in Utilities. The 2021 dip, with ten member companies receiving no article all year,
reflects FNSPID's collection rather than a quiet market. Energy, Real Estate and Financials
have one or two members, so their shares are noisy. Tables: `results/tables/news_coverage_*.csv`.

This sparsity motivates the graph: a company with no news of its own on a given day can still
receive a signal from related companies that have it.

## Limitations

- Date-only timestamps force the one-day lag above; timestamped commercial feeds (Bloomberg,
  Refinitiv, RavenPack) would allow same-day use without look-ahead.
- FNSPID tags include peripheral mentions (list articles, "stocks to consider" sections), so
  some links carry little firm-specific information.
- Coverage is uneven over time (2021) and across sectors.
- Only headlines are encoded; bodies may carry more information where they exist.
- Embeddings enter the models only as 64 PCA components. There was no ablation without
  them, so their share of the news effect is unknown.
- Articles are averaged with equal weight. A story syndicated under several URLs
  therefore counts several times against a story published once.

## References

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models.
  arXiv:1908.10063.
- Dong, Z., Fan, X., & Peng, Z. (2024). FNSPID: A comprehensive financial news dataset in time
  series. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data
  Mining (KDD '24)*. arXiv:2402.06698.
