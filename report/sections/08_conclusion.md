# Conclusion

> Owner: Everyone · Issue #51 · Due Fri 20 Nov, 12:00

<!-- Answer the research question honestly, limitations, future work. -->

## Limitations

- **Share classes counted twice.** Alphabet (GOOG, GOOGL), Fox (FOX, FOXA) and Liberty Global
  (LBTYA, LBTYK) are each two nodes with near-identical returns, so every prediction about
  them is scored twice and the effective sample is mildly inflated. Collapsing them would
  touch membership and prices, so it was not done. The graph does guard against it:
  `comention` and `corr` edges never link two classes of one company (SCHEMA.md decision 6).
- **Deal-arbitrage `corr` edges.** The strongest `corr` edges in the last fold (ADI–MXIM,
  AMD–XLNX, AZN–ALXN) come from pending stock-for-stock acquisitions. The target trades as a
  fixed ratio of the acquirer, so the correlation reflects a contract term, not the economic
  spillover the research question is about. They last only while the deal is pending (138–282
  shared days in fold 4). The 3.9 ablations rerun the model without them.
- **Folds are not independent replications.** The walk-forward protocol is a rolling window:
  504 training days, stepping 126. Adjacent folds share 75% of their training days (folds two
  apart 50%, three apart 25%; folds 0 and 4 none), so the five fitted models are strongly
  related. The five 126-day test windows are disjoint half-years, so each fold is scored on
  new data, but by correlated models. "Above 0.5 in 4 of 5 folds" therefore means
  *consistent across folds*, not *replicated five times*; for statistics over the training
  windows themselves, the effective number of independent observations is closer to two.
