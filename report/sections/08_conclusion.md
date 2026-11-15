# Conclusion

> Owner: Everyone · Issue #51 · Due Fri 20 Nov, 12:00

<!-- Answer the research question honestly, limitations, future work. -->

## Limitations

- **Share classes counted twice.** Alphabet (GOOG, GOOGL), Fox (FOX, FOXA) and Liberty Global
  (LBTYA, LBTYK) are each two nodes with near-identical returns, so every prediction about
  them is scored twice and the effective sample is mildly inflated. Collapsing them would
  touch membership and prices, so it was not done. The graph does guard against it:
  `comention` and `corr` edges never link two classes of one company (SCHEMA.md decision 6).
