# Method

> Owner: Modelling · Issue #47 · Due Fri 20 Nov

<!-- Graph construction (three edge types), CompanyEncoder, GAT, training setup. -->

## Graph construction

### `corr` edges

<!-- Draft note from task 3.4 (#22); expand in 3.11. Numbers: results/tables/corr_report.md. -->

The weight is the partial correlation of two firms' daily returns given the equal-weight
market, with each stock's beta fitted on the fold's training window (SCHEMA.md decision 7).
The choice of market adjustment matters most in volatile regimes. In folds 0–2 it barely
moves the selected edges, but in folds 3–4, whose training windows include the 2022 rate
shock, beta dispersion widens (p10–p90 from 0.6–1.4 to 0.4–1.7), and subtracting the market
with beta = 1 instead of the fitted beta raises the median selected edge from 0.23 to 0.32–0.34.

**Validation: pending acquisitions.** Without any deal data, `corr` edges pick up the
stock-for-stock acquisitions between index members. In fold 4, ADI–MXIM, AMD–XLNX and AZN–ALXN
are each the top peer of one side (partial correlation 0.72–0.88), on 164, 282 and 138 shared
days, just above the 126-day minimum: those edges exist only for the months the deal was pending
before the target delisted. ADI–MXIM and AMD–XLNX are already selected in folds 0–2, as
genuine semiconductor peers, before the deals dominate. The all-cash MSFT–ATVI deal, pending
throughout folds 3–4, is not picked up (−0.04 in fold 4): a cash offer pins the target to a
fixed price rather than to the acquirer's stock. So the edges respond to how two prices are
actually tied together, not to a label saying a relationship exists. See Limitations for why
these particular edges are a different mechanism from the one under test.
