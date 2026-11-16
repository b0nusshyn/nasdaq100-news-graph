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
