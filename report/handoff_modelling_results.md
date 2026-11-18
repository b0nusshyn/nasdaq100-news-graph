# Modelling → Nonlada (Evaluation): ladder results for `06_results.md`, 18 November 2026

These are findings from the 3.8 full run and its tuned re-run (3.12). They're for you to fold into the Results section (#50); the section is yours, so the wording is a starting point.

**Sources:**
- Every number below is test within-day AUC, from `results/tables/ladder_3_8.md` (tuned config T1, seed 42) and `results/tables/tuning_gat_lstm.md`.
- PRs #92 (untuned run and the orientation check) and #93 (tuning and the re-run).
- Significance (4.9) is still pending. Until it lands, read "detectable" as "not distinguishable by eye across folds".

## The framing the data supports

**News adds a small amount of cross-sectional signal. Graph propagation does not add a detectable amount.**

| within-day AUC, mean of 5 folds | B5 price | B6 + news | proposed + graph | news step | graph step |
|---|---|---|---|---|---|
| 1 day | 0.499 | 0.505 | 0.509 | +0.006 (positive 4/5) | +0.004 (positive 4/5) |
| 5 days | 0.481 | 0.507 | 0.499 | +0.026 (positive 5/5) | −0.008 (positive 1/5) |

- **1 day:** the graph step is positive on average (+0.004). The ladder rises at both steps in only 3 of 5 folds. In fold 1 the graph step is −0.007, and in fold 4 the news step is −0.013.
- **5 days:** the graph step is negative (mean −0.008, negative in 4/5 folds).
- **Don't use a "propagation speed" framing** (a signal present at 1 day and gone by 5). It needed a clean 1-day ladder, and we don't have one.

## The 5-day news step: two readings

B6 beats B5 by 0.026 at 5 days, but B5 sits at **0.481, below chance in all 5 folds** (0.468–0.493). Part of that gap is plausibly news keeping the model away from B5's inverted bet, rather than news adding predictive signal. The two readings are very different, so state it that way.

What the measurements say:
- **B5's 5-day P(up) is negatively correlated with each stock's past 5-day return** (mean within-day Spearman −0.11): it leans towards reversal.
- **B6's 5-day P(up) is almost uncorrelated with it** (−0.017).
- **B6 itself is above 0.5 in only 3 of 5 folds** (mean 0.507).

## Baseline 5 below chance: checked, not a sign error

`tests/test_orientation.py` pins three invariants:
- stored labels equal labels recomputed from prices;
- every model's `y_true` equals the stored label;
- the AUC functions score the true forward return at 1.000 and its negative at 0.000.

**A hypothesis, not a conclusion:** a short-term reversal effect that was present in the training windows and not in the test windows. Raw reversal (minus the past 5-day return) scored 0.511–0.517 within-day in the training windows of folds 0–2, and 0.487–0.492 in their test windows.

**That explanation is incomplete, and the write-up has to say so.** Tuned B5 does clearly worse than plain reversal in folds 1–4: 0.471, 0.468, 0.482 and 0.493, against 0.487, 0.490, 0.497 and 0.509. It's making the reversal bet plus something else that hurts.

## The pre-registered horizon-pattern check did not hold

**The check:**
- **Registered:** 16 Nov 2026 (#36), after the fold-0 smoke run suggested it and before the full run.
- **Hypothesis:** propagation takes time, so the graph step should be near zero at 1 day and positive at 5 days.
- **Criteria:** at 5 days the ladder is monotone in at least 4 of 5 folds, and the mean graph step at 5 days exceeds the one at 1 day.

**It did not hold on any run:**
- **Untuned:** fold AUC, at three seeds.
- **Tuned:** within-day AUC (5-day ladder monotone in 1/5 folds; mean graph step −0.008 at 5 days vs +0.004 at 1 day) and fold AUC.

Report it unchanged, with its registration date and the verdict. Its wording is pinned by a test, so it can't be quietly edited. It was suggested by fold 0, and fold 0 is part of the evaluated data, so say that too.

## A finding about the protocol: validation gains were selection noise

- **Validation and test moved in opposite directions.** Tuning raised the proposed model's 1-day validation within-day AUC from 0.531 (the 3.8 setup) to 0.543 (T1). On test it fell slightly, from 0.512 to 0.509.
- **Validation peaks almost immediately.** Runs now train for 238–868 optimiser steps, yet validation picks a checkpoint at 14–42 steps in 7 of 10 of the proposed model's fold-horizons.
- **Reading:** with about 50 validation days per fold, both say the validation gains were mostly selection noise. Every validation score is the best of many checks on the same days.
- **Treat this as a finding about the protocol.** It bounds how far any tuning on this protocol can be trusted, including 4.13.
- **Context:** fewer optimiser steps wasn't the cause. Every batch_days-8 config beat the 3.8 setup on validation, and more steps still didn't move test.

## Fold dependence

This is now in the Limitations of `08_conclusion.md`. Adjacent folds share 75% of their training days, and the test windows are disjoint. Write "consistent across folds", never "replicated five times".
