"""Run baselines on the shared walk-forward folds and write their predictions. Owner: Evaluation. Issue #39.

Output: `<data>/predictions/<baseline>.parquet` (SCHEMA.md `predictions`).

    python -m src.eval.run_baselines --all
    python -m src.eval.run_baselines --only logistic xgboost
"""

from __future__ import annotations

import argparse

from src.core.todo import todo

# Keep in sync with src.models.baselines.BASELINES (not imported: heavy deps would slow --help).
BASELINE_NAMES = ("buy_and_hold", "arima", "logistic", "xgboost", "lstm_price", "lstm_news")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true")
    which.add_argument("--only", nargs="+", choices=BASELINE_NAMES)
    p.add_argument("--horizon", type=int, choices=[1, 5], default=None, help="default: both")
    p.parse_args()
    todo(39, "run_baselines")


if __name__ == "__main__":
    main()
