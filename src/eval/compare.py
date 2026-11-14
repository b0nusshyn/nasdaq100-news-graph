"""Results table and figures from every model's predictions. Owner: Evaluation. Issue #45.

Reads `<data>/predictions/*.parquet`; writes `results/tables/*.csv|md` and `results/figures/*.png`.

    python -m src.eval.compare --output results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.todo import todo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--cost-bps-per-side", type=float, default=5.0, help="transaction cost per side, bps")
    p.parse_args()
    todo(45, "compare")


if __name__ == "__main__":
    main()
