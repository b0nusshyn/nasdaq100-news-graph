"""Train the proposed model across walk-forward folds and write predictions. Owner: Modelling. Issues #29, #36.

Output: `<data>/predictions/gat_lstm.parquet` (SCHEMA.md `predictions`). Normalisation
statistics are fitted on each training fold only.

    python -m src.models.train --config configs/gat_lstm.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.todo import todo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/gat_lstm.yaml"))
    p.add_argument("--horizon", type=int, choices=[1, 5], default=None, help="default: both")
    p.add_argument("--ablate", nargs="*", default=[], help="edge types or 'news' to drop (#41)")
    p.parse_args()
    todo(29, "training loop")


if __name__ == "__main__":
    main()
