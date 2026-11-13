"""Download FNSPID and keep NASDAQ-100 ever-members, 2019–2023. Owner: NLP. Issue #10.

Output: `<data>/raw/fnspid_filtered.csv` (raw, not a SCHEMA.md artifact).

    python -m src.nlp.filter_fnspid
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.todo import todo

HF_DATASET = "Zihan1004/FNSPID"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=None, help="default: <data>/raw/fnspid_filtered.csv")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2023-12-31")
    p.parse_args()
    todo(10, "filter FNSPID")


if __name__ == "__main__":
    main()
