"""One ticker per company: map historical symbols to the company's current symbol (SCHEMA.md).

Apply `canonical_ticker` wherever a ticker enters the pipeline: the Wikipedia membership
table, price downloads and news matching. Data extends RENAMES while building the membership
history (task 1.1, issue #6); chains such as A -> B -> C resolve to C.
"""

from __future__ import annotations

# Historical symbol -> current symbol (not a point-in-time symbol lookup).
RENAMES: dict[str, str] = {
    "FB": "META",  # 2022-06-09
    "PCLN": "BKNG",  # 2018-02-27
    "CTRP": "TCOM",  # 2019-11-05
    "WLTW": "WTW",  # 2022-01-10
    "FI": "FISV",  # 2025-11-11: Fiserv returned to its original ticker
    "NLOK": "GEN",  # 2022-11-08: continuing company, Gen Digital
}


def canonical_ticker(symbol: str) -> str:
    t = symbol.strip().upper()
    seen = {t}
    while t in RENAMES:
        t = RENAMES[t]
        if t in seen:
            raise ValueError(f"rename cycle involving {symbol!r}")
        seen.add(t)
    return t
