"""One ticker per company: map historical symbols to the company's current symbol (SCHEMA.md).

Apply `canonical_ticker` wherever a ticker enters the pipeline: the Wikipedia membership
table, price downloads and news matching. Data extends RENAMES while building the membership
history (task 1.1, issue #6); chains such as A -> B -> C resolve to C.
"""

from __future__ import annotations

# old symbol -> new symbol, with the date the new symbol started trading
RENAMES: dict[str, str] = {
    "FB": "META",  # 2022-06-09
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
