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


# Share classes of one company. SCHEMA.md keeps them as separate tickers, but news matching tags
# every class of a company together (task 2.8) and their returns are near-identical, so relation
# edges (`comention`, `corr`) must not link a company to itself: use `same_company`.
SHARE_CLASSES: tuple[tuple[str, ...], ...] = (
    ("GOOG", "GOOGL"), ("FOX", "FOXA"), ("LBTYA", "LBTYK"), ("TFCF", "TFCFA"),
)
_COMPANY = {t: group[0] for group in SHARE_CLASSES for t in group}


def company(ticker: str) -> str:
    """The company a ticker belongs to: its first-listed share class, else the ticker itself."""
    return _COMPANY.get(ticker, ticker)


def same_company(a: str, b: str) -> bool:
    """True for two share classes of one company (or the same ticker twice)."""
    return company(a) == company(b)


def canonical_ticker(symbol: str) -> str:
    t = symbol.strip().upper()
    seen = {t}
    while t in RENAMES:
        t = RENAMES[t]
        if t in seen:
            raise ValueError(f"rename cycle involving {symbol!r}")
        seen.add(t)
    return t
