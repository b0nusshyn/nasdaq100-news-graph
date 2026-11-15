"""Map each article to the tickers it is about. Owner: NLP. Issues #16, #17.

Output: `article_tickers.parquet` (SCHEMA.md), already aligned to trading days.
Modelling builds co-mention edges from it (#21).

An article matches a ticker in two ways (the `source` column records which):
- `tag`: FNSPID's own `Stock_symbol` for that row, already canonical from task 2.1.
- `title`: the headline names the company. Aliases come from `configs/ticker_aliases.csv`
  and match case-sensitively on word boundaries ("Intel" does not match "Intelligence").
  Cashtags (`$AAPL`), exchange tags (`NASDAQ: AAPL`) and a symbol in parentheses
  (`Fastenal (FAST)`, the Zacks headline style) also match. Bare symbols never match
  on their own (`ON`, `A`, `TEAM` are ordinary words); only aliases listed in the CSV do.

One company name can map to several share classes (`Alphabet` -> GOOG and GOOGL), which
SCHEMA.md keeps as separate tickers. Before the 2019-03-19 Fox spin-off, `FOX`/`FOXA`
refer to the old Twenty-First Century Fox issuer and are remapped to `TFCF`/`TFCFA`.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from src.core.paths import CONFIGS_DIR
from src.core.todo import todo

ALIASES_CSV = CONFIGS_DIR / "ticker_aliases.csv"
# New Fox Corporation began trading on 2019-03-19; earlier FOX/FOXA history is TFCF/TFCFA.
FOX_SPINOFF = pd.Timestamp("2019-03-19", tz="America/New_York")
FOX_OLD = {"FOX": "TFCF", "FOXA": "TFCFA"}


def article_id(url: str) -> str:
    """Stable id from the FNSPID URL: first 16 hex chars of its SHA-1."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def build_aliases(membership: pd.DataFrame, path: Path = ALIASES_CSV) -> dict[str, list[str]]:
    """Ticker -> name variants to search for, limited to tickers present in `membership`."""
    table = pd.read_csv(path, dtype=str)
    members = set(membership["ticker"].astype(str))
    table = table[table["ticker"].isin(members)]
    return {t: sorted(g["alias"].unique()) for t, g in table.groupby("ticker")}


def _alias_pattern(aliases: dict[str, list[str]]) -> tuple[re.Pattern, dict[str, list[str]]]:
    by_alias: dict[str, list[str]] = {}
    for ticker, names in aliases.items():
        for name in names:
            by_alias.setdefault(name, []).append(ticker)
    # Longest first so "Meta Platforms" wins over "Meta" at the same position.
    names = sorted(by_alias, key=len, reverse=True)
    body = "|".join(re.escape(n) for n in names)
    return re.compile(rf"(?<![\w.$-])(?:{body})(?![\w-])"), by_alias


def _symbol_pattern(tickers: list[str]) -> re.Pattern:
    body = "|".join(re.escape(t) for t in sorted(tickers, key=len, reverse=True))
    tag = rf"(?:\$|\b(?:NASDAQ|Nasdaq|NYSE)\s*:\s*)({body})\b"
    paren = rf"\(({body})\)"
    return re.compile(rf"{tag}|{paren}")


def title_matches(title: str, aliases: dict[str, list[str]]) -> set[str]:
    """Tickers named in one headline (convenience wrapper for tests and spot checks)."""
    alias_re, by_alias = _alias_pattern(aliases)
    return _match_title(title, alias_re, by_alias, _symbol_pattern(list(aliases)))


def _match_title(title: str, alias_re: re.Pattern, by_alias: dict[str, list[str]],
                 symbol_re: re.Pattern) -> set[str]:
    found: set[str] = set()
    for m in alias_re.finditer(title):
        found.update(by_alias[m.group(0)])
    for m in symbol_re.finditer(title):
        found.add(m.group(1) or m.group(2))
    return found


def match_articles(articles: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Return (article_id, published_at, ticker, source), one row per article-ticker match.

    `articles` has the 2.1 columns `Url`, `Date` (UTC string), `Article_title` and `ticker`
    (FNSPID's tag), one row per FNSPID row. A ticker found by both the tag and the title keeps
    `source = "tag"`. Only tickers in `aliases` are returned.
    """
    universe = set(aliases)
    df = articles.dropna(subset=["Url", "Date"]).copy()
    df["article_id"] = df["Url"].map(article_id)
    df["published_at"] = pd.to_datetime(df["Date"].str.replace(" UTC", "", regex=False), utc=True)

    tags = df.loc[df["ticker"].isin(universe), ["article_id", "published_at", "ticker"]]
    tags = tags.assign(source="tag")

    alias_re, by_alias = _alias_pattern(aliases)
    symbol_re = _symbol_pattern(list(aliases))
    heads = df.drop_duplicates("article_id")[["article_id", "published_at", "Article_title"]]
    hits = heads["Article_title"].fillna("").map(lambda t: sorted(_match_title(t, alias_re, by_alias, symbol_re)))
    titles = heads.assign(ticker=hits).explode("ticker").dropna(subset=["ticker"])
    titles = titles[["article_id", "published_at", "ticker"]].assign(source="title")

    out = pd.concat([tags, titles], ignore_index=True)
    # One timestamp per article: FNSPID rows of the same URL can differ slightly.
    out["published_at"] = out.groupby("article_id")["published_at"].transform("min")
    out = _remap_old_fox(out)
    out = out[out["ticker"].isin(universe)]
    out = out.sort_values("source", kind="stable")  # "tag" before "title"
    out = out.drop_duplicates(["article_id", "ticker"]).sort_values(["published_at", "article_id", "ticker"])
    return out.reset_index(drop=True)


def _remap_old_fox(df: pd.DataFrame) -> pd.DataFrame:
    old = (df["published_at"] < FOX_SPINOFF) & df["ticker"].isin(list(FOX_OLD))
    df.loc[old, "ticker"] = df.loc[old, "ticker"].map(FOX_OLD)
    return df


def assign_trading_day(published_at: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Convert to America/New_York; after 16:00 ET, or on a non-trading day, roll to the next trading day."""
    todo(17, "timestamp alignment")
