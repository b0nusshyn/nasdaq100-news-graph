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

Share classes are separate tickers in SCHEMA.md but one company, so a match on any class
adds the others (`source = "share_class"`): GOOG/GOOGL, FOX/FOXA, LBTYA/LBTYK, TFCF/TFCFA.
Before the 2019-03-19 Fox spin-off, `FOX`/`FOXA` refer to the old Twenty-First Century Fox
issuer and are remapped to `TFCF`/`TFCFA`. FNSPID tags that belong to another company with
the same symbol (AAL = Anglo American in London) are dropped by headline (`TAG_COLLISIONS`).

Trading day (task 2.4): the UTC timestamp is converted to US/Eastern; at or after 16:00 ET,
or on a non-trading day, the article belongs to the next NASDAQ session (the same calendar
`src.data` uses for prices). About 99.5% of FNSPID articles carry only a date (00:00:00 UTC):
they may have appeared after that day's close, so they go to the first session *after* their
date, never to the date itself (no look-ahead). Days outside [start, end] are dropped.
Membership is not applied per day here: models filter on `in_index`.

    python -m src.nlp.match_tickers

Writes `article_tickers.parquet` and `raw/article_tickers_audit.csv` (adds the UTC
`published_at` and the match `source`, for the 2.8 spot check).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from src.core.paths import CONFIGS_DIR

ALIASES_CSV = CONFIGS_DIR / "ticker_aliases.csv"
ET = "America/New_York"
CUTOFF_HOUR = 16  # NASDAQ close, 16:00 ET
# New Fox Corporation began trading on 2019-03-19; earlier FOX/FOXA history is TFCF/TFCFA.
FOX_SPINOFF = pd.Timestamp("2019-03-19", tz="America/New_York")
FOX_OLD = {"FOX": "TFCF", "FOXA": "TFCFA"}
# Share classes of one company: news about either applies to all (task 2.8 found FNSPID tags
# Alphabet news as GOOG-only 7,036 times and GOOGL-only 341 times).
SHARE_CLASSES = (("GOOG", "GOOGL"), ("FOX", "FOXA"), ("LBTYA", "LBTYK"), ("TFCF", "TFCFA"))
# FNSPID tags that collide with another company's symbol: ticker -> (headline pattern of the
# other company, headline pattern that still means ours). AAL is also Anglo American's London
# ticker; 2.8 found 442 of 5,078 AAL-tagged articles were mining/metals news.
TAG_COLLISIONS = {
    "AAL": (r"(?i)copper|platinum|\bPGMs?\b|Amplats|Anglo American|De Beers|diamond|iron ore|\bminers?\b"
            r"|mining|METALS-|Cochilco|Codelco|\bcoal\b|nickel|zinc|alumini?um",
            r"(?i)American Airlines|\bAAL\b|airline"),
}


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

    df = df[~_tag_collision(df)]
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
    out = out.drop_duplicates(["article_id", "ticker"])
    out = _expand_share_classes(out, universe)
    out = out.sort_values(["published_at", "article_id", "ticker"])
    return out.reset_index(drop=True)


def _tag_collision(df: pd.DataFrame) -> pd.Series:
    """FNSPID rows whose tag belongs to a different company with the same symbol elsewhere."""
    hit = pd.Series(False, index=df.index)
    titles = df["Article_title"].fillna("")
    for ticker, (other, ours) in TAG_COLLISIONS.items():
        theirs = titles.str.contains(other, regex=True) & ~titles.str.contains(ours, regex=True)
        hit |= (df["ticker"] == ticker) & theirs
    return hit


def _expand_share_classes(df: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """An article about one share class is about the company: add the other class(es)."""
    extra = []
    for group in SHARE_CLASSES:
        present = df[df["ticker"].isin(group)].drop_duplicates("article_id")
        for ticker in group:
            if ticker in universe:
                extra.append(present.assign(ticker=ticker, source="share_class"))
    if not extra:
        return df
    # Existing rows win over the added ones, so a real tag/title match keeps its source.
    return pd.concat([df, *extra], ignore_index=True).drop_duplicates(["article_id", "ticker"])


def _remap_old_fox(df: pd.DataFrame) -> pd.DataFrame:
    old = (df["published_at"] < FOX_SPINOFF) & df["ticker"].isin(list(FOX_OLD))
    df.loc[old, "ticker"] = df.loc[old, "ticker"].map(FOX_OLD)
    return df


def is_date_only(published_at: pd.Series) -> pd.Series:
    """FNSPID stamps most articles at exactly 00:00:00 UTC: only the publication date is known."""
    utc = published_at.dt.tz_convert("UTC")
    return (utc.dt.hour == 0) & (utc.dt.minute == 0) & (utc.dt.second == 0)


def assign_trading_day(published_at: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Trading day an article belongs to; tz-naive midnight dates, NaT if no session follows.

    Timed articles: convert to America/New_York; at or after 16:00 ET, or on a non-trading
    day, roll to the next session. Exactly 16:00:00 rolls forward (not in that day's close).
    The cutoff is 16:00 on every session, including early closes, as SCHEMA.md fixes it.

    Date-only articles (`is_date_only`, date D = publication date) cannot be placed against
    the cutoff: they may have appeared after D's close, so they go to the first session
    strictly after D. Never D itself, so no after-close article leaks into D.
    """
    et = published_at.dt.tz_convert(ET)
    undated = is_date_only(published_at)
    pub_date = published_at.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    day = et.dt.tz_localize(None).dt.normalize().where(~undated, pub_date)
    strictly_after = (et.dt.hour >= CUTOFF_HOUR) | undated

    days = pd.DatetimeIndex(trading_days).sort_values()
    pos = days.searchsorted(day.to_numpy(), side="left")
    on_day = (pos < len(days)) & (days[pos.clip(max=len(days) - 1)] == day.to_numpy())
    pos = pos + (strictly_after.to_numpy() & on_day)
    out = pd.Series(pd.NaT, index=published_at.index, dtype="datetime64[ns]")
    ok = pos < len(days)
    out[ok] = days[pos[ok]]
    return out


def build_article_tickers(articles: pd.DataFrame, aliases: dict[str, list[str]],
                          trading_days: pd.DatetimeIndex, start: str, end: str) -> pd.DataFrame:
    """Match, assign trading days, keep days in [start, end]. Keeps `published_at`/`source` for audit."""
    matched = match_articles(articles, aliases)
    matched["date"] = assign_trading_day(matched["published_at"], trading_days)
    keep = matched["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return matched.loc[keep, ["article_id", "date", "ticker", "published_at", "source"]].reset_index(drop=True)


def main() -> None:
    import argparse

    from src.core.io import read_table, write_table
    from src.core.paths import raw_dir
    from src.data.build_index_history import trading_days
    from src.nlp.filter_fnspid import universe

    p = argparse.ArgumentParser(description="Match FNSPID articles to tickers and trading days (tasks 2.3, 2.4).")
    p.add_argument("--input", type=Path, default=None, help="default: <data>/raw/fnspid_filtered.csv")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2023-12-31")
    args = p.parse_args()

    source = args.input or raw_dir() / "fnspid_filtered.csv"
    articles = pd.read_csv(source, usecols=["Url", "Date", "Article_title", "ticker"], dtype=str)
    membership = read_table("membership")
    members = universe(membership, args.start, args.end)
    aliases = {t: a for t, a in build_aliases(membership).items() if t in members}
    pad = pd.Timedelta(days=10)
    days = trading_days(pd.Timestamp(args.start) - pad, pd.Timestamp(args.end) + pad)

    out = build_article_tickers(articles, aliases, days, args.start, args.end)
    path = write_table(out[["article_id", "date", "ticker"]], "article_tickers")
    audit = raw_dir() / "article_tickers_audit.csv"
    out.to_csv(audit, index=False)
    print(f"{len(out):,} article-ticker rows, {out['article_id'].nunique():,} articles, "
          f"{out['ticker'].nunique()} tickers -> {path}")
    print(f"sources: {out['source'].value_counts().to_dict()}; audit with timestamps -> {audit}")


if __name__ == "__main__":
    main()
