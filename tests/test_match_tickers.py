import pandas as pd

from src.nlp.match_tickers import (
    ALIASES_CSV,
    article_id,
    build_aliases,
    match_articles,
    title_matches,
)

ALIASES = {
    "AAPL": ["Apple"],
    "INTC": ["Intel"],
    "META": ["Meta", "Meta Platforms", "Facebook"],
    "GOOG": ["Alphabet", "Google"],
    "GOOGL": ["Alphabet", "Google"],
    "ON": ["ON Semiconductor", "onsemi"],
    "FOX": ["Fox Corp"],
    "FOXA": ["Fox Corp"],
    "TFCF": ["21st Century Fox"],
    "TFCFA": ["21st Century Fox"],
    "NVDA": ["Nvidia"],
}


def test_title_word_boundaries_and_case():
    assert title_matches("Intel beats estimates", ALIASES) == {"INTC"}
    assert title_matches("Artificial Intelligence stocks rally", ALIASES) == set()
    assert title_matches("Apple, Nvidia lead tech higher", ALIASES) == {"AAPL", "NVDA"}
    assert title_matches("apple pie recipes", ALIASES) == set()


def test_bare_symbols_do_not_match_but_cashtags_do():
    assert title_matches("Lights ON as markets open", ALIASES) == set()
    assert title_matches("Why $ON jumped today", ALIASES) == {"ON"}
    assert title_matches("Apple Inc. (NASDAQ: AAPL) hits record", ALIASES) == {"AAPL"}
    assert title_matches("Lights ON for chipmakers; (ON) gains", ALIASES) == {"ON"}
    assert title_matches("Is Nvidia (NVDA) a buy?", ALIASES) == {"NVDA"}


def test_share_classes_both_match():
    assert title_matches("Google parent Alphabet reports", ALIASES) == {"GOOG", "GOOGL"}


def test_longest_alias_wins():
    assert title_matches("Meta Platforms cuts jobs", ALIASES) == {"META"}


def articles():
    return pd.DataFrame({
        "Url": ["u1", "u1", "u2", "u3", "u4"],
        "Date": ["2020-06-05 13:00:00 UTC", "2020-06-05 13:00:00 UTC", "2021-01-04 15:00:00 UTC",
                 "2019-02-01 15:00:00 UTC", "2020-03-02 15:00:00 UTC"],
        "Article_title": ["Apple and Intel team up", "Apple and Intel team up",
                          "Stocks to watch", "Fox Corp earnings preview", "Fox Corp earnings"],
        "ticker": ["AAPL", "INTC", "NVDA", "FOXA", "FOXA"],
    })


def test_match_articles_combines_tags_and_titles():
    out = match_articles(articles(), ALIASES)
    assert list(out.columns) == ["article_id", "published_at", "ticker", "source"]
    u1 = out[out["article_id"] == article_id("u1")]
    # Tagged twice and named in the title: one row per ticker, tag wins.
    assert sorted(u1["ticker"]) == ["AAPL", "INTC"]
    assert set(u1["source"]) == {"tag"}
    assert out.duplicated(["article_id", "ticker"]).sum() == 0
    assert str(out["published_at"].dt.tz) == "UTC"


def test_title_only_match_and_tag_only_match():
    df = articles()
    df.loc[2, "Article_title"] = "Nvidia and Apple partner"
    out = match_articles(df, ALIASES)
    u2 = out[out["article_id"] == article_id("u2")].set_index("ticker")["source"].to_dict()
    assert u2 == {"NVDA": "tag", "AAPL": "title"}


def test_fox_before_spinoff_is_old_issuer():
    out = match_articles(articles(), ALIASES)
    before = out[out["article_id"] == article_id("u3")]
    after = out[out["article_id"] == article_id("u4")]
    assert set(before["ticker"]) == {"TFCF", "TFCFA"}
    assert set(after["ticker"]) == {"FOX", "FOXA"}


def test_tickers_outside_aliases_are_dropped():
    df = articles().assign(ticker="ZZZZ")
    out = match_articles(df, ALIASES)
    assert set(out["source"]) == {"title"}


def test_alias_table_covers_members():
    members = pd.DataFrame({"ticker": ["AAPL", "GOOG", "GOOGL", "NOTREAL"]})
    aliases = build_aliases(members)
    assert set(aliases) == {"AAPL", "GOOG", "GOOGL"}
    table = pd.read_csv(ALIASES_CSV, dtype=str)
    assert table["alias"].notna().all() and not table.duplicated().any()
