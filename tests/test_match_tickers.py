import pandas as pd

from src.nlp.match_tickers import (
    ALIASES_CSV,
    article_id,
    assign_trading_day,
    build_aliases,
    build_article_tickers,
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


# --- 2.4: trading-day alignment -------------------------------------------------------------

# Thu 2020-07-02, (Fri 07-03 holiday), Mon 07-06, Tue 07-07
DAYS = pd.DatetimeIndex(["2020-07-01", "2020-07-02", "2020-07-06", "2020-07-07"])


def utc(*stamps):
    return pd.Series(pd.to_datetime(list(stamps), utc=True, format="ISO8601"))


def test_before_close_same_day_after_close_next_day():
    # 2020-07-02 is EDT (UTC-4): 19:59 UTC = 15:59 ET, 20:00 UTC = 16:00 ET.
    got = assign_trading_day(utc("2020-07-02 19:59:59", "2020-07-02 20:00:00", "2020-07-02 23:30"), DAYS)
    assert got.tolist() == [pd.Timestamp("2020-07-02"), pd.Timestamp("2020-07-06"), pd.Timestamp("2020-07-06")]


def test_utc_after_midnight_is_still_previous_et_evening():
    # 03:00 UTC on 07-02 is 23:00 ET on 07-01, after close -> 07-02.
    assert assign_trading_day(utc("2020-07-02 03:00"), DAYS).tolist() == [pd.Timestamp("2020-07-02")]


def test_holiday_and_weekend_roll_to_next_session():
    # Holiday Friday morning, Saturday, Sunday evening -> Monday.
    got = assign_trading_day(utc("2020-07-03 14:00", "2020-07-04 15:00", "2020-07-05 23:00"), DAYS)
    assert set(got) == {pd.Timestamp("2020-07-06")}


def test_winter_offset_uses_est():
    days = pd.DatetimeIndex(["2020-01-06", "2020-01-07"])
    # January is EST (UTC-5): 20:30 UTC = 15:30 ET (same day), 21:00 UTC = 16:00 ET (next day).
    got = assign_trading_day(utc("2020-01-06 20:30", "2020-01-06 21:00"), days)
    assert got.tolist() == [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07")]


def test_no_following_session_is_nat():
    assert assign_trading_day(utc("2020-07-07 22:00"), DAYS).isna().all()


def test_build_article_tickers_window_and_schema():
    from src.core.schema import conform

    df = pd.DataFrame({
        "Url": ["a", "b", "c"],
        "Date": ["2020-07-01 21:00:00 UTC", "2020-07-07 14:00:00 UTC", "2020-07-07 22:00:00 UTC"],
        "Article_title": ["Apple news", "Intel news", "Nvidia news"],
        "ticker": ["AAPL", "INTC", "NVDA"],
    })
    out = build_article_tickers(df, ALIASES, DAYS, "2020-07-02", "2020-07-07")
    # a: after close 07-01 -> 07-02 (inside); b: 07-07; c: no next session -> dropped.
    assert dict(zip(out["ticker"], out["date"], strict=True)) == {
        "AAPL": pd.Timestamp("2020-07-02"), "INTC": pd.Timestamp("2020-07-07")}
    conform(out[["article_id", "date", "ticker"]], "article_tickers")



def test_date_only_goes_to_next_session_never_same_day():
    # Date-only 2020-07-02 (a session) -> 07-06; date-only holiday 07-03 -> 07-06.
    got = assign_trading_day(utc("2020-07-02 00:00:00", "2020-07-03 00:00:00"), DAYS)
    assert got.tolist() == [pd.Timestamp("2020-07-06"), pd.Timestamp("2020-07-06")]


def test_no_date_only_article_lands_on_or_before_its_date():
    df = pd.DataFrame({
        "Url": ["a", "b"], "Date": ["2020-07-01 00:00:00 UTC", "2020-07-06 00:00:00 UTC"],
        "Article_title": ["Apple news", "Intel news"], "ticker": ["AAPL", "INTC"],
    })
    out = build_article_tickers(df, ALIASES, DAYS, "2020-07-01", "2020-07-07")
    pub = pd.to_datetime(df["Date"].str[:10])
    assert len(out) == 2
    assert (out["date"].to_numpy() > pub.to_numpy()).all()


# --- 2.8: systematic mismatch fixes --------------------------------------------------------


def _one(url, date, title, tag):
    return pd.DataFrame({"Url": [url], "Date": [date], "Article_title": [title], "ticker": [tag]})


def test_share_class_tag_adds_the_other_class():
    out = match_articles(_one("g", "2021-05-03 14:00:00 UTC", "Big tech earnings week", "GOOG"), ALIASES)
    assert out.set_index("ticker")["source"].to_dict() == {"GOOG": "tag", "GOOGL": "share_class"}


def test_share_class_keeps_old_fox_issuer():
    out = match_articles(_one("f", "2019-02-01 15:00:00 UTC", "Media stocks", "FOXA"), ALIASES)
    assert set(out["ticker"]) == {"TFCF", "TFCFA"}


def test_aal_mining_headline_is_anglo_american_not_the_airline():
    aliases = {**ALIASES, "AAL": ["American Airlines"]}
    mining = _one("m", "2021-05-03 14:00:00 UTC", "Anglo American lifts copper output", "AAL")
    assert match_articles(mining, aliases).empty
    airline = _one("a", "2021-05-03 14:00:00 UTC", "American Airlines adds flights", "AAL")
    assert match_articles(airline, aliases)["ticker"].tolist() == ["AAL"]
    both = _one("b", "2021-05-03 14:00:00 UTC", "Airline fuel costs rise with copper", "AAL")
    assert match_articles(both, aliases)["ticker"].tolist() == ["AAL"]
