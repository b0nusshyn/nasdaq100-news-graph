import pandas as pd

from src.nlp.filter_fnspid import OUT_COLUMNS
from src.nlp.match_tickers import article_id
from src.nlp.rebuild_timed_news import cutoff_sessions, prepare, source_articles, utc_values


def utc(*values):
    return utc_values(pd.Series(values), archive=True)


def test_actual_early_close_date_only_and_after_close_boundaries():
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("NASDAQ").schedule("2023-11-22", "2023-11-29")
    assert schedule.loc["2023-11-24", "market_close"] == pd.Timestamp("2023-11-24T18:00Z")
    pub = utc("2023-11-24T17:59Z", "2023-11-24T18:00Z", "2023-11-24T18:01Z",
              "2023-11-24 00:00:00 UTC", "2023-11-25", "2023-11-22T14:00Z")
    arrivals = utc(None, None, None, None, None, "2023-11-24T18:05Z")
    out = cutoff_sessions(pub, arrivals, schedule, publication_proxy=True)
    assert out.tolist() == list(pd.to_datetime(["2023-11-24"] + ["2023-11-27"] * 5))
    strict = cutoff_sessions(pub, arrivals, schedule, publication_proxy=False)
    assert strict.iloc[:5].isna().all()
    assert strict.iloc[5] == pd.Timestamp("2023-11-27")


def test_arrivals_require_offsets_and_cannot_predate_publication():
    schedule = pd.DataFrame({"market_close": [pd.Timestamp("2023-11-24T18:00Z")]},
                            index=pd.to_datetime(["2023-11-24"]))
    pub = utc("2023-11-24T14:00Z", "2023-11-24T14:00Z")
    arrivals = utc_values(pd.Series(["2023-11-24 15:00:00", "2023-11-24T13:00Z"]))
    assert pd.isna(arrivals.iloc[0])
    assert cutoff_sessions(pub, arrivals, schedule, publication_proxy=False).isna().all()


def test_source_recovery_keeps_original_links_and_quarantines_conflicting_times(tmp_path):
    url = "https://example.invalid/one"
    source = tmp_path / "local.csv"
    rows = pd.DataFrame({"Url": [url, url], "Date": ["2023-11-24T17:00Z", "2023-11-24T18:00Z"],
                         "Article_title": ["Headline", "Headline"]}).reindex(columns=OUT_COLUMNS)
    rows.to_csv(source, index=False)
    key = article_id(url)
    articles = source_articles(source, {key})
    assert articles.timestamp_conflict.tolist() == [True]
    assert articles.source_rows.tolist() == [2]
    links = pd.DataFrame({"article_id": [key, key, "unmatched"], "ticker": ["AAA", "BBB", "CCC"],
                          "date": pd.to_datetime(["2023-11-24"] * 3)})
    schedule = pd.DataFrame({"market_close": [pd.Timestamp("2023-11-24T18:00Z")]},
                            index=pd.to_datetime(["2023-11-24"]))
    out = prepare(links, articles, pd.Series(dtype="datetime64[ns, UTC]"), schedule, publication_proxy=True)
    pd.testing.assert_frame_equal(out[list(links)].assign(date=out.original_date), links)
    assert out.published_at.isna().all()
    assert not out.news_eligible.any()
    assert not out.timestamp_verified.any()
