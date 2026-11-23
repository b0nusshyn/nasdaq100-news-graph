import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from src.nlp.encode_external_candidate import aggregate_policies, validate_features
from src.nlp.encode_news import EMB_COLUMNS
from src.nlp.external_news_candidate import align_candidate, coverage_table, read_shards
from src.nlp.match_tickers import article_id


def test_shards_preserve_exact_ids_quarantine_conflicts_and_keep_missing_coverage(tmp_path):
    urls = [f"https://example.invalid/{n}" for n in range(5)]
    first = pd.DataFrame({
        "Url": urls[:4], "Date": ["2023-11-23", "2023-11-24T17:00Z", "2023-11-23", "2023-11-23"],
        "Article_title": ["Plain headline", "Early close day", "Duplicate time", "Company News for Nov 24, 2023"],
    })
    second = first.iloc[[0, 2]].copy()
    second.loc[2, "Date"] = "2023-11-24"
    paths = [tmp_path / "part0.parquet", tmp_path / "part1.parquet"]
    for frame, path in zip((first, second), paths, strict=True):
        frame.to_parquet(path, index=False)
    ids = [article_id(url) for url in urls]
    articles = read_shards(paths, set(ids))
    assert len(articles) == 4
    assert articles.set_index("article_id").loc[ids[0], "shard"] == "part0.parquet,part1.parquet"
    links = pd.DataFrame({"article_id": [*ids, ids[0]], "ticker": ["AAA"] * 5 + ["BBB"],
                          "date": pd.to_datetime(["2023-11-24"] * 6)})
    schedule = mcal.get_calendar("NASDAQ").schedule("2023-11-22", "2023-11-30")
    audit = align_candidate(links, articles, schedule)
    assert audit.article_id.tolist() == links.article_id.tolist()
    assert audit.quarantined.tolist() == [False, False, True, True, False, False]
    assert audit.next_session.iloc[0] == pd.Timestamp("2023-11-24")
    assert audit.next_session.iloc[1] == pd.Timestamp("2023-11-27")
    assert audit.extra_session.iloc[0] == pd.Timestamp("2023-11-27")
    assert audit.next_session.iloc[2:5].isna().all()
    assert not audit.timestamp_verified.any()
    official = pd.DataFrame({"date": pd.to_datetime(["2023-11-24", "2023-11-27"]),
                             "ticker": ["AAA", "AAA"], "n_articles": [5, 0]})
    daily = coverage_table(audit, official)
    assert daily.missing_source_count.tolist() == [1, 0]
    assert daily.quarantined_source_count.tolist() == [2, 0]
    assert daily.next_session_count.tolist() == [1, 1]


def test_feature_ids_cannot_be_reordered_and_multiticker_links_reuse_embedding():
    headlines = pd.DataFrame({"article_id": ["a", "b"]})
    features = pd.DataFrame(np.ones((2, len(EMB_COLUMNS))), columns=EMB_COLUMNS)
    features.insert(0, "confidence", 0.8)
    features.insert(0, "sentiment", 0.2)
    features.insert(0, "article_id", headlines.article_id)
    validate_features(features, headlines)
    with pytest.raises(ValueError, match="ordered"):
        validate_features(features.iloc[::-1], headlines)
    date = pd.Timestamp("2023-11-24")
    links = pd.DataFrame({"article_id": ["a", "a"], "ticker": ["AAA", "BBB"],
                          "next_session": [date] * 2, "extra_session": [pd.NaT] * 2})
    prices = links[["ticker"]].assign(date=date, in_index=True)
    coverage = prices[["date", "ticker"]].assign(next_session_count=1, extra_session_count=0, missing_source_count=0)
    result = aggregate_policies(features, links, prices, coverage)
    assert result["next_session"].n_articles.tolist() == [1, 1]
    assert result["extra_session"].n_articles.tolist() == [0, 0]
    np.testing.assert_array_equal(result["next_session"][EMB_COLUMNS], np.ones((2, len(EMB_COLUMNS))))
