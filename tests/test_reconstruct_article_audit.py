import pandas as pd
import pytest

from src.nlp.match_tickers import article_id
from src.nlp.reconstruct_article_audit import recover


@pytest.fixture
def inputs():
    url = "https://example.invalid/article"
    key = article_id(url)
    day = pd.Timestamp("2023-07-03")
    links = pd.DataFrame({"article_id": [key, key], "ticker": ["AAA", "BBB"], "date": [day, day]})
    review = pd.DataFrame({"Url": [url], "ticker": ["AAA"], "date": [day], "source": ["tag"],
                           "title": ["Headline"], "verdict": ["wrong: not mentioned"]})
    trace = pd.DataFrame({"article_id": [key, key], "src": ["AAA", "BBB"], "date": [day, day],
                          "title": ["Headline", "Headline"], "sentiment": [0.2, 0.2],
                          "confidence": [0.8, 0.8], "source_artifact": ["a.csv", "b.csv"]})
    return links, review, trace


def test_recovers_links_without_converting_aligned_date_to_publication_time(inputs):
    links, review, trace = inputs
    audit, sample = recover(links, review, [trace])
    pd.testing.assert_frame_equal(audit[links.columns], links)
    assert audit[["published_at", "feed_arrived_at"]].isna().all(axis=None)
    assert not audit.timestamp_verified.any()
    assert audit.matching_source.tolist() == ["tag", "upstream_mapping_method_unknown"]
    assert audit.matching_review_verdict.tolist() == ["wrong: not mentioned", "not_reviewed"]
    assert not audit.matching_verified.any()
    assert len(sample) == 1 and sample.source_artifact.iloc[0] == "a.csv;b.csv"
    assert sample.sentiment.iloc[0] == 0.2
    assert sample.model_revision.isna().all()


def test_rejects_source_ticker_or_date_not_in_mapping(inputs):
    links, review, trace = inputs
    trace.loc[0, "src"] = "CCC"
    with pytest.raises(ValueError, match="exact article ID"):
        recover(links, review, [trace])


def test_rejects_conflicting_recovered_values(inputs):
    links, review, trace = inputs
    trace.loc[0, "sentiment"] = 0.5
    with pytest.raises(ValueError, match="Conflicting"):
        recover(links, review, [trace])
