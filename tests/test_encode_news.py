import numpy as np
import pandas as pd
import pytest

from src.nlp.encode_news import EMB_COLUMNS, aggregate, encode, encode_articles, headlines
from src.nlp.match_tickers import article_id


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny random BERT classifier with FinBERT's label layout; no download, CPU only."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "apple", "beats", "misses", "stocks", "fall"]
    vocab_file = tmp_path_factory.mktemp("tok") / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
    config = BertConfig(vocab_size=len(vocab), hidden_size=768, num_hidden_layers=1, num_attention_heads=12,
                        intermediate_size=32, num_labels=3,
                        id2label={0: "positive", 1: "negative", 2: "neutral"},
                        label2id={"positive": 0, "negative": 1, "neutral": 2})
    torch.manual_seed(0)
    return tokenizer, BertForSequenceClassification(config).eval()


def test_encode_shapes_ranges_and_order(tiny):
    tokenizer, model = tiny
    texts = ["apple beats", "stocks fall", "apple misses", ""]
    out = encode(texts, batch_size=3, device="cpu", tokenizer=tokenizer, model=model)
    assert list(out.columns) == ["sentiment", "confidence", *EMB_COLUMNS]
    assert len(out) == len(texts)
    assert (out.dtypes == np.float32).all()
    assert out["sentiment"].between(-1, 1).all()
    assert out["confidence"].between(1 / 3, 1).all()
    # Batching must not change results or order.
    one = encode(texts, batch_size=1, device="cpu", tokenizer=tokenizer, model=model)
    np.testing.assert_allclose(out.to_numpy(), one.to_numpy(), atol=1e-5)


def test_sentiment_is_p_pos_minus_p_neg(tiny):
    tokenizer, model = tiny
    import torch

    out = encode(["apple beats"], device="cpu", tokenizer=tokenizer, model=model)
    with torch.no_grad():
        probs = model(**tokenizer(["apple beats"], return_tensors="pt")).logits.softmax(-1)[0]
    assert out.loc[0, "sentiment"] == pytest.approx(float(probs[0] - probs[1]), abs=1e-5)
    assert out.loc[0, "confidence"] == pytest.approx(float(probs.max()), abs=1e-5)


def test_headlines_one_per_article_and_filtered():
    filtered = pd.DataFrame({"Url": ["u1", "u1", "u2", "u3"],
                             "Article_title": ["Apple beats", "Apple beats", "Stocks fall", "Old news"]})
    arts = headlines(filtered, pd.Series([article_id("u1"), article_id("u2")]))
    assert arts["article_id"].tolist() == [article_id("u1"), article_id("u2")]


def test_encode_articles_keeps_ids(tiny):
    tokenizer, model = tiny
    arts = pd.DataFrame({"article_id": ["a", "b"], "Article_title": ["apple beats", None]})
    out = encode_articles(arts, device="cpu", tokenizer=tokenizer, model=model)
    assert out["article_id"].tolist() == ["a", "b"]
    assert len(out.columns) == 3 + len(EMB_COLUMNS)


# --- 2.6: aggregation to news_features ------------------------------------------------------


def _feats(rows):
    """rows: article_id -> (sentiment, confidence, emb fill value)."""
    data = {"article_id": list(rows)}
    data["sentiment"] = np.array([v[0] for v in rows.values()], dtype=np.float32)
    data["confidence"] = np.array([v[1] for v in rows.values()], dtype=np.float32)
    emb = np.array([[v[2]] * len(EMB_COLUMNS) for v in rows.values()], dtype=np.float32)
    return pd.concat([pd.DataFrame(data), pd.DataFrame(emb, columns=EMB_COLUMNS)], axis=1)


def _prices():
    d = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03", "2018-06-01"])
    return pd.DataFrame({"date": d, "ticker": ["AAA", "BBB", "AAA", "BBB", "AAA"],
                         "in_index": [True, True, True, False, True]})


def test_aggregate_mean_counts_and_zero_rows():
    from src.core.schema import conform

    feats = _feats({"a1": (0.8, 0.9, 1.0), "a2": (-0.4, 0.7, 3.0), "a3": (0.5, 0.6, 5.0)})
    links = pd.DataFrame({"article_id": ["a1", "a2", "a3"],
                          "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]),
                          "ticker": ["AAA", "AAA", "BBB"]})
    out = aggregate(feats, links, _prices(), "2019-01-01", "2023-12-31")
    conform(out, "news_features")
    # Rows = in-index keys inside the window: no 2018 row, no BBB 01-03 (not in index).
    assert list(zip(out["date"].dt.strftime("%m-%d"), out["ticker"], strict=True)) == [
        ("01-02", "AAA"), ("01-02", "BBB"), ("01-03", "AAA")]
    aaa = out.iloc[0]
    assert aaa["has_news"] and aaa["n_articles"] == 2
    assert aaa["sentiment"] == pytest.approx(0.2) and aaa["confidence"] == pytest.approx(0.8)
    assert aaa["emb_0"] == pytest.approx(2.0) and aaa["emb_767"] == pytest.approx(2.0)


def test_no_news_is_distinct_from_neutral():
    feats = _feats({"a1": (0.0, 0.95, 1.0)})  # neutral news
    links = pd.DataFrame({"article_id": ["a1"], "date": pd.to_datetime(["2020-01-02"]), "ticker": ["AAA"]})
    out = aggregate(feats, links, _prices(), "2019-01-01", "2023-12-31").set_index("ticker")
    day = out[out["date"] == "2020-01-02"]
    assert day.loc["AAA", "has_news"] and day.loc["AAA", "sentiment"] == 0
    assert not day.loc["BBB", "has_news"] and day.loc["BBB", "n_articles"] == 0
    assert (day.loc["BBB", ["sentiment", "confidence", *EMB_COLUMNS]] == 0).all()


def test_aggregate_refuses_missing_features():
    links = pd.DataFrame({"article_id": ["a1", "zz"], "date": pd.to_datetime(["2020-01-02"] * 2),
                          "ticker": ["AAA", "AAA"]})
    with pytest.raises(ValueError, match="no features"):
        aggregate(_feats({"a1": (0.1, 0.5, 0.0)}), links, _prices())
