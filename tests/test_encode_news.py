import numpy as np
import pandas as pd
import pytest

from src.nlp.encode_news import EMB_COLUMNS, encode, encode_articles, headlines
from src.nlp.match_tickers import article_id

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny random BERT classifier with FinBERT's label layout; no download, CPU only."""
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
