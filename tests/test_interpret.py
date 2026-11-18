"""Task 2.9 (#37): trace a prediction to its neighbours' articles. Light dependencies only (no torch)."""

import pandas as pd
import pytest

from src.nlp.interpret import (
    EVIDENCE_COLUMNS,
    format_trace,
    load_prediction,
    source_articles,
    top_neighbours,
    trace,
    type_order,
    window_days,
)

D = pd.Timestamp("2023-05-17")
CAL = pd.bdate_range("2023-04-01", "2023-05-31")
DAYS = window_days(CAL, D, 5)


def contribution(**over):
    row = {"horizon": 1, "fold": 4, "date": D, "ticker": "SGEN", "y_true": 1, "has_news": False,
           "p_up": 0.91, "p_up_no_graph": 0.49, "logit": 2.37, "logit_no_graph": -0.04,
           "graph_logit": 2.41, "graph_logit_within_day": 2.72,
           "graph_logit_sector": 0.17, "in_degree_sector": 2,
           "graph_logit_comention": 1.98, "in_degree_comention": 1,
           "graph_logit_corr": -0.20, "in_degree_corr": 2}
    return pd.DataFrame([{**row, **over}])


def edges():
    rows = [("VRTX", "comention", 0.003, 1.0), ("AMGN", "sector", 1.0, 0.6), ("GILD", "sector", 1.0, 0.4),
            ("PFE", "corr", 0.5, 0.3), ("BIIB", "corr", 0.4, 0.7)]
    return pd.DataFrame([{"horizon": 1, "fold": 4, "date": D, "src": s, "dst": "SGEN", "edge_type": t,
                          "weight": w, "alpha_mean": a} for s, t, w, a in rows])


def links():
    rows = [("a1", D, "VRTX"), ("a2", DAYS[0], "VRTX"), ("a3", DAYS[0] - pd.Timedelta(days=3), "VRTX"),
            ("a4", D, "AMGN"), ("a5", D, "SGEN"), ("a6", D, "ZZZZ")]
    return pd.DataFrame(rows, columns=["article_id", "date", "ticker"])


def test_window_is_n_trading_days_ending_on_the_date():
    assert len(DAYS) == 5 and DAYS[-1] == D and DAYS.is_monotonic_increasing
    with pytest.raises(ValueError, match="not a trading day"):
        window_days(CAL, "2023-05-20", 5)  # a Saturday


def test_load_prediction_selects_one_row_and_its_incoming_edges(tmp_path):
    other = contribution(ticker="AMGN")
    pd.concat([contribution(), other, contribution(horizon=5)]).to_parquet(tmp_path / "gat_lstm_contribution.parquet")
    e = edges()
    pd.concat([e, e.assign(dst="AMGN"), e.assign(horizon=5), e.assign(date=DAYS[0])]).to_parquet(
        tmp_path / "gat_lstm_edges.parquet")
    row, inc = load_prediction("2023-05-17", "SGEN", 1, tmp_path)
    assert row["ticker"] == "SGEN" and row["horizon"] == 1
    assert len(inc) == len(e) and (inc["dst"] == "SGEN").all() and (pd.to_datetime(inc["date"]) == D).all()
    with pytest.raises(ValueError, match="no 1d test prediction"):
        load_prediction("2023-05-16", "SGEN", 1, tmp_path)


def test_edge_types_ordered_by_contribution_and_neighbours_by_attention():
    row = contribution().iloc[0]
    order = type_order(row, ["sector", "comention", "corr"])
    assert order == ["comention", "corr", "sector"]  # |1.98| > |-0.20| > |0.17|
    nb = top_neighbours(edges(), order, top=1)
    assert nb["edge_type"].tolist() == order
    assert nb["src"].tolist() == ["VRTX", "BIIB", "AMGN"]  # highest attention within each type


def test_source_articles_only_inside_window():
    arts = source_articles(links(), "VRTX", DAYS)
    assert set(arts["article_id"]) == {"a1", "a2"}  # a3 is before the window
    assert arts["date"].is_monotonic_decreasing


def test_trace_joins_neighbours_to_their_articles():
    info = pd.DataFrame({"article_id": ["a1", "a2", "a4"], "sentiment": [0.8, -0.1, 0.0],
                         "title": ["Pfizer to buy Seagen", "weaker headline", "neutral"]})
    t = trace(contribution().iloc[0], edges(), links(), DAYS, info, top=2)
    ev = t["evidence"]
    assert ev.columns[:len(EVIDENCE_COLUMNS)].tolist() == EVIDENCE_COLUMNS
    assert set(ev["src"]) <= set(t["neighbours"]["src"])
    assert set(ev["article_id"]) == {"a1", "a2", "a4"}  # never the target's own or unrelated news
    assert ev.loc[ev["article_id"] == "a1", "edge_type"].item() == "comention"
    text = format_trace(t, max_articles=1)
    assert "SGEN on 2023-05-17" in text and "Pfizer to buy Seagen" in text and "weaker headline" not in text
    assert text.index("### comention") < text.index("### corr") < text.index("### sector")


def test_trace_with_no_news_anywhere_is_empty_not_an_error():
    t = trace(contribution().iloc[0], edges(), links().iloc[:0], DAYS)
    assert t["evidence"].empty and list(t["evidence"].columns) == EVIDENCE_COLUMNS
    assert "0 articles in window" in format_trace(t)
