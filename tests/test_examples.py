"""Task 2.10 (#43): worked interpretability examples."""

import pandas as pd
import pytest

from src.nlp.examples import EXAMPLES

pytest.importorskip("matplotlib")


def test_every_example_is_a_candidate_from_3_10():
    from src.models.attention import results_dir

    top = pd.read_csv(results_dir() / "top_contributions.csv")
    keys = set(zip(top["date"], top["ticker"], top["horizon"], strict=True))
    for name, date, ticker, horizon in EXAMPLES:
        assert (date, ticker, horizon) in keys, name


def test_figure_draws_three_panels(tmp_path):
    from src.nlp.examples import figure
    from src.nlp.interpret import trace
    from tests.test_interpret import DAYS, contribution, edges, links

    info = pd.DataFrame({"article_id": ["a1", "a2", "a4"], "sentiment": [0.8, -0.1, 0.0]})
    t = trace(contribution().iloc[0], edges(), links(), DAYS, info)
    fig = figure(t, tmp_path / "x.png")
    assert len(fig.axes) == 3 and (tmp_path / "x.png").stat().st_size > 0
