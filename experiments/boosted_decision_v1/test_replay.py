import pandas as pd

from experiments.boosted_decision_v1.replay import confidence_result, ranked, ranking_select, threshold


def test_threshold_resolves_compressed_probability_without_using_point_five():
    y = pd.Series([0, 0, 1, 1])
    p = pd.Series([.46, .47, .48, .49])
    cut = threshold(y, p)
    assert .47 < cut <= .48
    assert (p >= cut).equals(y.astype(bool))


def test_date_ranking_uses_prior_prevalence_and_deterministic_ties():
    f = pd.DataFrame({"date": [0, 0, 0, 1, 1, 1], "ticker": ["C", "B", "A", "C", "A", "B"],
                      "score": [3, 3, 3, 17, 1, 6], "y_true": [0, 1, 1, 1, 0, 0]})
    r = ranked(f, .3)
    assert r.loc[r.rank_call, "ticker"].tolist() == ["A", "C"]
    flipped = ranked(f.assign(y_true=1-f.y_true), .3)
    pd.testing.assert_series_equal(r.rank_call, flipped.rank_call)
    selected = ranking_select(r, 1/3)
    assert selected.groupby("date").size().tolist() == [1, 1]
    report = confidence_result(r, selected, "rank_call")
    assert report["rows"] == report["dates"] == 2
    assert report["coverage"] == 1/3
