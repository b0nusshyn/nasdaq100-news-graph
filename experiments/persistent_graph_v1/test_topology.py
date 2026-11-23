import pandas as pd

from experiments.persistent_graph_v1.study import topology


def graph(pairs):
    return pd.DataFrame([(s, d, w) for a, b, w in pairs for s, d in [(a, b), (b, a)]],
                        columns=["src", "dst", "weight"])


def test_screening_uses_both_halves_and_retains_original_weights():
    full = graph([("A", "B", .8), ("A", "C", .5), ("C", "D", .4)])
    first = graph([("A", "B", .7), ("A", "C", .2), ("C", "D", .1)])
    second = graph([("A", "B", -.2), ("A", "C", .3)])
    kept, sparse = topology(full, first, second)
    pd.testing.assert_frame_equal(kept, graph([("A", "C", .5)]))
    pd.testing.assert_frame_equal(sparse, graph([("A", "B", .8)]))
