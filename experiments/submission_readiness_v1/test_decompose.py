import numpy as np
import pandas as pd

from experiments.submission_readiness_v1.decompose import center, components, scores


def test_fit_only_targets_and_common_market_score():
    f = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]*3 + ["2024-01-03"]*3 + ["2024-01-04"]*3),
        "market": [1]*3+[2]*3+[3]*3, "stock": [1, 2, 4, 4, 6, 3, 6, 2, 3],
        "y": [0, 0, 1, 1, 1, 0, 1, 1, 0],
    })
    a, b = f.iloc[:6], f.iloc[6:]
    s, m = components(a, ["stock"], ["market"])
    raw, full = scores(b, s, m, ["stock"], ["market"])
    np.testing.assert_allclose(full - raw, (full - raw)[0])
    np.testing.assert_array_equal(raw.argsort(), full.argsort())
    for x, y in zip(scores(b.assign(y=1-b.y), s, m, ["stock"], ["market"]), (raw, full), strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_allclose(center(f, ["stock"]).groupby(f.date).sum(), 0, atol=1e-12)
