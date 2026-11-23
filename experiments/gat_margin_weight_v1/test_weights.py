from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.gat_margin_weight_v1.study import fit_weights, weighted


def test_fit_only_weights_ignore_other_labels_and_fail_outside_fit():
    dates = pd.date_range("2022-01-03", periods=2)
    t = pd.Index(["A", "B"])
    f = pd.MultiIndex.from_product([dates, t], names=["date", "ticker"]).to_frame(index=False)
    f["fwd_ret_1d"] = [.05, -.004, 1, -2]
    panel = SimpleNamespace(tickers=t, in_index=np.ones((2, 2)), day_index=dates.get_indexer,
                            rows=lambda d, h: f[f.date.isin(d)])
    w = fit_weights(panel, f, dates[:1], 1)
    np.testing.assert_array_equal(w[0], [1.25, 1.])
    assert np.isnan(w[1]).all()
    z = torch.zeros(1, requires_grad=True)
    batch = {"target": torch.tensor([True]), "y": torch.tensor([1.]), "at": np.array([[0, 0]])}
    weighted(z, batch, w).backward()
    torch.testing.assert_close(z.grad, torch.tensor([-.5]))
    with pytest.raises(ValueError, match="outside"):
        weighted(z, dict(batch, at=np.array([[1, 0]])), w)
