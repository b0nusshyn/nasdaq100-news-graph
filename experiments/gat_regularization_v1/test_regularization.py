from types import SimpleNamespace

import torch

from experiments.gat_regularization_v1.study import drop_pairs


def test_training_only_dropout_preserves_reverse_pairs_and_aligned_attributes():
    src = torch.arange(0, 100, 2)
    dst = src + 1
    edges = torch.stack((torch.cat((src, dst)), torch.cat((dst, src))))
    weights = torch.arange(50.).repeat(2)
    types = torch.zeros(100, dtype=torch.long)
    args = (None, edges, types, weights)
    assert drop_pairs(SimpleNamespace(training=False), args) is args
    torch.manual_seed(42)
    _, kept, t, w = drop_pairs(SimpleNamespace(training=True), args)
    assert 0 < kept.shape[1] < edges.shape[1]
    assert set(map(tuple, kept.T.tolist())) == set(map(tuple, kept.flip(0).T.tolist()))
    assert torch.equal(((kept.min(dim=0).values) // 2).float(), w)
    assert len(t) == len(w)
