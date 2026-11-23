import torch
from torch.nn import functional as F

from experiments.gat_joint_v1.loss import joint, pairwise


def test_pairwise_compares_only_within_dates_and_handles_ties():
    z = torch.tensor([0., 0., 1000., -1000.], requires_grad=True)
    y = torch.tensor([1., 0., 1., 0.])
    d = torch.tensor([1, 1, 2, 3])
    loss = pairwise(z, y, d)
    torch.testing.assert_close(loss, F.softplus(torch.tensor(0.)))
    loss.backward()
    torch.testing.assert_close(z.grad, torch.tensor([-.5, .5, 0., 0.]))


def test_joint_ignores_unlabelled_nodes_and_retains_bce_on_single_class():
    z = torch.tensor([.1, -.2, 1e5], requires_grad=True)
    batch = {"target": torch.tensor([True, True, False]), "y": torch.ones(3), "node_day": torch.tensor([1, 1, 1])}
    loss = joint(z, batch, .3)
    torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(z[:2], torch.ones(2)))
    loss.backward()
    assert z.grad[2] == 0 and torch.isfinite(z.grad).all()
