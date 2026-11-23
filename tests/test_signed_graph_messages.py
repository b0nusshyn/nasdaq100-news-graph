"""Signed reliability and within-session centering on the same production layers."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.models.gat_lstm import GATLSTM  # noqa: E402


def test_signed_message_strength_and_isolates():
    model = GATLSTM(1, hidden=2, heads=2, edge_types=("corr",), signed_messages=True)
    hidden = torch.tensor([[1., -2.], [0., 0.], [0., 0.], [0., 0.]], requires_grad=True)
    edges = torch.tensor([[0, 0, 0], [1, 2, 3]])
    types = torch.zeros(3, dtype=torch.long)
    outputs, attention = model._messages(hidden, edges, types, torch.tensor([1., -.25, 0.]))
    messages = outputs[0]
    assert messages[1].norm() > 0
    torch.testing.assert_close(messages[2], -.25 * messages[1])
    torch.testing.assert_close(messages[[0, 3]], torch.zeros(2, 2))
    torch.testing.assert_close(attention, torch.tensor([[1., 1.], [1., 1.], [0., 0.]]))
    messages[1].square().sum().backward()
    assert hidden.grad[0].norm() > 0
    assert model.convs[0].lin.weight.grad.norm() > 0


def test_centering_is_session_local_and_changes_relative_logits():
    torch.manual_seed(14)
    centered = GATLSTM(2, hidden=4, heads=2, edge_types=("corr",), signed_messages=True,
                      center_graph_inputs=True).eval()
    plain = GATLSTM(2, hidden=4, heads=2, edge_types=("corr",), signed_messages=True).eval()
    plain.load_state_dict(centered.state_dict())
    x = torch.randn(6, 5, 2)
    days = torch.tensor([7, 7, 7, 2, 2, 2])
    edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])
    types = torch.zeros(6, dtype=torch.long)
    weights = torch.tensor([.2, .5, -.6, .7, .8, .4])
    args = edges, types, weights
    h = centered.encoder(x)
    h_centered = centered._graph_inputs(h, days)
    for values in (h_centered[:3], h_centered[3:]):
        torch.testing.assert_close(values.mean(0), torch.zeros(4), atol=1e-7, rtol=0)
    z, alpha = centered(x, *args, node_day=days)
    first_only, _ = centered(x[:3], edges[:, :3], types[:3], weights[:3], node_day=days[:3])
    torch.testing.assert_close(z[:3], first_only)
    torch.testing.assert_close(centered.explain(x, *args, node_day=days)["logits"], z)
    torch.testing.assert_close(alpha, torch.ones_like(alpha))
    delta = z - plain(x, *args)[0]
    assert (delta[:3] - delta[0]).abs().max() > 1e-6
    with pytest.raises(ValueError, match="one session ID"):
        centered(x, *args)


def test_zero_edge_cannot_dilute_nonzero_neighbors():
    torch.manual_seed(10)
    model = GATLSTM(2, hidden=3, heads=2, edge_types=("corr",), signed_messages=True).eval()
    h = torch.randn(4, 3)
    edges = torch.tensor([[0, 1, 2], [3, 3, 3]])
    types = torch.zeros(3, dtype=torch.long)
    weights = torch.tensor([.2, -.7, 0.])
    all_out, attention = model._messages(h, edges, types, weights)
    kept_out, _ = model._messages(h, edges[:, :2], types[:2], weights[:2])
    torch.testing.assert_close(all_out[0], kept_out[0])
    torch.testing.assert_close(attention.sum(dim=0), torch.ones(2))
    torch.testing.assert_close(attention[2], torch.zeros(2))
    all_out[0][3].square().sum().backward()
    assert model.convs[0].att_edge.grad.norm() > 0
