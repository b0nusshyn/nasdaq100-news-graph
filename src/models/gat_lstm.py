"""Proposed model: CompanyEncoder -> GAT over company edges -> dense -> P(up). Owner: Modelling. Issue #9.

Attention weights are returned with every forward pass so predictions can be traced to the
article and edge type that produced them (#42, #37).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoder import CompanyEncoder


class GATLSTM(nn.Module):
    """One GAT per edge type, outputs summed: each type learns its own message transform, so a
    type can learn to pass a signal on with the opposite sign (README "Signed propagation").
    The dense layer sees the company's own state next to the graph signal, `[h ; g]`, so the
    model contains Baseline 6 (`dense(h)`) and any gap over it is the graph's contribution."""

    def __init__(self, n_features: int, hidden: int = 64, heads: int = 4, edge_types: tuple[str, ...] = ()):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.edge_types = tuple(edge_types)
        self.heads = heads
        self.encoder = CompanyEncoder(n_features, hidden)
        # No self-loops: a company's own state reaches the dense layer through the skip link.
        # Edge weight (co-mention count, correlation) is an edge feature of the attention score.
        self.convs = nn.ModuleList(
            GATConv(hidden, hidden, heads=heads, concat=False, edge_dim=1, add_self_loops=False)
            for _ in self.edge_types
        )
        self.head = nn.Linear(2 * hidden, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x [N, T, F]; edge_index [2, E] (src, dst); edge_type [E] indexes `edge_types`;
        edge_weight [E]. Return (logits [N], attention [E, heads] in input edge order)."""
        if ((edge_type < 0) | (edge_type >= len(self.edge_types))).any():
            raise ValueError(f"edge type index outside this model's edge types {self.edge_types}")
        h = self.encoder(x)
        g = torch.zeros_like(h)
        attention = h.new_zeros(edge_index.shape[1], self.heads)
        for t, conv in enumerate(self.convs):
            mask = edge_type == t
            out, (_, alpha) = conv(h, edge_index[:, mask], edge_attr=edge_weight[mask].unsqueeze(-1),
                                   return_attention_weights=True)
            g = g + out
            attention[mask] = alpha
        logits = self.head(torch.cat([h, F.elu(g)], dim=-1)).squeeze(-1)
        return logits, attention
