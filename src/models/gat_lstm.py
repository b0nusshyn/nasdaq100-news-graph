"""Proposed model: CompanyEncoder -> GAT over company edges -> dense -> P(up). Owner: Modelling. Issues #9, #35.

With no edge types the GAT branch does not exist and the model is `dense(encoder(x))`: Baseline 6
(LSTM + news, no graph) is this class with `edge_types=()`, trained by the same
`src.models.train.train_fold`, so the proposed model and Baseline 6 differ by the GAT alone.
The encoder is built first, so under one seed both start from identical encoder weights.

Attention weights are returned with every forward pass so predictions can be traced to the
article and edge type that produced them (#42, #37).
"""

from __future__ import annotations

from functools import cache

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoder import CompanyEncoder


@cache
def signed_message_gat() -> type:
    """GATConv subclass that applies a bounded signed reliability multiplier after attention
    normalization. Built on first use so importing this module does not need torch_geometric
    (CLAUDE.md: heavy packages are imported inside the functions that need them)."""
    from torch_geometric.nn import GATConv

    class SignedMessageGAT(GATConv):
        def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, dim_size):
            alpha = super().edge_update(alpha_j, alpha_i, edge_attr, index, ptr, dim_size)
            return alpha * edge_attr

    return SignedMessageGAT


class GATLSTM(nn.Module):
    """One GAT per edge type, outputs summed: each type learns its own message transform, so a
    type can learn to pass a signal on with the opposite sign (README "Signed propagation").
    The dense layer sees the company's own state next to the graph signal, `[h ; g]`, so the
    model contains Baseline 6 (`dense(h)`, built when `edge_types` is empty) and any gap over it
    is the graph's contribution."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        heads: int = 4,
        edge_types: tuple[str, ...] = (),
        *,
        layers: int = 1,
        dropout: float = 0.1,
        signed_messages: bool = False,
        center_graph_inputs: bool = False,
    ):
        from torch_geometric.nn import GATConv

        super().__init__()
        self.edge_types = tuple(edge_types)
        self.heads = heads
        self.signed_messages = signed_messages
        self.center_graph_inputs = center_graph_inputs
        self.encoder = CompanyEncoder(n_features, hidden, layers=layers, dropout=dropout)
        # No self-loops: a company's own state reaches the dense layer through the skip link.
        # Edge weight (co-mention count, correlation) is an edge feature of the attention score.
        conv_class = signed_message_gat() if signed_messages else GATConv
        self.convs = nn.ModuleList(
            conv_class(hidden, hidden, heads=heads, concat=False, edge_dim=1, add_self_loops=False,
                       bias=not signed_messages)
            for _ in self.edge_types
        )
        self.head = nn.Linear(2 * hidden if self.edge_types else hidden, 1)

    @property
    def has_graph(self) -> bool:
        return bool(self.edge_types)

    def _check(self, edge_type: torch.Tensor) -> None:
        if ((edge_type < 0) | (edge_type >= len(self.edge_types))).any():
            raise ValueError(f"edge type index outside this model's edge types {self.edge_types}")

    def _messages(self, h, edge_index, edge_type, edge_weight) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Each edge type's GAT output [N, hidden], and attention [E, heads] in input edge order."""
        attention = h.new_zeros(edge_index.shape[1], self.heads)
        outs = []
        for t, conv in enumerate(self.convs):
            mask = edge_type == t
            if self.signed_messages:
                mask = mask & (edge_weight != 0)
            weight = edge_weight[mask].unsqueeze(-1)
            out, (_, alpha) = conv(h, edge_index[:, mask], edge_attr=weight,
                                   return_attention_weights=True)
            outs.append(out)
            attention[mask] = alpha / weight if self.signed_messages else alpha
        return outs, attention

    def _graph_inputs(self, h: torch.Tensor, node_day: torch.Tensor | None) -> torch.Tensor:
        if not self.center_graph_inputs:
            return h
        if node_day is None or node_day.shape != (len(h),):
            raise ValueError("centered graph inputs require one session ID per node")
        _, group, counts = torch.unique(node_day, return_inverse=True, return_counts=True)
        totals = h.new_zeros(len(counts), h.shape[1]).index_add(0, group, h)
        return h - (totals / counts[:, None])[group]

    def _logits(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([h, F.elu(g)], dim=-1)).squeeze(-1)

    @staticmethod
    def _sum(h: torch.Tensor, outs: list[torch.Tensor]) -> torch.Tensor:
        g = torch.zeros_like(h)
        for out in outs:
            g = g + out
        return g

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor,
        *, node_day: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x [N, T, F]; edge_index [2, E] (src, dst); edge_type [E] indexes `edge_types`;
        edge_weight [E]. Return (logits [N], attention [E, heads] in input edge order)."""
        self._check(edge_type)
        h = self.encoder(x)
        if not self.has_graph:
            return self.head(h).squeeze(-1), h.new_zeros(edge_index.shape[1], self.heads)
        outs, attention = self._messages(self._graph_inputs(h, node_day), edge_index, edge_type, edge_weight)
        return self._logits(h, self._sum(h, outs)), attention

    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor,
        *, node_day: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """The forward pass plus how much the graph moved each logit (#42).

        Attention is a softmax over a node's valid incoming edges of one type, so it sums to one whether
        or not the graph changed the prediction: it says *which* neighbour, not *how much*.
        Signed messages further multiply these normalized coefficients by the edge weights.
        - `logits`: identical to `forward`.
        - `logits_no_graph`: the same model with the graph signal `g` set to zero, i.e. the dense
          layer sees `[h ; 0]`. This is the model with its GAT branch removed, not Baseline 6
          (which has its own trained head).
        - `drop_<type>`: the logit with that edge type's message removed from `g` and the others
          kept. ELU makes the types non-additive, so these need not sum to the overall change.
        """
        self._check(edge_type)
        if not self.has_graph:
            raise ValueError("explain needs a model with a graph")
        h = self.encoder(x)
        outs, attention = self._messages(self._graph_inputs(h, node_day), edge_index, edge_type, edge_weight)
        g = self._sum(h, outs)
        res = {"logits": self._logits(h, g), "logits_no_graph": self._logits(h, torch.zeros_like(h)),
               "attention": attention}
        for t, name in enumerate(self.edge_types):
            res[f"drop_{name}"] = self._logits(h, self._sum(h, outs[:t] + outs[t + 1:]))
        return res
