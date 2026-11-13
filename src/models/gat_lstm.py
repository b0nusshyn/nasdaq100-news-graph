"""Proposed model: CompanyEncoder -> GAT over company edges -> dense -> P(up). Owner: Modelling. Issue #9.

Attention weights are returned with every forward pass so predictions can be traced to the
article and edge type that produced them (#42, #37).
"""

from __future__ import annotations

import torch
from torch import nn

from src.core.todo import todo


class GATLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, heads: int = 4, edge_types: tuple[str, ...] = ()):
        super().__init__()
        todo(9, "GATLSTM")

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits [N], attention [E, heads])."""
        todo(9, "GATLSTM.forward")
