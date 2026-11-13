"""Per-company LSTM encoder, shared by the proposed model and Baseline 6. Owner: Modelling. Issue #35.

Baseline 6 must be exactly the proposed model minus the GAT, so both import this class
rather than defining their own LSTM.
"""

from __future__ import annotations

import torch
from torch import nn

from src.core.todo import todo


class CompanyEncoder(nn.Module):
    """[N companies, T days, F features] price/indicator window (+ news features) -> [N, hidden]."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        todo(9, "CompanyEncoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        todo(9, "CompanyEncoder.forward")
