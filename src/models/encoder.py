"""Per-company LSTM encoder, shared by the proposed model and Baseline 6. Owner: Modelling. Issue #35.

Baseline 6 must be exactly the proposed model minus the GAT, so both import this class
rather than defining their own LSTM.
"""

from __future__ import annotations

import torch
from torch import nn


class CompanyEncoder(nn.Module):
    """[N companies, T days, F features] price/indicator window (+ news features) -> [N, hidden]."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        # nn.LSTM's own dropout only acts between stacked layers, so apply it to the output too.
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.dropout(h_n[-1])  # last layer's state after the most recent day
