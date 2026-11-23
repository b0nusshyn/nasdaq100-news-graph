"""Training-only BCE plus an equally date-weighted cross-sectional pair objective."""
import torch
from torch.nn import functional as F


def pairwise(logits, labels, days):
    terms = []
    for day in days.unique():
        z, y = logits[days == day], labels[days == day]
        pos, neg = z[y == 1], z[y == 0]
        if len(pos) and len(neg):
            terms.append(F.softplus(neg[:, None] - pos[None, :]).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def joint(logits, batch, weight):
    use = batch["target"]
    z, y = logits[use], batch["y"][use]
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="sum") / use.sum().clamp_min(1)
    return bce + weight * pairwise(z, y, batch["node_day"][use]) if weight else bce
