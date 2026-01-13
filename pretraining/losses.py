from __future__ import annotations

import torch
from torch import nn


def _normalize(z: torch.Tensor) -> torch.Tensor:
    return z / (z.norm(dim=1, keepdim=True) + 1e-12)


def clip_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z_a = _normalize(z_a)
    z_b = _normalize(z_b)
    logits = (z_a @ z_b.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_a = nn.functional.cross_entropy(logits, targets)
    loss_b = nn.functional.cross_entropy(logits.T, targets)
    return 0.5 * (loss_a + loss_b)
