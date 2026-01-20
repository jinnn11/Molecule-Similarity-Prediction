from __future__ import annotations

import torch
import inspect

from torch import nn
from torchvision import models

try:
    from torch_geometric.nn.models import SchNet
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "torch_geometric is required for the SchNet 3D encoder."
    ) from exc


def build_resnet18() -> nn.Module:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
    except AttributeError:
        backbone = models.resnet18(pretrained=True)
    backbone.fc = nn.Identity()
    return backbone


class FingerprintMLP(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 1024, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphTower(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        sig = inspect.signature(SchNet.__init__)
        if "out_channels" in sig.parameters:
            self.model = SchNet(
                hidden_channels=128,
                num_filters=128,
                num_interactions=6,
                num_gaussians=50,
                cutoff=10.0,
                out_channels=out_dim,
            )
        else:
            self.model = SchNet(
                hidden_channels=128,
                num_filters=128,
                num_interactions=6,
                num_gaussians=50,
                cutoff=10.0,
            )
            if hasattr(self.model, "lin2") and isinstance(self.model.lin2, nn.Linear):
                self.model.lin2 = nn.Linear(self.model.lin2.in_features, out_dim)
        self.proj = nn.Identity()

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(self.model(z, pos, batch))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
