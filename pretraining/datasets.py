from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from torch_geometric.data import Batch, Data
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "torch_geometric is required for 3D graph data."
    ) from exc


@dataclass
class DatasetConfig:
    root: str
    image_size: int = 224
    fingerprint_bits: int = 2048
    augment: bool = True


def _random_rotation_matrix() -> torch.Tensor:
    # Uniform random rotation matrix using axis-angle.
    axis = torch.randn(3)
    axis = axis / (axis.norm() + 1e-12)
    angle = random.random() * 2.0 * math.pi
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return torch.tensor(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=torch.float32,
    )


def _rotate_positions(pos: torch.Tensor) -> torch.Tensor:
    pos_centered = pos - pos.mean(dim=0, keepdim=True)
    rot = _random_rotation_matrix().to(pos_centered.device)
    return pos_centered @ rot.T


def _make_image_transform(image_size: int, augment: bool) -> transforms.Compose:
    ops = []
    if augment:
        ops.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomRotation(30),
            ]
        )
    else:
        ops.append(transforms.Resize((image_size, image_size)))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(ops)


class PrecomputedMultiModalDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self.cfg = cfg
        root = Path(cfg.root)

        self.images_dir = root / "images"
        self.fingerprints = np.load(root / "fingerprints.npy", mmap_mode="r")
        self.z = np.load(root / "z.npy", mmap_mode="r")
        self.pos = np.load(root / "pos.npy", mmap_mode="r")
        self.offsets = np.load(root / "offsets.npy", mmap_mode="r")
        self.lengths = np.load(root / "lengths.npy", mmap_mode="r")

        fp_dim = self.fingerprints.shape[1]
        self.fp_packed = fp_dim * 8 == cfg.fingerprint_bits

        self.transform = _make_image_transform(cfg.image_size, cfg.augment)

        if len(self.offsets) != len(self.fingerprints):
            raise ValueError("Fingerprint and graph counts do not match.")

    def __len__(self) -> int:
        return len(self.fingerprints)

    def _load_image(self, idx: int) -> torch.Tensor:
        image_path = self.images_dir / f"{idx:06d}.png"
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def _load_fingerprint(self, idx: int) -> torch.Tensor:
        fp_row = self.fingerprints[idx]
        if self.fp_packed:
            unpacked = np.unpackbits(fp_row)[: self.cfg.fingerprint_bits]
            fp = unpacked.astype(np.float32)
        else:
            fp = fp_row.astype(np.float32)
        return torch.from_numpy(fp)

    def _load_graph(self, idx: int) -> Data:
        start = int(self.offsets[idx])
        length = int(self.lengths[idx])
        z = torch.from_numpy(self.z[start : start + length].astype(np.int64))
        pos = torch.from_numpy(self.pos[start : start + length].astype(np.float32))
        if self.cfg.augment:
            pos = _rotate_positions(pos)
        return Data(z=z, pos=pos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Data, torch.Tensor]:
        image = self._load_image(idx)
        graph = self._load_graph(idx)
        fingerprint = self._load_fingerprint(idx)
        return image, graph, fingerprint


def multimodal_collate(batch):
    images, graphs, fingerprints = zip(*batch)
    images = torch.stack(images, dim=0)
    fingerprints = torch.stack(fingerprints, dim=0)
    graphs = Batch.from_data_list(list(graphs))
    return images, graphs, fingerprints
