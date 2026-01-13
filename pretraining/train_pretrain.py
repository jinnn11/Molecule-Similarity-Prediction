#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datasets import DatasetConfig, PrecomputedMultiModalDataset, multimodal_collate  # noqa: E402
from losses import clip_loss  # noqa: E402
from models import FingerprintMLP, GraphTower, ProjectionHead, build_resnet18  # noqa: E402


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain tri-tower encoders with contrastive loss.")
    parser.add_argument("--data-dir", default="pretraining_data/drugs", help="Precomputed data folder.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-dir", default="pretraining_runs")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Enable autocast mixed precision.")
    parser.add_argument("--no-channels-last", action="store_true", help="Disable channels_last for images.")
    parser.add_argument("--log-interval", type=int, default=50, help="Steps between progress logs.")
    parser.add_argument("--device", default=_default_device())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    cfg = DatasetConfig(
        root=args.data_dir,
        image_size=args.image_size,
        fingerprint_bits=2048,
        augment=not args.no_augment,
    )
    dataset = PrecomputedMultiModalDataset(cfg)
    print(f"Loaded dataset with {len(dataset)} molecules from {args.data_dir}")
    device = torch.device(args.device)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=multimodal_collate,
    )

    img_encoder = build_resnet18().to(device)
    graph_encoder = GraphTower(out_dim=128).to(device)
    fp_encoder = FingerprintMLP(in_dim=2048, hidden_dim=1024, out_dim=512).to(device)

    proj_2d = ProjectionHead(512, 256).to(device)
    proj_3d = ProjectionHead(128, 256).to(device)
    proj_1d = ProjectionHead(512, 256).to(device)

    params = list(img_encoder.parameters()) + list(graph_encoder.parameters())
    params += list(fp_encoder.parameters()) + list(proj_2d.parameters())
    params += list(proj_3d.parameters()) + list(proj_1d.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / args.accumulation_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, steps_per_epoch * args.warmup_epochs)
    min_lr = min(args.min_lr, args.lr)

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        lr = min_lr + (args.lr - min_lr) * cosine
        return lr / args.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)

    img_encoder.train()
    graph_encoder.train()
    fp_encoder.train()
    proj_2d.train()
    proj_3d.train()
    proj_1d.train()

    channels_last = not args.no_channels_last
    if channels_last:
        img_encoder = img_encoder.to(memory_format=torch.channels_last)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()
        start = time.time()
        for step, (images, graphs, fps) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            fps = fps.to(device, non_blocking=True)
            graphs = graphs.to(device)

            if args.amp:
                if device.type == "cuda":
                    autocast_dtype = torch.float16
                elif device.type == "mps":
                    autocast_dtype = torch.float16
                else:
                    autocast_dtype = torch.bfloat16
                with torch.autocast(device.type, dtype=autocast_dtype):
                    z2d = proj_2d(img_encoder(images))
                    z3d = proj_3d(graph_encoder(graphs.z, graphs.pos, graphs.batch))
                    z1d = proj_1d(fp_encoder(fps))

                    loss = clip_loss(z2d, z3d, args.temperature)
                    loss += clip_loss(z2d, z1d, args.temperature)
                    loss += clip_loss(z3d, z1d, args.temperature)
            else:
                z2d = proj_2d(img_encoder(images))
                z3d = proj_3d(graph_encoder(graphs.z, graphs.pos, graphs.batch))
                z1d = proj_1d(fp_encoder(fps))

                loss = clip_loss(z2d, z3d, args.temperature)
                loss += clip_loss(z2d, z1d, args.temperature)
                loss += clip_loss(z3d, z1d, args.temperature)

            loss = loss / args.accumulation_steps
            loss.backward()

            if step % args.accumulation_steps == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * args.accumulation_steps
            if args.log_interval and (step % args.log_interval == 0 or step == len(loader)):
                percent = 100.0 * step / max(len(loader), 1)
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch:03d} step {step:05d}/{len(loader)} "
                    f"({percent:5.1f}%) lr {current_lr:.6g}"
                )

        elapsed = time.time() - start
        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"epoch {epoch:03d} loss {avg_loss:.4f} time {elapsed:.1f}s")

    torch.save(img_encoder.state_dict(), os.path.join(args.save_dir, "resnet18.pt"))
    torch.save(graph_encoder.state_dict(), os.path.join(args.save_dir, "schnet.pt"))
    torch.save(fp_encoder.state_dict(), os.path.join(args.save_dir, "fingerprint_mlp.pt"))
    print(f"saved encoders to {args.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
