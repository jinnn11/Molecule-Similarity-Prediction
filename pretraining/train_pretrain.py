#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional
    plt = None
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (SCRIPT_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets import DatasetConfig, PrecomputedMultiModalDataset, multimodal_collate  # noqa: E402
from losses import clip_loss  # noqa: E402
from models import FingerprintMLP, GraphTower, ProjectionHead, build_resnet18  # noqa: E402
from utils.chem import render_image, fingerprint, load_pdb, embed_3d, get_pdb_path  # noqa: E402
from utils.metrics import pearson  # noqa: E402

try:
    from rdkit import Chem
except ImportError:
    Chem = None


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
    parser.add_argument(
        "--graph-device",
        default=None,
        help="Device for 3D graph tower (default: cpu on mps, else same as --device)",
    )
    parser.add_argument(
        "--eval-csv",
        action="append",
        help="CSV file(s) with molecule pairs for eval (repeatable).",
    )
    parser.add_argument("--eval-every", type=int, default=5, help="Epoch interval for eval.")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-no-pdb", action="store_true", help="Do not use PDB; use RDKit 3D.")
    return parser.parse_args()


def _load_eval_pairs(csv_paths: list[Path]) -> list[dict]:
    import csv as _csv

    rows = []
    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = _csv.DictReader(handle)
            for row in reader:
                row["__csv_path"] = str(csv_path)
                rows.append(row)
    return rows


class _EvalDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], image_size: int, no_pdb: bool):
        self.rows = rows
        self.image_size = image_size
        self.no_pdb = no_pdb
        self.image_cache = {}
        self.fp_cache = {}
        self.graph_cache = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _get_image(self, smiles: str) -> torch.Tensor:
        cached = self.image_cache.get(smiles)
        if cached is not None:
            return cached
        mol = Chem.MolFromSmiles(smiles)
        img = render_image(mol, self.image_size)
        self.image_cache[smiles] = img
        return img

    def _get_fp(self, smiles: str) -> torch.Tensor:
        cached = self.fp_cache.get(smiles)
        if cached is not None:
            return cached
        mol = Chem.MolFromSmiles(smiles)
        fp = fingerprint(mol)
        self.fp_cache[smiles] = fp
        return fp

    def _get_graph(self, row: dict, side: str):
        csv_path = Path(row["__csv_path"])
        id_pair = int(row.get("id_pair", "0"))
        smiles = row[f"curated_smiles_molecule_{side}"]
        key = None
        if not self.no_pdb:
            pdb_path = get_pdb_path(csv_path, id_pair, side)
            key = str(pdb_path)
            if key in self.graph_cache:
                return self.graph_cache[key]
            graph = load_pdb(pdb_path)
            if graph is None:
                graph = embed_3d(smiles)
            if graph is None:
                raise ValueError("missing 3d")
            self.graph_cache[key] = graph
            return graph
        key = smiles
        if key in self.graph_cache:
            return self.graph_cache[key]
        graph = embed_3d(smiles)
        if graph is None:
            raise ValueError("missing 3d")
        self.graph_cache[key] = graph
        return graph

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        smiles_a = row["curated_smiles_molecule_a"]
        smiles_b = row["curated_smiles_molecule_b"]
        img_a = self._get_image(smiles_a)
        img_b = self._get_image(smiles_b)
        fp_a = self._get_fp(smiles_a)
        fp_b = self._get_fp(smiles_b)
        graph_a = self._get_graph(row, "a")
        graph_b = self._get_graph(row, "b")
        label = float(row["frac_similar"])
        tan = float(row.get("TanimotoCombo", "nan")) if "TanimotoCombo" in row else float("nan")
        return img_a, graph_a, fp_a, img_b, graph_b, fp_b, label, tan


def _eval_collate(batch):
    from torch_geometric.data import Batch, Data

    imgs_a, graphs_a, fps_a, imgs_b, graphs_b, fps_b, labels, tans = zip(*batch)
    return (
        torch.stack(imgs_a, dim=0),
        Batch.from_data_list([g if isinstance(g, Data) else Data(z=g["z"], pos=g["pos"]) for g in graphs_a]),
        torch.stack(fps_a, dim=0),
        torch.stack(imgs_b, dim=0),
        Batch.from_data_list([g if isinstance(g, Data) else Data(z=g["z"], pos=g["pos"]) for g in graphs_b]),
        torch.stack(fps_b, dim=0),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(tans, dtype=torch.float32),
    )


def _prepare_eval_loader(csv_paths, image_size: int, no_pdb: bool, batch_size: int):
    rows = _load_eval_pairs(csv_paths)
    dataset = _EvalDataset(rows, image_size=image_size, no_pdb=no_pdb)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_eval_collate,
    )
    return loader, len(dataset)


def _run_eval(
    img_encoder,
    graph_encoder,
    fp_encoder,
    eval_loader,
    device: torch.device,
    graph_device: torch.device,
):
    img_encoder.eval()
    graph_encoder.eval()
    fp_encoder.eval()
    cos2d = []
    cos3d = []
    cos1d = []
    labels = []
    tani = []
    with torch.no_grad():
        for batch in eval_loader:
            img_a, graph_a, fp_a, img_b, graph_b, fp_b, label, tan = batch
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            fp_a = fp_a.to(device)
            fp_b = fp_b.to(device)
            graph_a = graph_a.to(graph_device)
            graph_b = graph_b.to(graph_device)

            emb_a2d = img_encoder(img_a)
            emb_b2d = img_encoder(img_b)
            emb_a1d = fp_encoder(fp_a)
            emb_b1d = fp_encoder(fp_b)
            emb_a3d = graph_encoder(graph_a.z, graph_a.pos, graph_a.batch)
            emb_b3d = graph_encoder(graph_b.z, graph_b.pos, graph_b.batch)

            cos2d.append(F.cosine_similarity(emb_a2d, emb_b2d).detach().cpu().numpy())
            cos1d.append(F.cosine_similarity(emb_a1d, emb_b1d).detach().cpu().numpy())
            if graph_device != device:
                emb_a3d = emb_a3d.to(device)
                emb_b3d = emb_b3d.to(device)
            cos3d.append(F.cosine_similarity(emb_a3d, emb_b3d).detach().cpu().numpy())
            labels.append(label.numpy())
            tani.append(tan.numpy())

    cos2d = np.concatenate(cos2d)
    cos3d = np.concatenate(cos3d)
    cos1d = np.concatenate(cos1d)
    labels = np.concatenate(labels)
    tani = np.concatenate(tani)
    avg_cos = (cos2d + cos3d + cos1d) / 3.0
    metrics = {
        "cos2d_pearson": pearson(cos2d, labels),
        "cos3d_pearson": pearson(cos3d, labels),
        "cos1d_pearson": pearson(cos1d, labels),
        "avg_cos_pearson": pearson(avg_cos, labels),
    }
    if np.isfinite(tani).any():
        metrics["tanimoto_pearson"] = pearson(tani[np.isfinite(tani)], labels[np.isfinite(tani)])
    return metrics


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    if args.save_dir == "pretraining_runs":
        run_dir = repo_root / "outputs" / "pretrain_runs" / run_id
    else:
        run_dir = Path(args.save_dir)
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = str(run_dir)

    log_path = run_dir / "train.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_handle.write(msg + "\n")
        log_handle.flush()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    eval_loader = None
    eval_size = 0
    if args.eval_csv:
        if Chem is None:
            raise ImportError("RDKit is required for --eval-csv.")
        eval_loader, eval_size = _prepare_eval_loader(
            [Path(p) for p in args.eval_csv],
            image_size=args.image_size,
            no_pdb=args.eval_no_pdb,
            batch_size=args.eval_batch_size,
        )
        log(f"Eval dataset: {eval_size} pairs")

    cfg = DatasetConfig(
        root=args.data_dir,
        image_size=args.image_size,
        fingerprint_bits=2048,
        augment=not args.no_augment,
    )
    dataset = PrecomputedMultiModalDataset(cfg)
    log(f"Loaded dataset with {len(dataset)} molecules from {args.data_dir}")
    device = torch.device(args.device)
    if args.graph_device:
        graph_device = torch.device(args.graph_device)
    else:
        graph_device = torch.device("cpu") if device.type == "mps" else device
    if graph_device != device:
        log(f"Using graph device {graph_device} (main device {device})")
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
    graph_encoder = GraphTower(out_dim=128).to(graph_device)
    fp_encoder = FingerprintMLP(in_dim=2048, hidden_dim=1024, out_dim=512).to(device)

    proj_2d = ProjectionHead(512, 256).to(device)
    proj_3d = ProjectionHead(128, 256).to(graph_device)
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

    run_info = {
        "args": vars(args),
        "devices": {
            "main": str(device),
            "graph": str(graph_device),
        },
        "dataset_size": len(dataset),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2)

    img_encoder.train()
    graph_encoder.train()
    fp_encoder.train()
    proj_2d.train()
    proj_3d.train()
    proj_1d.train()

    channels_last = not args.no_channels_last
    if channels_last:
        img_encoder = img_encoder.to(memory_format=torch.channels_last)

    metrics_history = []
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()
        start = time.time()
        for step, (images, graphs, fps) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            fps = fps.to(device, non_blocking=True)
            graphs = graphs.to(graph_device)

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
                    if graph_device != device:
                        z3d = z3d.to(device)
                    z1d = proj_1d(fp_encoder(fps))

                    loss = clip_loss(z2d, z3d, args.temperature)
                    loss += clip_loss(z2d, z1d, args.temperature)
                    loss += clip_loss(z3d, z1d, args.temperature)
            else:
                z2d = proj_2d(img_encoder(images))
                z3d = proj_3d(graph_encoder(graphs.z, graphs.pos, graphs.batch))
                if graph_device != device:
                    z3d = z3d.to(device)
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
                log(
                    f"epoch {epoch:03d} step {step:05d}/{len(loader)} "
                    f"({percent:5.1f}%) lr {current_lr:.6g}"
                )

        elapsed = time.time() - start
        avg_loss = epoch_loss / max(len(loader), 1)
        log(f"epoch {epoch:03d} loss {avg_loss:.4f} time {elapsed:.1f}s")

        metrics = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "time_sec": round(elapsed, 2),
            "lr": scheduler.get_last_lr()[0],
        }
        metrics_history.append(metrics)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

        if eval_loader and (epoch % args.eval_every == 0 or epoch == args.epochs):
            eval_metrics = _run_eval(
                img_encoder,
                graph_encoder,
                fp_encoder,
                eval_loader,
                device=device,
                graph_device=graph_device,
            )
            eval_metrics["epoch"] = epoch
            with (run_dir / "pretrain_eval.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_metrics) + "\n")
            log(
                "eval "
                + " ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items() if k != "epoch")
            )

        checkpoint = {
            "epoch": epoch,
            "img_encoder": img_encoder.state_dict(),
            "graph_encoder": graph_encoder.state_dict(),
            "fp_encoder": fp_encoder.state_dict(),
            "proj_2d": proj_2d.state_dict(),
            "proj_3d": proj_3d.state_dict(),
            "proj_1d": proj_1d.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        }
        torch.save(checkpoint, run_dir / "checkpoint_last.pt")

    torch.save(img_encoder.state_dict(), run_dir / "resnet18.pt")
    torch.save(graph_encoder.state_dict(), run_dir / "schnet.pt")
    torch.save(fp_encoder.state_dict(), run_dir / "fingerprint_mlp.pt")
    if plt and metrics_history:
        epochs = [m["epoch"] for m in metrics_history]
        losses = [m["avg_loss"] for m in metrics_history]
        lrs = [m["lr"] for m in metrics_history]
        times = [m["time_sec"] for m in metrics_history]

        def save_plot(path: Path, y, title: str, ylabel: str) -> None:
            plt.figure(figsize=(7, 4))
            plt.plot(epochs, y, marker="o", linewidth=1.5)
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()

        save_plot(run_dir / "loss_curve.png", losses, "Pretrain Loss", "Loss")
        save_plot(run_dir / "lr_curve.png", lrs, "Learning Rate", "LR")
        save_plot(run_dir / "epoch_time.png", times, "Epoch Time", "Seconds")
        log("saved plots: loss_curve.png, lr_curve.png, epoch_time.png")
    log(f"saved encoders to {run_dir}")
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
