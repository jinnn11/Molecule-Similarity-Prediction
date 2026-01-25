#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Siamese head with 5-fold CV.")
    parser.add_argument("--features", required=True, help="Path to extracted features .npz")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=5)
    return parser.parse_args()


class GatedFusion(nn.Module):
    def __init__(self, dim_2d: int, dim_3d: int, dim_1d: int, fusion_dim: int):
        super().__init__()
        self.proj_2d = nn.Linear(dim_2d, fusion_dim)
        self.proj_3d = nn.Linear(dim_3d, fusion_dim)
        self.proj_1d = nn.Linear(dim_1d, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 3, 3)

    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor, x1d: torch.Tensor) -> torch.Tensor:
        v2d = self.proj_2d(x2d)
        v3d = self.proj_3d(x3d)
        v1d = self.proj_1d(x1d)
        weights = torch.softmax(self.gate(torch.cat([v2d, v3d, v1d], dim=-1)), dim=-1)
        fused = (
            weights[:, 0:1] * v2d
            + weights[:, 1:2] * v3d
            + weights[:, 2:3] * v1d
        )
        return fused


class SiameseHead(nn.Module):
    def __init__(self, dim_2d: int, dim_3d: int, dim_1d: int, fusion_dim: int, hidden_dim: int):
        super().__init__()
        self.fusion = GatedFusion(dim_2d, dim_3d, dim_1d, fusion_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        a2d: torch.Tensor,
        a3d: torch.Tensor,
        a1d: torch.Tensor,
        b2d: torch.Tensor,
        b3d: torch.Tensor,
        b1d: torch.Tensor,
    ) -> torch.Tensor:
        va = self.fusion(a2d, a3d, a1d)
        vb = self.fusion(b2d, b3d, b1d)
        diff = torch.abs(va - vb)
        return self.mlp(diff).squeeze(-1)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(((x - y) ** 2).mean()))


def _make_folds(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return np.array_split(indices, k)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.features, allow_pickle=True)
    a_2d = data["a_2d"]
    b_2d = data["b_2d"]
    a_3d = data["a_3d"]
    b_3d = data["b_3d"]
    a_1d = data["a_1d"]
    b_1d = data["b_1d"]
    y = data["y"].astype(np.float32)
    tanimoto = data["tanimoto"] if "tanimoto" in data else None
    pair_ids = data["pair_ids"] if "pair_ids" in data else None

    run_dir = Path("outputs") / "finetune_runs" / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_handle.write(msg + "\n")
        log_handle.flush()

    history_path = run_dir / "training_history.jsonl"
    history_handle = history_path.open("w", encoding="utf-8")

    start_time = time.time()
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        run_config = vars(args)
        run_config.update(
            {
                "features_path": str(args.features),
                "num_samples": int(len(y)),
                "feature_shapes": {
                    "a_2d": list(a_2d.shape),
                    "a_3d": list(a_3d.shape),
                    "a_1d": list(a_1d.shape),
                },
            }
        )
        json.dump(run_config, handle, indent=2)

    folds = _make_folds(len(y), args.folds, args.seed)
    metrics = []
    all_preds = np.zeros_like(y)

    for fold_idx in range(args.folds):
        fold_start = time.time()
        log(f"starting fold {fold_idx + 1}/{args.folds}")
        test_idx = folds[fold_idx]
        train_idx = np.hstack([folds[i] for i in range(args.folds) if i != fold_idx])

        model = SiameseHead(
            dim_2d=a_2d.shape[1],
            dim_3d=a_3d.shape[1],
            dim_1d=a_1d.shape[1],
            fusion_dim=args.fusion_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.MSELoss()

        train_ds = TensorDataset(
            torch.from_numpy(a_2d[train_idx]),
            torch.from_numpy(a_3d[train_idx]),
            torch.from_numpy(a_1d[train_idx]),
            torch.from_numpy(b_2d[train_idx]),
            torch.from_numpy(b_3d[train_idx]),
            torch.from_numpy(b_1d[train_idx]),
            torch.from_numpy(y[train_idx]),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            for batch in train_loader:
                batch = [b.to(device) for b in batch]
                pred = model(*batch[:-1])
                loss = loss_fn(pred, batch[-1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss = epoch_loss / max(len(train_loader), 1)
            history.append(epoch_loss)

            history_handle.write(
                json.dumps(
                    {
                        "fold": fold_idx + 1,
                        "epoch": epoch,
                        "loss": round(float(epoch_loss), 6),
                        "elapsed_sec": round(time.time() - epoch_start, 3),
                    }
                )
                + "\n"
            )
            history_handle.flush()
            if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
                elapsed = time.time() - fold_start
                rate = epoch / max(elapsed, 1e-6)
                eta = (args.epochs - epoch) / max(rate, 1e-6)
                log(
                    f"fold {fold_idx + 1} epoch {epoch}/{args.epochs} "
                    f"loss {epoch_loss:.6f} elapsed {elapsed/60:.2f}m eta {eta/60:.2f}m"
                )

        model.eval()
        with torch.no_grad():
            preds = model(
                torch.from_numpy(a_2d[test_idx]).to(device),
                torch.from_numpy(a_3d[test_idx]).to(device),
                torch.from_numpy(a_1d[test_idx]).to(device),
                torch.from_numpy(b_2d[test_idx]).to(device),
                torch.from_numpy(b_3d[test_idx]).to(device),
                torch.from_numpy(b_1d[test_idx]).to(device),
            ).cpu().numpy()
        all_preds[test_idx] = preds

        fold_rmse = _rmse(preds, y[test_idx])
        fold_pearson = _pearson(preds, y[test_idx])
        metrics.append(
            {
                "fold": fold_idx + 1,
                "rmse": fold_rmse,
                "pearson": fold_pearson,
                "train_time_sec": round(time.time() - fold_start, 2),
            }
        )
        torch.save(model.state_dict(), run_dir / f"head_fold_{fold_idx + 1}.pt")

        if plt:
            plt.figure(figsize=(7, 4))
            plt.plot(history, linewidth=1.5)
            plt.title(f"Fold {fold_idx + 1} Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.tight_layout()
            plt.savefig(run_dir / f"loss_fold_{fold_idx + 1}.png", dpi=150)
            plt.close()

    summary = {
        "rmse_mean": float(np.mean([m["rmse"] for m in metrics])),
        "rmse_std": float(np.std([m["rmse"] for m in metrics])),
        "pearson_mean": float(np.mean([m["pearson"] for m in metrics])),
        "pearson_std": float(np.std([m["pearson"] for m in metrics])),
        "runtime_sec": round(time.time() - start_time, 2),
    }
    if tanimoto is not None:
        mask = np.isfinite(tanimoto)
        if mask.any():
            summary["tanimoto_pearson"] = _pearson(tanimoto[mask], y[mask])

    with (run_dir / "fold_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    pred_path = run_dir / "predictions.csv"
    with pred_path.open("w", encoding="utf-8") as handle:
        if pair_ids is not None:
            handle.write("index,pair_id,pred,y\n")
            for i, pred in enumerate(all_preds):
                handle.write(f"{i},{pair_ids[i]},{pred:.6f},{y[i]:.6f}\n")
        else:
            handle.write("index,pred,y\n")
            for i, pred in enumerate(all_preds):
                handle.write(f"{i},{pred:.6f},{y[i]:.6f}\n")

    print(f"wrote results to {run_dir}")
    history_handle.close()
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
