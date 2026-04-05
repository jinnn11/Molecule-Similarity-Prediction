#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import pearson, spearman, rmse, mae, r2, make_folds, split_train_val  # noqa: E402

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
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--drop-2d", action="store_true", help="Ignore 2D image features.")
    parser.add_argument(
        "--fusion-mode",
        choices=["gate", "concat"],
        default="gate",
        help="Fuse modalities with gating or simple concatenation.",
    )
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=5)
    return parser.parse_args()


class GatedFusion(nn.Module):
    def __init__(self, dims: List[int], fusion_dim: int):
        super().__init__()
        if not dims:
            raise ValueError("At least one modality is required.")
        self.projs = nn.ModuleList([nn.Linear(dim, fusion_dim) for dim in dims])
        self.gate = nn.Linear(fusion_dim * len(dims), len(dims))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        vectors = [proj(x) for proj, x in zip(self.projs, features)]
        weights = torch.softmax(self.gate(torch.cat(vectors, dim=-1)), dim=-1)
        fused = torch.zeros_like(vectors[0])
        for i, vec in enumerate(vectors):
            fused = fused + weights[:, i : i + 1] * vec
        return fused


class ConcatFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(features, dim=-1)


class SiameseHead(nn.Module):
    def __init__(
        self,
        dim_2d: int,
        dim_3d: int,
        dim_1d: int,
        fusion_dim: int,
        hidden_dim: int,
        use_2d: bool = True,
        use_3d: bool = True,
        use_1d: bool = True,
        fusion_mode: str = "gate",
    ):
        super().__init__()
        self.use_2d = use_2d
        self.use_3d = use_3d
        self.use_1d = use_1d
        dims = []
        if use_2d:
            dims.append(dim_2d)
        if use_3d:
            dims.append(dim_3d)
        if use_1d:
            dims.append(dim_1d)
        if fusion_mode == "gate":
            self.fusion = GatedFusion(dims, fusion_dim)
            fusion_out = fusion_dim
        else:
            self.fusion = ConcatFusion()
            fusion_out = int(sum(dims))
        self.fusion_mode = fusion_mode
        self.mlp = nn.Sequential(
            nn.Linear(fusion_out, hidden_dim),
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
        feats_a = []
        feats_b = []
        if self.use_2d:
            feats_a.append(a2d)
            feats_b.append(b2d)
        if self.use_3d:
            feats_a.append(a3d)
            feats_b.append(b3d)
        if self.use_1d:
            feats_a.append(a1d)
            feats_b.append(b1d)
        va = self.fusion(feats_a)
        vb = self.fusion(feats_b)
        diff = torch.abs(va - vb)
        return self.mlp(diff).squeeze(-1)


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

    exp_name = args.experiment_name.strip().replace(" ", "_")
    run_dir = Path("outputs") / "finetune_runs" / exp_name / time.strftime("run_%Y%m%d_%H%M%S")
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
    modality_names = []
    if not args.drop_2d:
        modality_names.append("2d")
    modality_names.extend(["3d", "1d"])
    log(f"modalities: {'+'.join(modality_names)} fusion={args.fusion_mode}")
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        run_config = vars(args)
        run_config["modalities"] = {
            "use_2d": not args.drop_2d,
            "use_3d": True,
            "use_1d": True,
        }
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

    folds = make_folds(len(y), args.folds, args.seed)
    metrics = []
    all_preds = np.zeros_like(y)
    use_2d = not args.drop_2d

    for fold_idx in range(args.folds):
        fold_start = time.time()
        log(f"starting fold {fold_idx + 1}/{args.folds}")
        test_idx = folds[fold_idx]
        train_idx = np.hstack([folds[i] for i in range(args.folds) if i != fold_idx])
        train_idx, val_idx = split_train_val(train_idx, args.val_split, args.seed + fold_idx + 1)

        model = SiameseHead(
            dim_2d=a_2d.shape[1],
            dim_3d=a_3d.shape[1],
            dim_1d=a_1d.shape[1],
            fusion_dim=args.fusion_dim,
            hidden_dim=args.hidden_dim,
            use_2d=use_2d,
            fusion_mode=args.fusion_mode,
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
        val_loader = None
        if val_idx.size > 0:
            val_ds = TensorDataset(
                torch.from_numpy(a_2d[val_idx]),
                torch.from_numpy(a_3d[val_idx]),
                torch.from_numpy(a_1d[val_idx]),
                torch.from_numpy(b_2d[val_idx]),
                torch.from_numpy(b_3d[val_idx]),
                torch.from_numpy(b_1d[val_idx]),
                torch.from_numpy(y[val_idx]),
            )
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

        history = []
        best_val = float("inf")
        best_state = None
        best_epoch = None
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

            val_loss = None
            val_rmse = None
            val_mae = None
            val_pearson = None
            val_spearman = None
            val_r2 = None
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    vloss = 0.0
                    vpreds = []
                    vtargets = []
                    for batch in val_loader:
                        batch = [b.to(device) for b in batch]
                        pred = model(*batch[:-1])
                        loss = loss_fn(pred, batch[-1])
                        vloss += loss.item()
                        vpreds.append(pred.detach().cpu().numpy())
                        vtargets.append(batch[-1].detach().cpu().numpy())
                val_loss = vloss / max(len(val_loader), 1)
                if vpreds:
                    vpreds_np = np.concatenate(vpreds).astype(np.float32)
                    vtargets_np = np.concatenate(vtargets).astype(np.float32)
                    val_rmse = rmse(vpreds_np, vtargets_np)
                    val_mae = mae(vpreds_np, vtargets_np)
                    val_pearson = pearson(vpreds_np, vtargets_np)
                    val_spearman = spearman(vpreds_np, vtargets_np)
                    val_r2 = r2(vpreds_np, vtargets_np)
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            history_handle.write(
                json.dumps(
                    {
                        "fold": fold_idx + 1,
                        "epoch": epoch,
                        "loss": round(float(epoch_loss), 6),
                        "train_rmse": round(float(math.sqrt(epoch_loss)), 6),
                        "val_loss": round(float(val_loss), 6) if val_loss is not None else None,
                        "val_rmse": round(float(val_rmse), 6) if val_rmse is not None else None,
                        "val_mae": round(float(val_mae), 6) if val_mae is not None else None,
                        "val_pearson": round(float(val_pearson), 6) if val_pearson is not None else None,
                        "val_spearman": round(float(val_spearman), 6) if val_spearman is not None else None,
                        "val_r2": round(float(val_r2), 6) if val_r2 is not None else None,
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
                if val_loss is not None:
                    msg = (
                        f"fold {fold_idx + 1} epoch {epoch}/{args.epochs} "
                        f"loss {epoch_loss:.6f} val {val_loss:.6f} "
                        f"val_rmse {val_rmse:.4f} val_pearson {val_pearson:.3f} "
                        f"elapsed {elapsed/60:.2f}m eta {eta/60:.2f}m"
                    )
                else:
                    msg = (
                        f"fold {fold_idx + 1} epoch {epoch}/{args.epochs} "
                        f"loss {epoch_loss:.6f} "
                        f"elapsed {elapsed/60:.2f}m eta {eta/60:.2f}m"
                    )
                log(msg)

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        val_metrics = None
        if val_idx.size > 0:
            with torch.no_grad():
                val_preds = model(
                    torch.from_numpy(a_2d[val_idx]).to(device),
                    torch.from_numpy(a_3d[val_idx]).to(device),
                    torch.from_numpy(a_1d[val_idx]).to(device),
                    torch.from_numpy(b_2d[val_idx]).to(device),
                    torch.from_numpy(b_3d[val_idx]).to(device),
                    torch.from_numpy(b_1d[val_idx]).to(device),
                ).cpu().numpy()
            val_metrics = {
                "rmse": rmse(val_preds, y[val_idx]),
                "mae": mae(val_preds, y[val_idx]),
                "pearson": pearson(val_preds, y[val_idx]),
                "spearman": spearman(val_preds, y[val_idx]),
                "r2": r2(val_preds, y[val_idx]),
            }
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

        fold_rmse = rmse(preds, y[test_idx])
        fold_mae = mae(preds, y[test_idx])
        fold_pearson = pearson(preds, y[test_idx])
        fold_spearman = spearman(preds, y[test_idx])
        fold_r2 = r2(preds, y[test_idx])
        metrics.append(
            {
                "fold": fold_idx + 1,
                "rmse": fold_rmse,
                "mae": fold_mae,
                "pearson": fold_pearson,
                "spearman": fold_spearman,
                "r2": fold_r2,
                "val_best": float(best_val) if best_state is not None else None,
                "val_best_epoch": int(best_epoch) if best_epoch is not None else None,
                "val_metrics": val_metrics,
                "train_time_sec": round(time.time() - fold_start, 2),
                "train_size": int(train_idx.size),
                "val_size": int(val_idx.size),
                "test_size": int(test_idx.size),
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
        "mae_mean": float(np.mean([m["mae"] for m in metrics])),
        "mae_std": float(np.std([m["mae"] for m in metrics])),
        "pearson_mean": float(np.mean([m["pearson"] for m in metrics])),
        "pearson_std": float(np.std([m["pearson"] for m in metrics])),
        "spearman_mean": float(np.mean([m["spearman"] for m in metrics])),
        "spearman_std": float(np.std([m["spearman"] for m in metrics])),
        "r2_mean": float(np.mean([m["r2"] for m in metrics])),
        "r2_std": float(np.std([m["r2"] for m in metrics])),
        "overall_rmse": rmse(all_preds, y),
        "overall_mae": mae(all_preds, y),
        "overall_pearson": pearson(all_preds, y),
        "overall_spearman": spearman(all_preds, y),
        "overall_r2": r2(all_preds, y),
        "runtime_sec": round(time.time() - start_time, 2),
    }
    if tanimoto is not None:
        mask = np.isfinite(tanimoto)
        if mask.any():
            summary["tanimoto_pearson"] = pearson(tanimoto[mask], y[mask])
            summary["tanimoto_spearman"] = spearman(tanimoto[mask], y[mask])

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
