#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
import math
from pathlib import Path
from typing import List

import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from rdkit import Chem

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "pretraining"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import FingerprintMLP, GraphTower, build_resnet18  # noqa: E402
from utils.chem import render_image, fingerprint, load_pdb, embed_3d, get_pdb_path  # noqa: E402
from utils.metrics import pearson, spearman, rmse, mae, r2, make_folds, split_train_val  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Light finetuning (last layers only).")
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-device", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-pdb", action="store_true")
    parser.add_argument("--drop-2d", action="store_true")
    parser.add_argument("--include-tanimoto", action="store_true")
    parser.add_argument("--experiment-name", default="light_finetune")
    return parser.parse_args()


def _load_pairs(csv_path: Path) -> List[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            row["__csv_path"] = str(csv_path)
            rows.append(row)
    return rows


class PairDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        image_size: int,
        no_pdb: bool,
        include_tanimoto: bool,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.no_pdb = no_pdb
        self.include_tanimoto = include_tanimoto
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
        if mol is None:
            raise ValueError("invalid smiles")
        img = render_image(mol, self.image_size)
        self.image_cache[smiles] = img
        return img

    def _get_fp(self, smiles: str) -> torch.Tensor:
        cached = self.fp_cache.get(smiles)
        if cached is not None:
            return cached
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("invalid smiles")
        fp = fingerprint(mol)
        self.fp_cache[smiles] = fp
        return fp

    def _get_graph(self, row: dict, side: str) -> Data:
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
        tan = float(row.get("TanimotoCombo", "nan")) if self.include_tanimoto else float("nan")
        if self.include_tanimoto and not math.isfinite(tan):
            tan = 0.0
        return img_a, graph_a, fp_a, img_b, graph_b, fp_b, label, tan


def _collate(batch):
    imgs_a, graphs_a, fps_a, imgs_b, graphs_b, fps_b, labels, tans = zip(*batch)
    return (
        torch.stack(imgs_a, dim=0),
        Batch.from_data_list(list(graphs_a)),
        torch.stack(fps_a, dim=0),
        torch.stack(imgs_b, dim=0),
        Batch.from_data_list(list(graphs_b)),
        torch.stack(fps_b, dim=0),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(tans, dtype=torch.float32),
    )


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_resnet_last(model: nn.Module) -> None:
    if hasattr(model, "layer4"):
        for p in model.layer4.parameters():
            p.requires_grad = True


def _unfreeze_fp_last(model: FingerprintMLP) -> None:
    if hasattr(model, "net") and isinstance(model.net[-1], nn.Linear):
        for p in model.net[-1].parameters():
            p.requires_grad = True


def _unfreeze_graph_last(model: GraphTower) -> None:
    schnet = model.model
    if hasattr(schnet, "lin2"):
        for p in schnet.lin2.parameters():
            p.requires_grad = True
    for attr in ("interactions", "interaction_blocks"):
        blocks = getattr(schnet, attr, None)
        if blocks is not None and len(blocks) > 0:
            for p in blocks[-1].parameters():
                p.requires_grad = True
            break


class CosineHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _prepare_models(weights_dir: Path, device: torch.device, graph_device: torch.device):
    img_encoder = build_resnet18().to(device)
    graph_encoder = GraphTower(out_dim=128).to(graph_device)
    fp_encoder = FingerprintMLP(in_dim=2048, hidden_dim=1024, out_dim=512).to(device)

    img_encoder.load_state_dict(torch.load(weights_dir / "resnet18.pt", map_location=device))
    graph_encoder.load_state_dict(torch.load(weights_dir / "schnet.pt", map_location=graph_device))
    fp_encoder.load_state_dict(torch.load(weights_dir / "fingerprint_mlp.pt", map_location=device))

    for model in (img_encoder, graph_encoder, fp_encoder):
        _freeze_all(model)

    _unfreeze_resnet_last(img_encoder)
    _unfreeze_graph_last(graph_encoder)
    _unfreeze_fp_last(fp_encoder)

    return img_encoder, graph_encoder, fp_encoder


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if args.graph_device:
        graph_device = torch.device(args.graph_device)
    else:
        graph_device = torch.device("cpu") if device.type == "mps" else device

    pairs: List[dict] = []
    for csv_file in args.csv:
        pairs.extend(_load_pairs(Path(csv_file)))
    labels_arr = np.array([float(row["frac_similar"]) for row in pairs], dtype=np.float32)

    dataset = PairDataset(
        pairs,
        image_size=args.image_size,
        no_pdb=args.no_pdb,
        include_tanimoto=args.include_tanimoto,
    )

    run_dir = Path("outputs") / "finetune_runs" / args.experiment_name / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_handle.write(msg + "\n")
        log_handle.flush()

    history_path = run_dir / "training_history.jsonl"
    history_handle = history_path.open("w", encoding="utf-8")

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    folds = make_folds(len(dataset), args.folds, args.seed)
    metrics = []
    all_preds = np.zeros(len(dataset), dtype=np.float32)
    start_time = time.time()

    log("starting light finetuning")
    for fold_idx in range(args.folds):
        fold_start = time.time()
        test_idx = folds[fold_idx]
        train_idx = np.hstack([folds[i] for i in range(args.folds) if i != fold_idx])
        train_idx, val_idx = split_train_val(train_idx, args.val_split, args.seed + fold_idx + 1)

        img_encoder, graph_encoder, fp_encoder = _prepare_models(Path(args.weights_dir), device, graph_device)
        head_dim = 2 if args.drop_2d else 3
        if args.include_tanimoto:
            head_dim += 1
        head = CosineHead(head_dim).to(device)

        params = [p for m in (img_encoder, graph_encoder, fp_encoder, head) for p in m.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            collate_fn=_collate,
        )
        val_loader = None
        if val_idx.size > 0:
            val_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(val_idx),
                collate_fn=_collate,
            )

        best_val = float("inf")
        best_state = None
        for epoch in range(1, args.epochs + 1):
            img_encoder.train()
            graph_encoder.train()
            fp_encoder.train()
            head.train()
            epoch_loss = 0.0
            for batch in train_loader:
                img_a, graph_a, fp_a, img_b, graph_b, fp_b, labels, tans = batch
                img_a = img_a.to(device)
                img_b = img_b.to(device)
                fp_a = fp_a.to(device)
                fp_b = fp_b.to(device)
                labels = labels.to(device)
                graph_a = graph_a.to(graph_device)
                graph_b = graph_b.to(graph_device)

                emb_a2d = img_encoder(img_a)
                emb_b2d = img_encoder(img_b)
                emb_a1d = fp_encoder(fp_a)
                emb_b1d = fp_encoder(fp_b)
                emb_a3d = graph_encoder(graph_a.z, graph_a.pos, graph_a.batch)
                emb_b3d = graph_encoder(graph_b.z, graph_b.pos, graph_b.batch)

                cos2d = torch.nn.functional.cosine_similarity(emb_a2d, emb_b2d)
                cos3d = torch.nn.functional.cosine_similarity(emb_a3d, emb_b3d)
                cos1d = torch.nn.functional.cosine_similarity(emb_a1d, emb_b1d)

                feats = []
                if not args.drop_2d:
                    feats.append(cos2d)
                feats.extend([cos3d, cos1d])
                if args.include_tanimoto:
                    feats.append(tans.to(device))
                x = torch.stack(feats, dim=1)
                preds = head(x)
                loss = loss_fn(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= max(len(train_loader), 1)
            val_loss = None
            if val_loader is not None:
                img_encoder.eval()
                graph_encoder.eval()
                fp_encoder.eval()
                head.eval()
                vloss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        img_a, graph_a, fp_a, img_b, graph_b, fp_b, labels, tans = batch
                        img_a = img_a.to(device)
                        img_b = img_b.to(device)
                        fp_a = fp_a.to(device)
                        fp_b = fp_b.to(device)
                        labels = labels.to(device)
                        graph_a = graph_a.to(graph_device)
                        graph_b = graph_b.to(graph_device)

                        emb_a2d = img_encoder(img_a)
                        emb_b2d = img_encoder(img_b)
                        emb_a1d = fp_encoder(fp_a)
                        emb_b1d = fp_encoder(fp_b)
                        emb_a3d = graph_encoder(graph_a.z, graph_a.pos, graph_a.batch)
                        emb_b3d = graph_encoder(graph_b.z, graph_b.pos, graph_b.batch)

                        cos2d = torch.nn.functional.cosine_similarity(emb_a2d, emb_b2d)
                        cos3d = torch.nn.functional.cosine_similarity(emb_a3d, emb_b3d)
                        cos1d = torch.nn.functional.cosine_similarity(emb_a1d, emb_b1d)

                        feats = []
                        if not args.drop_2d:
                            feats.append(cos2d)
                        feats.extend([cos3d, cos1d])
                        if args.include_tanimoto:
                            feats.append(tans.to(device))
                        x = torch.stack(feats, dim=1)
                        preds = head(x)
                        vloss += loss_fn(preds, labels).item()
                val_loss = vloss / max(len(val_loader), 1)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

            history_handle.write(
                json.dumps(
                    {
                        "fold": fold_idx + 1,
                        "epoch": epoch,
                        "loss": round(float(epoch_loss), 6),
                        "val_loss": round(float(val_loss), 6) if val_loss is not None else None,
                    }
                )
                + "\n"
            )
            history_handle.flush()
            if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
                msg = (
                    f"fold {fold_idx + 1} epoch {epoch}/{args.epochs} "
                    f"loss {epoch_loss:.6f}"
                )
                if val_loss is not None:
                    msg += f" val {val_loss:.6f}"
                log(msg)

        if best_state is not None:
            head.load_state_dict(best_state)

        img_encoder.eval()
        graph_encoder.eval()
        fp_encoder.eval()
        head.eval()
        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
            collate_fn=_collate,
        )
        preds_list = []
        targets_list = []
        with torch.no_grad():
            for batch in test_loader:
                img_a, graph_a, fp_a, img_b, graph_b, fp_b, labels, tans = batch
                img_a = img_a.to(device)
                img_b = img_b.to(device)
                fp_a = fp_a.to(device)
                fp_b = fp_b.to(device)
                labels = labels.to(device)
                graph_a = graph_a.to(graph_device)
                graph_b = graph_b.to(graph_device)

                emb_a2d = img_encoder(img_a)
                emb_b2d = img_encoder(img_b)
                emb_a1d = fp_encoder(fp_a)
                emb_b1d = fp_encoder(fp_b)
                emb_a3d = graph_encoder(graph_a.z, graph_a.pos, graph_a.batch)
                emb_b3d = graph_encoder(graph_b.z, graph_b.pos, graph_b.batch)

                cos2d = torch.nn.functional.cosine_similarity(emb_a2d, emb_b2d)
                cos3d = torch.nn.functional.cosine_similarity(emb_a3d, emb_b3d)
                cos1d = torch.nn.functional.cosine_similarity(emb_a1d, emb_b1d)
                feats = []
                if not args.drop_2d:
                    feats.append(cos2d)
                feats.extend([cos3d, cos1d])
                if args.include_tanimoto:
                    feats.append(tans.to(device))
                x = torch.stack(feats, dim=1)
                preds = head(x)
                preds_list.append(preds.cpu().numpy())
                targets_list.append(labels.cpu().numpy())

        preds_np = np.concatenate(preds_list)
        targets_np = np.concatenate(targets_list)
        all_preds[test_idx] = preds_np
        fold_rmse = rmse(preds_np, targets_np)
        fold_mae = mae(preds_np, targets_np)
        fold_pearson = pearson(preds_np, targets_np)
        fold_spearman = spearman(preds_np, targets_np)
        fold_r2 = r2(preds_np, targets_np)
        metrics.append(
            {
                "fold": fold_idx + 1,
                "rmse": fold_rmse,
                "mae": fold_mae,
                "pearson": fold_pearson,
                "spearman": fold_spearman,
                "r2": fold_r2,
                "val_best": float(best_val) if best_state is not None else None,
                "train_time_sec": round(time.time() - fold_start, 2),
            }
        )

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
        "overall_rmse": rmse(all_preds, labels_arr),
        "overall_pearson": pearson(all_preds, labels_arr),
        "runtime_sec": round(time.time() - start_time, 2),
    }

    with (run_dir / "fold_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    log(f"wrote results to {run_dir}")
    log_handle.close()
    history_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
