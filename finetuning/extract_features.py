#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torchvision import transforms

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Draw
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        rdMolDraw2D = None
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError:  # pragma: no cover
        GetMorganGenerator = None
except ImportError as exc:  # pragma: no cover
    raise ImportError("RDKit is required for feature extraction.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "pretraining"))

from models import FingerprintMLP, GraphTower, build_resnet18  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract tri-modal features for finetuning.")
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="CSV file with pairs (repeatable).",
    )
    parser.add_argument(
        "--weights-dir",
        required=True,
        help="Folder containing resnet18.pt, schnet.pt, fingerprint_mlp.pt",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-device", default=None)
    parser.add_argument("--out", default="outputs/finetune_features/features.npz")
    parser.add_argument("--no-pdb", action="store_true", help="Do not use PDB; use RDKit 3D.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser.parse_args()


def _image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _render_image(mol: "Chem.Mol", image_size: int) -> torch.Tensor:
    if rdMolDraw2D is not None and hasattr(rdMolDraw2D, "MolDraw2DCairo"):
        drawer = rdMolDraw2D.MolDraw2DCairo(image_size, image_size)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(png)).convert("RGB")
    else:
        img = Draw.MolToImage(mol, size=(image_size, image_size))
    return _image_transform(image_size)(img)


_MORGAN_GEN = {}


def _fingerprint(mol: "Chem.Mol", bits: int = 2048) -> torch.Tensor:
    if GetMorganGenerator is not None:
        gen = _MORGAN_GEN.get(bits)
        if gen is None:
            gen = GetMorganGenerator(radius=2, fpSize=bits)
            _MORGAN_GEN[bits] = gen
        fp = gen.GetFingerprint(mol)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
    arr = np.zeros((bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.from_numpy(arr)


def _load_pdb(pdb_path: Path) -> Optional[Data]:
    if not pdb_path.exists():
        return None
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    return Data(z=z, pos=pos)


def _embed_3d(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    return Data(z=z, pos=pos)


def _get_pdb_path(csv_path: Path, id_pair: int, side: str) -> Path:
    base = csv_path.parent / "conformers_3D"
    return base / f"best_rocs_conformer_{id_pair:03d}{side}.pdb"


def _load_pairs(csv_path: Path) -> List[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            row["__csv_path"] = str(csv_path)
            rows.append(row)
    return rows


def _prepare_models(weights_dir: Path, device: torch.device, graph_device: torch.device):
    img_encoder = build_resnet18().to(device).eval()
    graph_encoder = GraphTower(out_dim=128).to(graph_device).eval()
    fp_encoder = FingerprintMLP(in_dim=2048, hidden_dim=1024, out_dim=512).to(device).eval()

    img_encoder.load_state_dict(torch.load(weights_dir / "resnet18.pt", map_location=device))
    graph_encoder.load_state_dict(torch.load(weights_dir / "schnet.pt", map_location=graph_device))
    fp_encoder.load_state_dict(torch.load(weights_dir / "fingerprint_mlp.pt", map_location=device))

    for model in (img_encoder, graph_encoder, fp_encoder):
        for p in model.parameters():
            p.requires_grad = False
    return img_encoder, graph_encoder, fp_encoder


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if args.graph_device:
        graph_device = torch.device(args.graph_device)
    else:
        graph_device = torch.device("cpu") if device.type == "mps" else device

    weights_dir = Path(args.weights_dir)
    img_encoder, graph_encoder, fp_encoder = _prepare_models(weights_dir, device, graph_device)

    pairs = []
    for csv_file in args.csv:
        pairs.extend(_load_pairs(Path(csv_file)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "extract.log"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_handle.write(msg + "\n")
        log_handle.flush()

    a_2d, a_3d, a_1d = [], [], []
    b_2d, b_3d, b_1d = [], [], []
    labels = []
    tanimoto = []
    pair_ids = []
    skipped = []
    stats = {
        "seen": 0,
        "kept": 0,
        "skipped": 0,
        "pdb_used": 0,
        "pdb_missing": 0,
        "pdb_failed": 0,
        "embed_used": 0,
        "embed_failed": 0,
        "invalid_smiles": 0,
    }
    total = len(pairs)

    start = time.time()
    log(
        "starting extraction: "
        f"pairs={total} device={device.type} graph_device={graph_device.type}"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        with torch.no_grad():
            for idx, row in enumerate(pairs, start=1):
                stats["seen"] += 1
                try:
                    id_pair = int(row.get("id_pair", "0"))
                    csv_path = Path(row["__csv_path"])
                    pair_key = f"{csv_path.stem}:{id_pair:03d}"
                    smiles_a = row["curated_smiles_molecule_a"]
                    smiles_b = row["curated_smiles_molecule_b"]
                    label = float(row["frac_similar"])
                    tan = float(row.get("TanimotoCombo", "nan")) if "TanimotoCombo" in row else float("nan")

                    mol_a = Chem.MolFromSmiles(smiles_a)
                    mol_b = Chem.MolFromSmiles(smiles_b)
                    if mol_a is None or mol_b is None:
                        stats["invalid_smiles"] += 1
                        stats["skipped"] += 1
                        skipped.append({"pair_id": pair_key, "reason": "invalid_smiles"})
                        continue

                    img_a = _render_image(mol_a, args.image_size).unsqueeze(0).to(device)
                    img_b = _render_image(mol_b, args.image_size).unsqueeze(0).to(device)

                    fp_a = _fingerprint(mol_a).unsqueeze(0).to(device)
                    fp_b = _fingerprint(mol_b).unsqueeze(0).to(device)

                    graph_a = None
                    graph_b = None
                    if args.no_pdb:
                        graph_a = _embed_3d(smiles_a)
                        graph_b = _embed_3d(smiles_b)
                        if graph_a is not None and graph_b is not None:
                            stats["embed_used"] += 2
                    else:
                        pdb_a = _get_pdb_path(csv_path, id_pair, "a")
                        pdb_b = _get_pdb_path(csv_path, id_pair, "b")
                        graph_a = _load_pdb(pdb_a)
                        graph_b = _load_pdb(pdb_b)
                        if graph_a is None:
                            if pdb_a.exists():
                                stats["pdb_failed"] += 1
                            else:
                                stats["pdb_missing"] += 1
                            graph_a = _embed_3d(smiles_a)
                            if graph_a is not None:
                                stats["embed_used"] += 1
                            else:
                                stats["embed_failed"] += 1
                        else:
                            stats["pdb_used"] += 1
                        if graph_b is None:
                            if pdb_b.exists():
                                stats["pdb_failed"] += 1
                            else:
                                stats["pdb_missing"] += 1
                            graph_b = _embed_3d(smiles_b)
                            if graph_b is not None:
                                stats["embed_used"] += 1
                            else:
                                stats["embed_failed"] += 1
                        else:
                            stats["pdb_used"] += 1

                    if graph_a is None or graph_b is None:
                        stats["skipped"] += 1
                        skipped.append({"pair_id": pair_key, "reason": "missing_3d"})
                        continue

                    graph_a = graph_a.to(graph_device)
                    graph_b = graph_b.to(graph_device)

                    feat_a_2d = img_encoder(img_a).cpu().numpy().squeeze(0)
                    feat_b_2d = img_encoder(img_b).cpu().numpy().squeeze(0)
                    feat_a_1d = fp_encoder(fp_a).cpu().numpy().squeeze(0)
                    feat_b_1d = fp_encoder(fp_b).cpu().numpy().squeeze(0)

                    feat_a_3d = graph_encoder(graph_a.z, graph_a.pos, None).cpu().numpy().squeeze(0)
                    feat_b_3d = graph_encoder(graph_b.z, graph_b.pos, None).cpu().numpy().squeeze(0)

                    a_2d.append(feat_a_2d)
                    b_2d.append(feat_b_2d)
                    a_1d.append(feat_a_1d)
                    b_1d.append(feat_b_1d)
                    a_3d.append(feat_a_3d)
                    b_3d.append(feat_b_3d)
                    labels.append(label)
                    tanimoto.append(tan)
                    pair_ids.append(pair_key)
                    stats["kept"] += 1
                except Exception as exc:  # pragma: no cover
                    stats["skipped"] += 1
                    skipped.append({"pair_id": row.get("id_pair", "unknown"), "reason": str(exc)})

                if idx % args.log_interval == 0 or idx == total:
                    elapsed = time.time() - start
                    rate = stats["seen"] / max(elapsed, 1e-6)
                    remaining = total - stats["seen"]
                    eta = remaining / max(rate, 1e-6)
                    log(
                        f"progress {stats['seen']}/{total} "
                        f"({stats['seen'] / max(total, 1) * 100:.1f}%) "
                        f"kept {stats['kept']} skipped {stats['skipped']} "
                        f"rate {rate:.2f}/s elapsed {elapsed/60:.1f}m eta {eta/60:.1f}m"
                    )
    finally:
        log_handle.close()

    np.savez_compressed(
        out_path,
        a_2d=np.asarray(a_2d),
        b_2d=np.asarray(b_2d),
        a_3d=np.asarray(a_3d),
        b_3d=np.asarray(b_3d),
        a_1d=np.asarray(a_1d),
        b_1d=np.asarray(b_1d),
        y=np.asarray(labels),
        tanimoto=np.asarray(tanimoto),
        pair_ids=np.asarray(pair_ids),
        image_size=args.image_size,
    )

    meta = {
        "pairs_total": total,
        "pairs_kept": len(labels),
        "pairs_skipped": stats["skipped"],
        "csv_files": args.csv,
        "weights_dir": str(weights_dir),
        "device": str(device),
        "graph_device": str(graph_device),
        "stats": stats,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rdkit": Chem.rdchem._GetRDKitVersion() if hasattr(Chem, "rdchem") else "unknown",
        },
        "feature_shapes": {
            "a_2d": list(np.asarray(a_2d).shape),
            "b_2d": list(np.asarray(b_2d).shape),
            "a_3d": list(np.asarray(a_3d).shape),
            "b_3d": list(np.asarray(b_3d).shape),
            "a_1d": list(np.asarray(a_1d).shape),
            "b_1d": list(np.asarray(b_1d).shape),
        },
        "elapsed_sec": round(time.time() - start, 2),
    }
    with (out_path.parent / "extract_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    if skipped:
        with (out_path.parent / "skipped_pairs.jsonl").open("w", encoding="utf-8") as handle:
            for item in skipped:
                handle.write(json.dumps(item) + "\n")

    print(f"wrote features to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
