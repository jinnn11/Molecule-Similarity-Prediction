#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import msgpack
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("msgpack is required to read the dataset archive.") from exc

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Draw
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("RDKit is required for SMILES, images, and fingerprints.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute images, fingerprints, and 3D graphs.")
    parser.add_argument("--archive", required=True, help="Path to drugs_crude.msgpack.tar.gz")
    parser.add_argument("--out", required=True, help="Output directory for precomputed data")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--max-mols", type=int, default=0, help="Optional cap for a quick smoke test")
    return parser.parse_args()


def _iter_msgpack(path: Path) -> Iterable[dict]:
    with path.open("rb") as handle:
        unpacker = msgpack.Unpacker(handle, raw=False)
        for obj in unpacker:
            if isinstance(obj, dict):
                yield obj


def _resolve_msgpack(path: Path, out_dir: Path) -> Path:
    if path.suffix == ".msgpack":
        return path
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".msgpack")]
            if not members:
                raise ValueError("No .msgpack file found in archive.")
            if len(members) > 1:
                raise ValueError("Archive contains multiple .msgpack files; pick one.")
            tar.extract(members[0], path=out_dir)
            return out_dir / members[0].name
    raise ValueError("Expected a .msgpack or .tar.gz/.tgz archive.")


def _pick_atoms(entry: dict) -> Optional[list]:
    atoms = entry.get("atoms") or entry.get("atom_numbers") or entry.get("atomic_numbers")
    if atoms and isinstance(atoms[0], dict):
        atoms = [a.get("atomic_num") or a.get("atomic_number") or a.get("z") for a in atoms]
    return atoms


def _pick_xyz(entry: dict) -> Optional[np.ndarray]:
    conformers = entry.get("conformers") or []
    if conformers:
        def conf_energy(c):
            return c.get("totalenergy", c.get("energy", 0.0))
        conf = min(conformers, key=conf_energy)
        xyz = conf.get("xyz") or conf.get("coords") or conf.get("positions")
    else:
        xyz = entry.get("xyz") or entry.get("coords") or entry.get("positions")
    if xyz is None:
        return None
    if isinstance(xyz[0], dict):
        xyz = [[p.get("x"), p.get("y"), p.get("z")] for p in xyz]
    return np.asarray(xyz, dtype=np.float32)


def _fingerprint(mol: "Chem.Mol", bits: int) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
    arr = np.zeros((bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    msgpack_path = _resolve_msgpack(Path(args.archive), out_dir / "raw")

    fingerprints = []
    z_list = []
    pos_list = []
    smiles_list = []

    kept = 0
    for entry in _iter_msgpack(msgpack_path):
        if args.max_mols and kept >= args.max_mols:
            break

        smiles = entry.get("smiles")
        atoms = _pick_atoms(entry)
        xyz = _pick_xyz(entry)
        if smiles is None or atoms is None or xyz is None:
            continue
        if len(atoms) != len(xyz):
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        fp = _fingerprint(mol, args.fingerprint_bits)
        fingerprints.append(np.packbits(fp))

        z_list.append(np.asarray(atoms, dtype=np.int64))
        pos_list.append(np.asarray(xyz, dtype=np.float32))
        smiles_list.append(smiles)

        img = Draw.MolToImage(mol, size=(args.image_size, args.image_size))
        img.save(images_dir / f"{kept:06d}.png", format="PNG")

        kept += 1
        if kept % 1000 == 0:
            print(f"processed {kept}")

    if not fingerprints:
        raise RuntimeError("No molecules processed. Check the archive format.")

    z_all = np.concatenate(z_list, axis=0)
    pos_all = np.concatenate(pos_list, axis=0)
    lengths = np.array([len(z) for z in z_list], dtype=np.int32)
    offsets = np.zeros_like(lengths, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths[:-1])

    np.save(out_dir / "fingerprints.npy", np.stack(fingerprints, axis=0))
    np.save(out_dir / "z.npy", z_all)
    np.save(out_dir / "pos.npy", pos_all)
    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "offsets.npy", offsets)

    with (out_dir / "smiles.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(smiles_list))

    meta = {
        "count": kept,
        "image_size": args.image_size,
        "fingerprint_bits": args.fingerprint_bits,
        "fingerprints_packed": True,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"wrote {kept} molecules to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
