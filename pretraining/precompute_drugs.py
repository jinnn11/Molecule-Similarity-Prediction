#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shutil
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from multiprocessing import get_context
from typing import Iterable, Optional, Tuple, Union

import numpy as np

try:
    import msgpack
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("msgpack is required to read the dataset archive.") from exc

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Draw
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:  # pragma: no cover - optional for older RDKit
        rdMolDraw2D = None
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError:  # pragma: no cover - optional for older RDKit
        GetMorganGenerator = None
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("RDKit is required for SMILES, images, and fingerprints.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute images, fingerprints, and 3D graphs.")
    parser.add_argument("--archive", required=True, help="Path to drugs_crude.msgpack.tar.gz")
    parser.add_argument("--out", required=True, help="Output directory for precomputed data")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--png-compress-level", type=int, default=1, help="0-9 (lower is faster, larger files)")
    parser.add_argument("--kekulize", action="store_true", help="Use kekulized depiction (slower)")
    parser.add_argument("--bond-line-width", type=float, default=2.0, help="Bond line width for 2D drawings")
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--max-mols", type=int, default=0, help="Optional cap for a quick smoke test")
    parser.add_argument("--max-atoms", type=int, default=100, help="Skip molecules with more atoms than this")
    default_workers = max(1, min(4, os.cpu_count() or 2))
    parser.add_argument("--num-workers", type=int, default=default_workers)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=1000, help="Entries between progress logs")
    parser.add_argument("--log-seconds", type=float, default=5.0, help="Seconds between progress logs")
    return parser.parse_args()


class _CountingReader:
    def __init__(self, handle, total_bytes: int, log_seconds: float, log_fn):
        self.handle = handle
        self.total_bytes = total_bytes
        self.log_seconds = log_seconds
        self.log_fn = log_fn
        self.bytes_read = 0
        self.last_log = time.time()

    def read(self, size: int = -1) -> bytes:
        chunk = self.handle.read(size)
        self.bytes_read += len(chunk)
        now = time.time()
        if self.log_seconds and (now - self.last_log) >= self.log_seconds:
            total_mb = self.total_bytes / (1024 * 1024) if self.total_bytes else 0.0
            read_mb = self.bytes_read / (1024 * 1024)
            percent = (100.0 * self.bytes_read / self.total_bytes) if self.total_bytes else 0.0
            self.log_fn(
                f"read {read_mb:.1f}MB/{total_mb:.1f}MB ({percent:5.1f}%)",
            )
            self.last_log = now
        return chunk


def _iter_msgpack(path: Path, log_seconds: float, log_fn) -> Iterable[Union[dict, Tuple[str, dict]]]:
    total_bytes = path.stat().st_size
    with path.open("rb") as handle:
        reader = _CountingReader(handle, total_bytes, log_seconds, log_fn)
        unpacker = msgpack.Unpacker(reader, raw=False)
        for obj in unpacker:
            if isinstance(obj, dict):
                # Handle top-level dict: {smiles: data}
                first_key = next(iter(obj.keys()), None)
                first_val = next(iter(obj.values()), None)
                if isinstance(first_key, str) and isinstance(first_val, dict):
                    for smi, data in obj.items():
                        if isinstance(data, dict):
                            yield (smi, data)
                else:
                    yield obj
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item


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


def _extract_entry(entry: Union[dict, Tuple[str, dict]]) -> Tuple[Optional[str], Optional[dict]]:
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str) and isinstance(entry[1], dict):
        return entry[0], entry[1]
    if isinstance(entry, dict):
        return entry.get("smiles"), entry
    return None, None


def _pick_conformer(entry: dict) -> Optional[dict]:
    conformers = entry.get("conformers") or []
    if not conformers:
        return None
    has_energy = any("totalenergy" in c or "energy" in c for c in conformers)
    has_weight = any("boltzmannweight" in c or "weight" in c for c in conformers)
    if has_energy:
        def conf_energy(c):
            return c.get("totalenergy", c.get("energy", 0.0))
        return min(conformers, key=conf_energy)
    if has_weight:
        def conf_weight(c):
            return c.get("boltzmannweight", c.get("weight", 0.0))
        return max(conformers, key=conf_weight)
    return conformers[0]


def _atoms_and_xyz(entry: dict) -> Tuple[Optional[list], Optional[np.ndarray]]:
    atoms = entry.get("atoms") or entry.get("atom_numbers") or entry.get("atomic_numbers")
    if atoms and isinstance(atoms[0], dict):
        atoms = [a.get("atomic_num") or a.get("atomic_number") or a.get("z") for a in atoms]
    if atoms and isinstance(atoms[0], str):
        table = Chem.GetPeriodicTable()
        atoms = [table.GetAtomicNumber(a) for a in atoms]

    conf = _pick_conformer(entry)
    if conf is not None:
        xyz = conf.get("xyz") or conf.get("coords") or conf.get("positions")
    else:
        xyz = entry.get("xyz") or entry.get("coords") or entry.get("positions")
    if xyz is None:
        return atoms, None

    if isinstance(xyz[0], dict):
        coords = [[p.get("x"), p.get("y"), p.get("z")] for p in xyz]
        return atoms, np.asarray(coords, dtype=np.float32)

    first = xyz[0]
    if isinstance(first, (list, tuple)) and len(first) >= 4:
        symbol_or_z = first[0]
        if atoms is None:
            if isinstance(symbol_or_z, str):
                table = Chem.GetPeriodicTable()
                atoms = [table.GetAtomicNumber(row[0]) for row in xyz]
            elif isinstance(symbol_or_z, (int, float)):
                atoms = [int(row[0]) for row in xyz]
        if isinstance(symbol_or_z, (str, int, float)):
            coords = [row[1:4] for row in xyz]
            return atoms, np.asarray(coords, dtype=np.float32)

    coords = np.asarray(xyz, dtype=np.float32)
    return atoms, coords


_MORGAN_GEN = {}


def _fingerprint(mol: "Chem.Mol", bits: int) -> np.ndarray:
    if GetMorganGenerator is not None:
        gen = _MORGAN_GEN.get(bits)
        if gen is None:
            gen = GetMorganGenerator(radius=2, fpSize=bits)
            _MORGAN_GEN[bits] = gen
        fp = gen.GetFingerprint(mol)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
    arr = np.zeros((bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _render_png(
    mol: "Chem.Mol",
    image_path: Path,
    *,
    image_size: int,
    kekulize: bool,
    bond_line_width: float,
    png_compress_level: int,
) -> None:
    from PIL import Image

    if rdMolDraw2D is None or not hasattr(rdMolDraw2D, "MolDraw2DCairo"):
        img = Draw.MolToImage(mol, size=(image_size, image_size), kekulize=kekulize)
        img.save(
            image_path,
            format="PNG",
            compress_level=png_compress_level,
            optimize=False,
        )
        return

    mol_to_draw = Chem.Mol(mol)
    if kekulize:
        try:
            Chem.Kekulize(mol_to_draw, clearAromaticFlags=True)
        except Exception:
            mol_to_draw = Chem.Mol(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(image_size, image_size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = bond_line_width
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol_to_draw)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()

    if png_compress_level == 1:
        image_path.write_bytes(png)
    else:
        img = Image.open(io.BytesIO(png))
        img.save(
            image_path,
            format="PNG",
            compress_level=png_compress_level,
            optimize=False,
        )


def _process_entry(
    entry: Union[dict, Tuple[str, dict]],
    image_path: Path,
    *,
    image_size: int,
    fingerprint_bits: int,
    max_atoms: int,
    png_compress_level: int,
    kekulize: bool,
    bond_line_width: float,
) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    smiles, data = _extract_entry(entry)
    if smiles is None or data is None:
        return False, "entry_format", None

    atoms, xyz = _atoms_and_xyz(data)
    if atoms is None:
        return False, "missing_atoms", None
    if xyz is None:
        return False, "missing_xyz", None
    if len(atoms) != len(xyz):
        return False, "length_mismatch", None
    if max_atoms and len(atoms) > max_atoms:
        return False, "too_many_atoms", None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "mol_parse_fail", None

    fp = _fingerprint(mol, fingerprint_bits)
    fp_packed = np.packbits(fp)

    _render_png(
        mol,
        image_path,
        image_size=image_size,
        kekulize=kekulize,
        bond_line_width=bond_line_width,
        png_compress_level=png_compress_level,
    )

    return True, "ok", (
        smiles,
        fp_packed,
        np.asarray(atoms, dtype=np.int64),
        np.asarray(xyz, dtype=np.float32),
    )


_WORKER_CFG = {}


def _init_worker(cfg: dict) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _process_entry_worker(args: Tuple[int, dict]) -> Tuple[int, bool, str, Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray]], str]:
    idx, entry = args
    cfg = _WORKER_CFG
    tmp_path = Path(cfg["images_tmp"]) / f"{idx:08d}.png"
    ok, reason, payload = _process_entry(
        entry,
        tmp_path,
        image_size=cfg["image_size"],
        fingerprint_bits=cfg["fingerprint_bits"],
        max_atoms=cfg["max_atoms"],
        png_compress_level=cfg["png_compress_level"],
        kekulize=cfg["kekulize"],
        bond_line_width=cfg["bond_line_width"],
    )
    return idx, ok, reason, payload, str(tmp_path)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images_tmp = images_dir / "_tmp"
    if images_tmp.exists():
        shutil.rmtree(images_tmp)
    images_tmp.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    info_dir = repo_root / "outputs" / "precompute_info"
    info_dir.mkdir(parents=True, exist_ok=True)
    log_path = info_dir / "progress.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_handle.write(msg + "\n")
        log_handle.flush()

    msgpack_path = _resolve_msgpack(Path(args.archive), out_dir / "raw")
    size_mb = msgpack_path.stat().st_size / (1024 * 1024)
    log(f"starting precompute on {msgpack_path} ({size_mb:.1f}MB)")

    fingerprints = []
    z_list = []
    pos_list = []
    smiles_list = []

    start_time = time.time()
    last_log = start_time
    kept = 0
    seen = 0
    skipped = 0
    skip_counts: Counter[str] = Counter()
    if args.num_workers > 0:
        cfg = {
            "image_size": args.image_size,
            "fingerprint_bits": args.fingerprint_bits,
            "max_atoms": args.max_atoms,
            "png_compress_level": args.png_compress_level,
            "kekulize": args.kekulize,
            "bond_line_width": args.bond_line_width,
            "images_tmp": str(images_tmp),
        }
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(cfg,),
        ) as pool:
            iterator = pool.imap_unordered(
                _process_entry_worker,
                enumerate(_iter_msgpack(msgpack_path, args.log_seconds, log)),
                chunksize=args.chunksize,
            )
            for result in iterator:
                seen += 1
                idx, ok, reason, payload, tmp_path = result
                if not ok or payload is None:
                    skipped += 1
                    skip_counts[reason] += 1
                else:
                    smiles, fp_packed, atoms, xyz = payload
                    final_path = images_dir / f"{kept:06d}.png"
                    os.replace(tmp_path, final_path)
                    fingerprints.append(fp_packed)
                    z_list.append(atoms)
                    pos_list.append(xyz)
                    smiles_list.append(smiles)
                    kept += 1

                if args.max_mols and kept >= args.max_mols:
                    pool.terminate()
                    break

                now = time.time()
                if args.log_interval and (seen % args.log_interval == 0):
                    elapsed = now - start_time
                    rate = seen / elapsed if elapsed > 0 else 0.0
                    if args.max_mols:
                        percent = 100.0 * kept / args.max_mols
                        remaining = max(args.max_mols - kept, 0)
                        eta = remaining / rate if rate > 0 else 0.0
                        log(
                            f"seen {seen} kept {kept} skipped {skipped} "
                            f"{percent:5.1f}% rate {rate:.1f}/s "
                            f"elapsed {elapsed/60:.1f}m eta {eta/60:.1f}m"
                        )
                    else:
                        log(
                            f"seen {seen} kept {kept} skipped {skipped} "
                            f"rate {rate:.1f}/s elapsed {elapsed/60:.1f}m"
                        )
                    last_log = now
                elif args.log_seconds and (now - last_log) >= args.log_seconds:
                    elapsed = now - start_time
                    rate = seen / elapsed if elapsed > 0 else 0.0
                    log(
                        f"seen {seen} kept {kept} skipped {skipped} "
                        f"rate {rate:.1f}/s elapsed {elapsed/60:.1f}m"
                    )
                    last_log = now
    else:
        for entry in _iter_msgpack(msgpack_path, args.log_seconds, log):
            if args.max_mols and kept >= args.max_mols:
                break

            seen += 1
            ok, reason, payload = _process_entry(
                entry,
                images_dir / f"{kept:06d}.png",
                image_size=args.image_size,
                fingerprint_bits=args.fingerprint_bits,
                max_atoms=args.max_atoms,
                png_compress_level=args.png_compress_level,
                kekulize=args.kekulize,
                bond_line_width=args.bond_line_width,
            )
            if not ok or payload is None:
                skipped += 1
                skip_counts[reason] += 1
            else:
                smiles, fp_packed, atoms, xyz = payload
                fingerprints.append(fp_packed)
                z_list.append(atoms)
                pos_list.append(xyz)
                smiles_list.append(smiles)
                kept += 1

            now = time.time()
            if args.log_interval and (seen % args.log_interval == 0):
                elapsed = now - start_time
                rate = seen / elapsed if elapsed > 0 else 0.0
                if args.max_mols:
                    percent = 100.0 * kept / args.max_mols
                    remaining = max(args.max_mols - kept, 0)
                    eta = remaining / rate if rate > 0 else 0.0
                    log(
                        f"seen {seen} kept {kept} skipped {skipped} "
                        f"{percent:5.1f}% rate {rate:.1f}/s "
                        f"elapsed {elapsed/60:.1f}m eta {eta/60:.1f}m"
                    )
                else:
                    log(
                        f"seen {seen} kept {kept} skipped {skipped} "
                        f"rate {rate:.1f}/s elapsed {elapsed/60:.1f}m"
                    )
                last_log = now
            elif args.log_seconds and (now - last_log) >= args.log_seconds:
                elapsed = now - start_time
                rate = seen / elapsed if elapsed > 0 else 0.0
                log(
                    f"seen {seen} kept {kept} skipped {skipped} "
                    f"rate {rate:.1f}/s elapsed {elapsed/60:.1f}m"
                )
                last_log = now

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
        "bond_line_width": args.bond_line_width,
        "png_compress_level": args.png_compress_level,
        "kekulize": args.kekulize,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    shutil.rmtree(images_tmp, ignore_errors=True)

    elapsed = time.time() - start_time
    rate = seen / elapsed if elapsed > 0 else 0.0
    log(
        f"wrote {kept} molecules to {out_dir} "
        f"(rate {rate:.1f}/s, elapsed {elapsed/60:.1f}m)"
    )

    run_info = {
        "archive": str(msgpack_path),
        "archive_size_mb": round(size_mb, 2),
        "output_dir": str(out_dir),
        "args": vars(args),
        "counts": {
            "seen": seen,
            "kept": kept,
            "skipped": skipped,
        },
        "skip_reasons": dict(skip_counts),
        "timing": {
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_time)),
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "elapsed_seconds": round(elapsed, 2),
            "rate_per_sec": round(rate, 2),
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    with (info_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2)
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
