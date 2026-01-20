#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
import time
from collections import Counter
from pathlib import Path
from itertools import islice
from typing import Iterable, Optional, Sequence

try:
    import msgpack
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("msgpack is required to read the dataset archive.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("matplotlib is required to save EDA figures.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for drugs msgpack dataset.")
    parser.add_argument("--archive", required=True, help="Path to drugs_featurized.msgpack or .tar.gz")
    parser.add_argument("--out", default="eda", help="Output folder for figures and summary")
    parser.add_argument("--max-mols", type=int, default=0, help="Optional cap for quick EDA")
    parser.add_argument("--log-interval", type=int, default=1000, help="Entries between progress logs")
    parser.add_argument("--log-seconds", type=float, default=5.0, help="Seconds between progress logs")
    return parser.parse_args()


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


class _CountingReader:
    def __init__(self, handle, total_bytes: int, log_seconds: float):
        self.handle = handle
        self.total_bytes = total_bytes
        self.log_seconds = log_seconds
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
            print(
                f"read {read_mb:.1f}MB/{total_mb:.1f}MB ({percent:5.1f}%)",
                flush=True,
            )
            self.last_log = now
        return chunk


def _iter_msgpack(path: Path, log_seconds: float) -> Iterable[dict]:
    total_bytes = path.stat().st_size
    with path.open("rb") as handle:
        if log_seconds:
            reader = _CountingReader(handle, total_bytes, log_seconds)
            unpacker = msgpack.Unpacker(reader, raw=False, strict_map_key=False)
        else:
            unpacker = msgpack.Unpacker(handle, raw=False, strict_map_key=False)
        for obj in unpacker:
            if isinstance(obj, dict):
                if _looks_like_smiles_map(obj):
                    for smiles, data in obj.items():
                        if isinstance(data, dict):
                            entry = dict(data)
                            entry["smiles"] = smiles
                            yield entry
                        elif isinstance(data, list):
                            yield {"smiles": smiles, "conformers": data}
                else:
                    yield obj
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item


def _looks_like_smiles_map(obj: dict) -> bool:
    # Dataset format: top-level dict mapping SMILES -> dict or list of conformers.
    if not obj:
        return False
    sample_items = list(islice(obj.items(), 5))
    if not all(isinstance(k, str) for k, _ in sample_items):
        return False
    if not all(isinstance(v, (list, dict)) for _, v in sample_items):
        return False
    return True


def _pick_atoms(entry: dict) -> Optional[Sequence[int]]:
    atoms = entry.get("atoms") or entry.get("atom_numbers") or entry.get("atomic_numbers")
    if atoms is None:
        conformers = entry.get("conformers") or []
        if conformers and isinstance(conformers[0], dict):
            atoms = (
                conformers[0].get("atoms")
                or conformers[0].get("atom_numbers")
                or conformers[0].get("atomic_numbers")
            )
    if atoms is None:
        conformers = entry.get("conformers") or []
        conf = conformers[0] if conformers else entry
        xyz = conf.get("xyz") or conf.get("coords") or conf.get("positions")
        if xyz and isinstance(xyz[0], (list, tuple)) and len(xyz[0]) >= 4:
            first = xyz[0][0]
            if isinstance(first, str):
                return [row[0] for row in xyz]
            if isinstance(first, (int, float)):
                return [int(row[0]) for row in xyz]
    if atoms and isinstance(atoms[0], dict):
        atoms = [
            a.get("atomic_num") or a.get("atomic_number") or a.get("z") or a.get("atom_type")
            for a in atoms
        ]
    return atoms


def _pick_xyz_len(entry: dict, atoms_len: Optional[int] = None) -> Optional[int]:
    conformers = entry.get("conformers") or []
    if conformers:
        conf = conformers[0]
        xyz = conf.get("xyz") or conf.get("coords") or conf.get("positions")
        if xyz is None:
            return atoms_len
    else:
        xyz = entry.get("xyz") or entry.get("coords") or entry.get("positions")
        if xyz is None:
            return atoms_len
    if xyz is None:
        return None
    return len(xyz)


class _StatCounter:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.min: Optional[int] = None
        self.max: Optional[int] = None
        self.counter: Counter[int] = Counter()

    def update(self, value: int) -> None:
        self.count += 1
        self.total += value
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value
        self.counter[value] += 1

    def mean(self) -> float:
        return (self.total / self.count) if self.count else 0.0


def _plot_hist_counts(counter: Counter[int], title: str, xlabel: str, out_path: Path) -> None:
    if not counter:
        return
    plt.figure(figsize=(8, 5))
    values = list(counter.keys())
    weights = list(counter.values())
    plt.hist(values, bins=50, weights=weights, color="#2a6f97", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_top_elements(counter: Counter, out_path: Path, top_n: int = 20) -> None:
    if not counter:
        return
    most_common = counter.most_common(top_n)
    labels = [str(k) for k, _ in most_common]
    values = [v for _, v in most_common]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values, color="#6a4c93")
    plt.title("Top Elements")
    plt.xlabel("Atomic Number")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    msgpack_path = _resolve_msgpack(Path(args.archive), out_dir / "raw")
    size_mb = msgpack_path.stat().st_size / (1024 * 1024)
    print(f"starting EDA on {msgpack_path} ({size_mb:.1f}MB)")

    atom_stats = _StatCounter()
    smiles_stats = _StatCounter()
    conformer_stats = _StatCounter()
    element_counter = Counter()
    xyz_stats = _StatCounter()
    processed = 0
    skipped = 0
    start_time = time.time()
    last_log = start_time

    for entry in _iter_msgpack(msgpack_path, args.log_seconds):
        if args.max_mols and processed >= args.max_mols:
            break

        smiles = entry.get("smiles") or entry.get("canon_smiles") or entry.get("xyz2mol_smiles")
        atoms = _pick_atoms(entry)
        if atoms is not None:
            atoms = [a for a in atoms if a is not None]
        xyz_len = _pick_xyz_len(entry, atoms_len=(len(atoms) if atoms else None))
        if smiles is None or atoms is None or xyz_len is None:
            skipped += 1
            continue
        if len(atoms) != xyz_len:
            skipped += 1
            continue

        conformers = entry.get("conformers") or []
        atom_stats.update(len(atoms))
        smiles_stats.update(len(smiles))
        xyz_stats.update(xyz_len)
        conformer_stats.update(len(conformers))
        element_counter.update(atoms)
        processed += 1

        total_seen = processed + skipped
        now = time.time()
        if args.log_interval and total_seen % args.log_interval == 0:
            elapsed = time.time() - start_time
            rate = total_seen / elapsed if elapsed > 0 else 0.0
            if args.max_mols:
                percent = 100.0 * processed / args.max_mols
                print(
                    f"seen {total_seen} (processed {processed}, skipped {skipped}) "
                    f"{percent:5.1f}% rate {rate:.1f}/s",
                    flush=True,
                )
            else:
                print(
                    f"seen {total_seen} (processed {processed}, skipped {skipped}) "
                    f"rate {rate:.1f}/s",
                    flush=True,
                )
            last_log = now
        elif args.log_seconds and (now - last_log) >= args.log_seconds:
            elapsed = now - start_time
            rate = total_seen / elapsed if elapsed > 0 else 0.0
            print(
                f"seen {total_seen} (processed {processed}, skipped {skipped}) "
                f"rate {rate:.1f}/s",
                flush=True,
            )
            last_log = now

    if not processed:
        raise RuntimeError("No molecules processed. Check the archive format.")

    summary = {
        "processed": processed,
        "skipped": skipped,
        "atoms_min": int(atom_stats.min),
        "atoms_mean": float(atom_stats.mean()),
        "atoms_max": int(atom_stats.max),
        "smiles_len_min": int(smiles_stats.min),
        "smiles_len_mean": float(smiles_stats.mean()),
        "smiles_len_max": int(smiles_stats.max),
        "conformers_min": int(conformer_stats.min),
        "conformers_mean": float(conformer_stats.mean()),
        "conformers_max": int(conformer_stats.max),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _plot_hist_counts(atom_stats.counter, "Atoms per Molecule", "Atom Count", out_dir / "atoms_per_molecule.png")
    _plot_hist_counts(smiles_stats.counter, "SMILES Length", "SMILES Length", out_dir / "smiles_length.png")
    _plot_hist_counts(
        conformer_stats.counter,
        "Conformers per Molecule",
        "Conformer Count",
        out_dir / "conformers_per_molecule.png",
    )
    _plot_hist_counts(xyz_stats.counter, "XYZ Atoms per Molecule", "Atom Count", out_dir / "xyz_atoms_per_molecule.png")
    _plot_top_elements(element_counter, out_dir / "top_elements.png")

    elapsed = time.time() - start_time
    rate = (processed + skipped) / elapsed if elapsed > 0 else 0.0
    print(f"wrote EDA outputs to {out_dir} (rate {rate:.1f}/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
