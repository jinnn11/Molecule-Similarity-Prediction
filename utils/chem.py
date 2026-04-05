from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
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
    raise ImportError("RDKit is required.") from exc

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None


def image_transform(image_size: int) -> transforms.Compose:
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


def render_image(mol: "Chem.Mol", image_size: int) -> torch.Tensor:
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
    return image_transform(image_size)(img)


_MORGAN_GEN = {}


def fingerprint(mol: "Chem.Mol", bits: int = 2048) -> torch.Tensor:
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


def load_pdb(pdb_path: Path) -> Optional[Data]:
    if not pdb_path.exists():
        return None
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    if Data is not None:
        return Data(z=z, pos=pos)
    return {"z": z, "pos": pos}


def embed_3d(smiles: str) -> Optional[Data]:
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
    if Data is not None:
        return Data(z=z, pos=pos)
    return {"z": z, "pos": pos}


def get_pdb_path(csv_path: Path, id_pair: int, side: str) -> Path:
    base = csv_path.parent / "conformers_3D"
    return base / f"best_rocs_conformer_{id_pair:03d}{side}.pdb"
