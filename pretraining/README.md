# Pretraining (Tri-Tower)

This folder contains a minimal tri-tower pretraining setup:
2D image tower (ResNet-18), 3D tower (SchNet), and 1D fingerprint MLP,
aligned with a CLIP-style contrastive loss.

## Precompute and train

1) Precompute images, fingerprints, and 3D graphs from the archive:

```
python pretraining/precompute_drugs.py --archive /path/to/drugs_crude.msgpack.tar.gz --out pretraining_data/drugs --max-mols 1000 --max-atoms 100
```

2) Run pretraining:

```
python pretraining/train_pretrain.py --data-dir pretraining_data/drugs --epochs 5
```

By default, pretraining runs are written to `outputs/pretrain_runs/run_*/`.

Optional: monitor downstream cosine signal during pretraining

```
python pretraining/train_pretrain.py \
  --data-dir pretraining_data/drugs \
  --epochs 5 \
  --eval-csv dataset_Similarity_Prediction/original_training_set/original_training_set.csv \
  --eval-csv dataset_Similarity_Prediction/new_dataset/new_dataset.csv \
  --eval-every 5
```

## Requirements

- PyTorch + torchvision
- torch_geometric
- RDKit
- msgpack
