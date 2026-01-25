# Finetuning (Frozen Encoders + Siamese Head)

## 1) Extract features (offline)

```
python finetuning/extract_features.py \
  --csv dataset_Similarity_Prediction/original_training_set/original_training_set.csv \
  --csv dataset_Similarity_Prediction/new_dataset/new_dataset.csv \
  --weights-dir outputs/pretrain_runs/<RUN_ID> \
  --image-size 384 \
  --out outputs/finetune_features/features.npz
```

## 2) Train the Siamese head (5-fold CV)

```
python finetuning/train_siamese.py \
  --features outputs/finetune_features/features.npz \
  --epochs 75 \
  --batch-size 32 \
  --val-split 0.2 \
  --experiment-name baseline
```

### Drop 2D branch (3D + 1D only)

```
python finetuning/train_siamese.py \
  --features outputs/finetune_features/features.npz \
  --epochs 75 \
  --batch-size 32 \
  --val-split 0.2 \
  --experiment-name drop_2d \
  --drop-2d
```

### Drop gating (concat 1D + 3D)

```
python finetuning/train_siamese.py \
  --features outputs/finetune_features/features.npz \
  --epochs 75 \
  --batch-size 32 \
  --val-split 0.2 \
  --experiment-name concat_1d3d \
  --drop-2d \
  --fusion-mode concat
```

Outputs land in `outputs/finetune_runs/<experiment_name>/run_*/`.

## Cosine baseline (no neural head)

```
python finetuning/cosine_baseline.py \
  --features outputs/finetune_features/features.npz \
  --mode avg \
  --experiment-name cosine_avg
```

Use `--mode linear` to fit a tiny linear regressor on `[cos2d, cos3d, cos1d]`.
