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

### Additional cosine experiments

```
# ridge regression (regularized linear)
python finetuning/cosine_baseline.py \
  --features outputs/finetune_features/features.npz \
  --mode ridge \
  --ridge-alpha 1.0 \
  --experiment-name cosine_ridge

# include Tanimoto as 4th feature
python finetuning/cosine_baseline.py \
  --features outputs/finetune_features/features.npz \
  --mode linear \
  --include-tanimoto \
  --experiment-name cosine_linear_tani

# z-score normalization per fold
python finetuning/cosine_baseline.py \
  --features outputs/finetune_features/features.npz \
  --mode ridge \
  --zscore \
  --experiment-name cosine_ridge_z

# tiny MLP on cosine features
python finetuning/cosine_baseline.py \
  --features outputs/finetune_features/features.npz \
  --mode mlp \
  --mlp-hidden 16 \
  --mlp-epochs 200 \
  --experiment-name cosine_mlp

# XGBoost on cosine features (requires xgboost)
python finetuning/cosine_baseline.py \
  --features outputs/finetune_features/features.npz \
  --mode xgb \
  --experiment-name cosine_xgb
```

## Light finetuning (last layers only)

```
python finetuning/finetune_last_layers.py \
  --csv dataset_Similarity_Prediction/original_training_set/original_training_set.csv \
  --csv dataset_Similarity_Prediction/new_dataset/new_dataset.csv \
  --weights-dir outputs/pretrain_runs/<RUN_ID> \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-5 \
  --experiment-name light_finetune
```

Add `--drop-2d` to skip images, and `--include-tanimoto` to add Tanimoto as a feature in the final linear head.
