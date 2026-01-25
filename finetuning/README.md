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
  --val-split 0.2
```

Outputs land in `outputs/finetune_runs/run_*/`.
