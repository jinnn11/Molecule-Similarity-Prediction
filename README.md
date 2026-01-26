# Molecule Similarity Prediction

A multi-modal deep learning framework that predicts molecular similarity (0–1) with ~0.92 correlation to human expert scores, outperforming the industry-standard Tanimoto metric. The best downstream model is `cosine_mlp`, which regresses the human similarity score from cosine similarities of frozen 1D/2D/3D embeddings.

Tri-modal pretraining (2D images, 3D conformers, 1D fingerprints) followed by a frozen-encoder similarity regression pipeline. The downstream stage uses cosine similarities across modalities and a tiny regression head to predict a human similarity score in the 0–1 range.

## Architecture

### Pretraining
![Pretraining diagram](diagram/diagram_pretraining.png)

### Similarity Regression (Frozen Encoders)
![Similarity Regression](diagram/diagram_similarity_boxed.png)

## Highlights
- Tri-tower contrastive pretraining (ResNet-18, SchNet, MLP)
- Frozen-encoder downstream evaluation with cosine similarities
- Small-data friendly (200 pairs) with cross-validation
- Experiment tracking with plots, logs, and JSON summaries

## Data
- Pretraining: GEOM-Drugs (drugs_crude.msgpack) with one conformer per molecule and atom-count filtering.
- Downstream: 200 labeled similarity pairs in `dataset_Similarity_Prediction/`.

## Setup

Recommended Python: 3.10

```
conda create -n simpred python=3.10
conda activate simpred
pip install -r requirements.txt
```

## Precompute (Pretraining Inputs)

```
python pretraining/precompute_drugs.py \
  --archive /path/to/drugs_crude.msgpack \
  --out pretraining_data/drugs \
  --max-atoms 100 \
  --image-size 384
```

## Pretraining

```
python pretraining/train_pretrain.py \
  --data-dir pretraining_data/drugs \
  --image-size 384 \
  --batch-size 64 \
  --accumulation-steps 4 \
  --epochs 100 \
  --device cuda
```

Optional: evaluate cosine signal during pretraining

```
python pretraining/train_pretrain.py \
  --data-dir pretraining_data/drugs \
  --image-size 384 \
  --batch-size 64 \
  --accumulation-steps 4 \
  --epochs 100 \
  --device cuda \
  --eval-csv dataset_Similarity_Prediction/original_training_set/original_training_set.csv \
  --eval-csv dataset_Similarity_Prediction/new_dataset/new_dataset.csv \
  --eval-every 5
```

Outputs land in `outputs/pretrain_runs/run_*/`.

## Feature Extraction (Frozen Encoders)

```
python finetuning/extract_features.py \
  --csv dataset_Similarity_Prediction/original_training_set/original_training_set.csv \
  --csv dataset_Similarity_Prediction/new_dataset/new_dataset.csv \
  --weights-dir outputs/pretrain_runs/<RUN_ID> \
  --image-size 384 \
  --out outputs/finetune_features/features.npz
```

## Similarity Regression (Cosine Baselines)

Average cosine:

```
python finetuning/cosine_baseline.py \
  --features outputs/finetune_features/features.npz \
  --mode avg \
  --experiment-name cosine_avg
```

Linear cosine with Tanimoto:

```
python finetuning/cosine_baseline.py \
  --features outputs/finetune_features/features.npz \
  --mode linear \
  --include-tanimoto \
  --experiment-name cosine_linear_tani
```

All results are saved under `outputs/finetune_runs/<experiment_name>/run_*/`.

## Aggregate Results

```
python results/aggregate_results.py
```

Produces:
- `results/summary_table.csv`
- `results/summary_table.md`
- `results/metrics_overview.png`
- `results/pearson_vs_tanimoto.png`

## Repository Layout

- `pretraining/`: precompute + contrastive training
- `finetuning/`: feature extraction + similarity regression
- `results/`: aggregated result tables and plots
- `diagram/`: architecture diagrams
- `outputs/`: training artifacts (plots, logs, metrics)

## Reproducibility

- Most scripts accept `--seed`.
- Run configs and metrics are saved alongside each run.

## Notes

- The downstream stage is best described as **feature extraction + similarity regression** because encoders are frozen.
- If you unfreeze encoders, it becomes true finetuning.

## Conclusions

- Frozen tri-modal embeddings are highly aligned with human similarity: cosine correlations are strong per modality, indicating effective pretraining.
- Simple cosine-based fusion outperforms complex heads; attention/Siamese heads underperform on 200 pairs due to overfitting.
- The best downstream model is a tiny regressor on cosine similarities (`cosine_mlp`, Pearson ≈0.92).
- Tanimoto remains a strong baseline, but tri-modal cosine fusion exceeds it (≈0.92 vs 0.85 Pearson).
- Adding Tanimoto as a feature improves error metrics, showing complementarity between learned embeddings and classical similarity.
- Model complexity must match data scale; low-parameter regressors generalize better on small expert-labeled datasets.
