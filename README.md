# Molecule Similarity Prediction

A multi-modal deep learning framework that predicts molecular similarity (0–1) with ~0.92 correlation to human expert scores, outperforming the industry-standard Tanimoto metric. The best downstream model is `cosine_mlp`, which regresses the human similarity score from cosine similarities of frozen 1D/2D/3D embeddings.

Tri-modal pretraining (2D images, 3D conformers, 1D fingerprints) followed by a frozen-encoder similarity regression pipeline. The downstream stage uses cosine similarities across modalities and a tiny regression head to predict a human similarity score in the 0–1 range.

## Architecture

### Pretraining
![Pretraining diagram](diagram/diagram_pretraining.png)

### Similarity Regression (Frozen Encoders)
![Similarity Regression](diagram/diagram_similarity_boxed.png)

The final prediction head is a lightweight MLP operating on cosine similarity outputs from the frozen encoders:

- Inputs: cosine similarities computed from frozen 2D/3D/1D embeddings.
- Architecture: `Linear(3 → 16) → ReLU → Linear(16 → 1)` (regression, no sigmoid).

## Highlights
- Tri-tower contrastive pretraining (ResNet-18, SchNet, MLP)
- Frozen-encoder downstream evaluation with cosine similarities
- Small-data friendly (200 pairs) with cross-validation
- Experiment tracking with plots, logs, and JSON summaries

## Data
- Pretraining: GEOM-Drugs (drugs_crude.msgpack) with one conformer per molecule and atom-count filtering.
- Downstream: 200 labeled similarity pairs in `dataset_Similarity_Prediction/`.
- The raw datasets and generated run artifacts are intentionally not versioned in this portfolio snapshot.

## Setup

Recommended Python: 3.10

```
conda env create -f environment.yml
conda activate simpred
```

Alternative:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Validation

```bash
python3 -m pytest
python3 finetuning/cosine_baseline.py --help
python3 results/aggregate_results.py --help
```

## Reproducing The Project

You will need to supply the two local data sources below before running the full pipeline:

- GEOM-Drugs archive for pretraining, passed to `pretraining/precompute_drugs.py --archive ...`
- The expert-labeled downstream CSVs and associated 3D conformers under `dataset_Similarity_Prediction/`

The repository keeps the source code, diagrams, and curated summary results in `results/`, but does not commit raw datasets, precomputed tensors, or fold-by-fold training artifacts.

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
- `tests/`: lightweight validation for utility code
- `outputs/`: local run artifacts, ignored by git

## Reproducibility

- Most scripts accept `--seed`.
- Run configs and metrics are saved alongside each run.

## Notes

- The downstream stage is best described as **feature extraction + similarity regression** because encoders are frozen.
- If you unfreeze encoders, it becomes true finetuning.

## Limitations

- **Small evaluation set.** The downstream dataset contains only 200 expert-labeled pairs. With 5-fold CV each test fold has ~40 samples, so metric estimates carry high variance. No bootstrap confidence intervals or paired significance tests are reported.
- **Molecule-level leakage not controlled.** CV splits are by pair, not by molecule. The same molecule can appear in both train and test folds (in different pairs), which may inflate correlation estimates.
- **Tanimoto comparison is not fully calibrated.** TanimotoCombo is used as a raw score whereas the model's downstream heads are trained regressors. A fairer baseline would be a linear regression on Tanimoto alone; the raw-score comparison (`cosine_avg` 0.90 vs Tanimoto 0.85) is the most apples-to-apples.

## Conclusions

- Frozen tri-modal embeddings are highly aligned with human similarity: cosine correlations are strong per modality, indicating effective pretraining.
- Simple cosine-based fusion outperforms complex heads; attention/Siamese heads underperform on 200 pairs due to overfitting.
- The best downstream model is a tiny regressor on cosine similarities (`cosine_mlp`, Pearson ≈0.92).
- Tanimoto remains a strong baseline, but tri-modal cosine fusion exceeds it (≈0.92 vs 0.85 Pearson).
- Adding Tanimoto as a feature improves error metrics, showing complementarity between learned embeddings and classical similarity.
- Model complexity must match data scale; low-parameter regressors generalize better on small expert-labeled datasets.
