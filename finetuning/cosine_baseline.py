#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosine baseline on frozen features.")
    parser.add_argument("--features", required=True, help="Path to extracted features .npz")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["avg", "linear"], default="avg")
    parser.add_argument("--experiment-name", default="cosine_baseline")
    return parser.parse_args()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(((x - y) ** 2).mean()))


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    ss_res = float(((y - x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _make_folds(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return np.array_split(indices, k)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=1)


def _fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    weights, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
    return weights


def _predict_linear(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def main() -> int:
    args = parse_args()
    data = np.load(args.features, allow_pickle=True)
    y = data["y"].astype(np.float32)
    tanimoto = data["tanimoto"] if "tanimoto" in data else None
    pair_ids = data["pair_ids"] if "pair_ids" in data else None

    c2d = _cosine(data["a_2d"], data["b_2d"])
    c3d = _cosine(data["a_3d"], data["b_3d"])
    c1d = _cosine(data["a_1d"], data["b_1d"])
    feats = np.stack([c2d, c3d, c1d], axis=1).astype(np.float32)

    run_dir = Path("outputs") / "finetune_runs" / args.experiment_name / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_handle.write(msg + "\n")
        log_handle.flush()

    start_time = time.time()
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    folds = _make_folds(len(y), args.folds, args.seed)
    metrics = []
    all_preds = np.zeros_like(y)

    log(f"mode={args.mode} folds={args.folds} samples={len(y)}")
    for fold_idx in range(args.folds):
        test_idx = folds[fold_idx]
        train_idx = np.hstack([folds[i] for i in range(args.folds) if i != fold_idx])

        if args.mode == "avg":
            preds = feats[test_idx].mean(axis=1)
        else:
            weights = _fit_linear(feats[train_idx], y[train_idx])
            preds = _predict_linear(feats[test_idx], weights)

        all_preds[test_idx] = preds
        fold_rmse = _rmse(preds, y[test_idx])
        fold_mae = _mae(preds, y[test_idx])
        fold_pearson = _pearson(preds, y[test_idx])
        fold_spearman = _spearman(preds, y[test_idx])
        fold_r2 = _r2(preds, y[test_idx])
        metrics.append(
            {
                "fold": fold_idx + 1,
                "rmse": fold_rmse,
                "mae": fold_mae,
                "pearson": fold_pearson,
                "spearman": fold_spearman,
                "r2": fold_r2,
                "test_size": int(test_idx.size),
                "train_size": int(train_idx.size),
            }
        )
        log(
            f"fold {fold_idx + 1}/{args.folds} "
            f"rmse {fold_rmse:.4f} pearson {fold_pearson:.4f}"
        )

    summary = {
        "rmse_mean": float(np.mean([m["rmse"] for m in metrics])),
        "rmse_std": float(np.std([m["rmse"] for m in metrics])),
        "mae_mean": float(np.mean([m["mae"] for m in metrics])),
        "mae_std": float(np.std([m["mae"] for m in metrics])),
        "pearson_mean": float(np.mean([m["pearson"] for m in metrics])),
        "pearson_std": float(np.std([m["pearson"] for m in metrics])),
        "spearman_mean": float(np.mean([m["spearman"] for m in metrics])),
        "spearman_std": float(np.std([m["spearman"] for m in metrics])),
        "r2_mean": float(np.mean([m["r2"] for m in metrics])),
        "r2_std": float(np.std([m["r2"] for m in metrics])),
        "overall_rmse": _rmse(all_preds, y),
        "overall_mae": _mae(all_preds, y),
        "overall_pearson": _pearson(all_preds, y),
        "overall_spearman": _spearman(all_preds, y),
        "overall_r2": _r2(all_preds, y),
        "runtime_sec": round(time.time() - start_time, 2),
    }
    if tanimoto is not None:
        mask = np.isfinite(tanimoto)
        if mask.any():
            summary["tanimoto_pearson"] = _pearson(tanimoto[mask], y[mask])
            summary["tanimoto_spearman"] = _spearman(tanimoto[mask], y[mask])

    with (run_dir / "fold_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    pred_path = run_dir / "predictions.csv"
    with pred_path.open("w", encoding="utf-8") as handle:
        if pair_ids is not None:
            handle.write("index,pair_id,pred,y,cos2d,cos3d,cos1d\n")
            for i, pred in enumerate(all_preds):
                handle.write(
                    f"{i},{pair_ids[i]},{pred:.6f},{y[i]:.6f},"
                    f"{c2d[i]:.6f},{c3d[i]:.6f},{c1d[i]:.6f}\n"
                )
        else:
            handle.write("index,pred,y,cos2d,cos3d,cos1d\n")
            for i, pred in enumerate(all_preds):
                handle.write(
                    f"{i},{pred:.6f},{y[i]:.6f},"
                    f"{c2d[i]:.6f},{c3d[i]:.6f},{c1d[i]:.6f}\n"
                )

    log(f"wrote results to {run_dir}")
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
