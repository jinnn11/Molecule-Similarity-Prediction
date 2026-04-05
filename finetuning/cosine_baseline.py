#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import pearson, spearman, rmse, mae, r2, make_folds, cosine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosine baseline on frozen features.")
    parser.add_argument("--features", required=True, help="Path to extracted features .npz")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mode",
        choices=["avg", "linear", "ridge", "mlp", "xgb"],
        default="avg",
    )
    parser.add_argument("--include-tanimoto", action="store_true")
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--mlp-hidden", type=int, default=16)
    parser.add_argument("--mlp-epochs", type=int, default=200)
    parser.add_argument("--mlp-lr", type=float, default=1e-2)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--experiment-name", default="cosine_baseline")
    return parser.parse_args()


def _fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    weights, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
    return weights


def _predict_linear(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    n_features = x_aug.shape[1]
    reg = np.eye(n_features, dtype=x.dtype) * alpha
    reg[-1, -1] = 0.0
    xtx = x_aug.T @ x_aug + reg
    xty = x_aug.T @ y
    weights = np.linalg.solve(xtx, xty)
    return weights


def _zscore(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def _fit_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).float(),
    )
    loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(test_x).float()).squeeze(-1).cpu().numpy()
    return preds


def main() -> int:
    args = parse_args()
    data = np.load(args.features, allow_pickle=True)
    y = data["y"].astype(np.float32)
    tanimoto = data["tanimoto"] if "tanimoto" in data else None
    pair_ids = data["pair_ids"] if "pair_ids" in data else None

    c2d = cosine(data["a_2d"], data["b_2d"])
    c3d = cosine(data["a_3d"], data["b_3d"])
    c1d = cosine(data["a_1d"], data["b_1d"])
    feats = np.stack([c2d, c3d, c1d], axis=1).astype(np.float32)
    if args.include_tanimoto:
        if tanimoto is None:
            raise ValueError("TanimotoCombo missing from features file.")
        tan = np.asarray(tanimoto, dtype=np.float32)
        feats = np.concatenate([feats, tan.reshape(-1, 1)], axis=1)

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

    folds = make_folds(len(y), args.folds, args.seed)
    metrics = []
    all_preds = np.zeros_like(y)

    log(f"mode={args.mode} folds={args.folds} samples={len(y)}")
    for fold_idx in range(args.folds):
        test_idx = folds[fold_idx]
        train_idx = np.hstack([folds[i] for i in range(args.folds) if i != fold_idx])

        train_x = feats[train_idx]
        test_x = feats[test_idx]
        train_y = y[train_idx]
        test_y = y[test_idx]

        if args.include_tanimoto:
            tan_train = train_x[:, -1]
            if np.isnan(tan_train).any():
                fill = np.nanmean(tan_train)
                train_x[:, -1] = np.where(np.isnan(tan_train), fill, tan_train)
            tan_test = test_x[:, -1]
            if np.isnan(tan_test).any():
                fill = np.nanmean(train_x[:, -1])
                test_x[:, -1] = np.where(np.isnan(tan_test), fill, tan_test)

        if args.zscore:
            train_x, test_x = _zscore(train_x, test_x)

        if args.mode == "avg":
            preds = test_x.mean(axis=1)
        elif args.mode == "linear":
            weights = _fit_linear(train_x, train_y)
            preds = _predict_linear(test_x, weights)
        elif args.mode == "ridge":
            weights = _fit_ridge(train_x, train_y, args.ridge_alpha)
            preds = _predict_linear(test_x, weights)
        elif args.mode == "mlp":
            preds = _fit_mlp(
                train_x,
                train_y,
                test_x,
                hidden=args.mlp_hidden,
                epochs=args.mlp_epochs,
                lr=args.mlp_lr,
                weight_decay=args.mlp_weight_decay,
            )
        else:
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise ImportError(
                    "xgboost is required for --mode xgb. Install with `pip install xgboost`."
                ) from exc
            model = xgb.XGBRegressor(
                n_estimators=args.xgb_n_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                objective="reg:squarederror",
                random_state=args.seed + fold_idx,
            )
            model.fit(train_x, train_y)
            preds = model.predict(test_x)

        all_preds[test_idx] = preds
        fold_rmse = rmse(preds, test_y)
        fold_mae = mae(preds, test_y)
        fold_pearson = pearson(preds, test_y)
        fold_spearman = spearman(preds, test_y)
        fold_r2 = r2(preds, test_y)
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
        "overall_rmse": rmse(all_preds, y),
        "overall_mae": mae(all_preds, y),
        "overall_pearson": pearson(all_preds, y),
        "overall_spearman": spearman(all_preds, y),
        "overall_r2": r2(all_preds, y),
        "runtime_sec": round(time.time() - start_time, 2),
    }
    if tanimoto is not None:
        mask = np.isfinite(tanimoto)
        if mask.any():
            summary["tanimoto_pearson"] = pearson(tanimoto[mask], y[mask])
            summary["tanimoto_spearman"] = spearman(tanimoto[mask], y[mask])

    with (run_dir / "fold_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    pred_path = run_dir / "predictions.csv"
    with pred_path.open("w", encoding="utf-8") as handle:
        cols = ["index", "pred", "y", "cos2d", "cos3d", "cos1d"]
        if args.include_tanimoto:
            cols.append("tanimoto")
        if pair_ids is not None:
            cols.insert(1, "pair_id")
        handle.write(",".join(cols) + "\n")
        for i, pred in enumerate(all_preds):
            row = [str(i)]
            if pair_ids is not None:
                row.append(str(pair_ids[i]))
            row.extend(
                [
                    f"{pred:.6f}",
                    f"{y[i]:.6f}",
                    f"{c2d[i]:.6f}",
                    f"{c3d[i]:.6f}",
                    f"{c1d[i]:.6f}",
                ]
            )
            if args.include_tanimoto:
                row.append(f"{float(tanimoto[i]):.6f}")
            handle.write(",".join(row) + "\n")

    log(f"wrote results to {run_dir}")
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
