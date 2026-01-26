#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate finetuning results.")
    parser.add_argument("--runs-root", default="outputs/finetune_runs")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_runs(root: Path) -> List[Dict]:
    rows = []
    for summary_path in root.rglob("summary.json"):
        run_dir = summary_path.parent
        if run_dir.name.startswith("."):
            continue
        if run_dir.parent == root:
            exp_name = "legacy"
        else:
            exp_name = run_dir.parent.name
        run_id = run_dir.name
        summary = _load_json(summary_path)
        run_cfg_path = run_dir / "run_config.json"
        run_cfg = _load_json(run_cfg_path) if run_cfg_path.exists() else {}
        row = {
            "experiment": exp_name,
            "run_id": run_id,
            "pearson_mean": _safe_float(summary.get("pearson_mean")),
            "overall_pearson": _safe_float(summary.get("overall_pearson")),
            "tanimoto_pearson": _safe_float(summary.get("tanimoto_pearson")),
            "rmse_mean": _safe_float(summary.get("rmse_mean")),
            "mae_mean": _safe_float(summary.get("mae_mean")),
            "spearman_mean": _safe_float(summary.get("spearman_mean")),
            "r2_mean": _safe_float(summary.get("r2_mean")),
            "runtime_sec": _safe_float(summary.get("runtime_sec")),
            "mode": run_cfg.get("mode"),
        }
        if math.isfinite(row["pearson_mean"]) and math.isfinite(row["tanimoto_pearson"]):
            row["delta_pearson"] = row["pearson_mean"] - row["tanimoto_pearson"]
        else:
            row["delta_pearson"] = float("nan")
        rows.append(row)
    return rows


def _write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def _write_markdown(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        return
    headers = [
        "experiment",
        "run_id",
        "pearson_mean",
        "overall_pearson",
        "tanimoto_pearson",
        "delta_pearson",
        "rmse_mean",
        "mae_mean",
    ]
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            vals = []
            for key in headers:
                val = row.get(key, "")
                if isinstance(val, float):
                    vals.append(f"{val:.6f}" if math.isfinite(val) else "")
                else:
                    vals.append(str(val))
            handle.write("| " + " | ".join(vals) + " |\n")


def _plot_bars(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        return
    rows = [r for r in rows if math.isfinite(r["pearson_mean"])]
    rows.sort(key=lambda r: r["pearson_mean"], reverse=True)
    labels = [f"{r['experiment']}:{r['run_id']}" for r in rows]
    pearson = [r["pearson_mean"] for r in rows]
    rmse = [r["rmse_mean"] for r in rows]
    mae = [r["mae_mean"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=220, sharex=True)
    axes[0].bar(labels, pearson, color="#2b7bba")
    axes[0].set_ylabel("Pearson (mean)")
    axes[0].set_ylim(0, min(1.0, max(pearson) + 0.05))
    axes[1].bar(labels, rmse, color="#f08a5d")
    axes[1].set_ylabel("RMSE (mean)")
    axes[2].bar(labels, mae, color="#6a9a4f")
    axes[2].set_ylabel("MAE (mean)")
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    axes[2].set_xlabel("Experiment:Run")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_pearson_delta(rows: List[Dict], out_path: Path) -> None:
    rows = [r for r in rows if math.isfinite(r["pearson_mean"]) and math.isfinite(r["tanimoto_pearson"])]
    if not rows:
        return
    rows.sort(key=lambda r: r["pearson_mean"], reverse=True)
    labels = [f"{r['experiment']}:{r['run_id']}" for r in rows]
    model_p = np.array([r["pearson_mean"] for r in rows])
    tani_p = np.array([r["tanimoto_pearson"] for r in rows])
    delta = model_p - tani_p

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=220, sharex=True)
    axes[0].bar(labels, model_p, color="#3d84a8", label="Model")
    axes[0].bar(labels, tani_p, color="#f7c873", label="Tanimoto", alpha=0.8)
    axes[0].set_ylabel("Pearson")
    axes[0].legend()
    axes[1].bar(labels, delta, color="#4a7c59")
    axes[1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_ylabel("Model - Tanimoto")
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_xlabel("Experiment:Run")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_runs(root)
    if not rows:
        print("no summary.json files found")
        return 1

    rows.sort(key=lambda r: (r["experiment"], r["run_id"]))
    _write_csv(rows, out_dir / "summary_table.csv")
    _write_markdown(rows, out_dir / "summary_table.md")

    rows_sorted = sorted(rows, key=lambda r: r["pearson_mean"], reverse=True)
    _plot_bars(rows_sorted, out_dir / "metrics_overview.png")
    _plot_pearson_delta(rows_sorted, out_dir / "pearson_vs_tanimoto.png")

    print(f"wrote results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
