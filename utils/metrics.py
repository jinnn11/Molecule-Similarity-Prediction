from __future__ import annotations

from typing import List, Tuple

import numpy as np


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def rankdata(a: np.ndarray) -> np.ndarray:
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


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(((x - y) ** 2).mean()))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    ss_res = float(((y - x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def make_folds(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return np.array_split(indices, k)


def split_train_val(
    indices: np.ndarray, val_split: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if val_split <= 0 or val_split >= 1:
        return indices, np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_split)))
    val_idx = shuffled[:val_size]
    train_idx = shuffled[val_size:]
    if train_idx.size == 0:
        train_idx, val_idx = val_idx, train_idx
    return train_idx, val_idx


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=1)
