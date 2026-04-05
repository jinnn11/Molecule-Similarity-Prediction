import numpy as np

from utils.metrics import cosine, mae, make_folds, pearson, r2, rmse, spearman, split_train_val


def test_pearson_and_spearman_perfect_match():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    assert pearson(x, y) == 1.0
    assert spearman(x, y) == 1.0


def test_error_metrics_are_zero_for_identical_predictions():
    x = np.array([0.2, 0.4, 0.6], dtype=np.float32)

    assert rmse(x, x) == 0.0
    assert mae(x, x) == 0.0
    assert r2(x, x) == 1.0


def test_cosine_returns_expected_similarity():
    a = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [1.0, -1.0]], dtype=np.float32)

    out = cosine(a, b)

    assert np.allclose(out, np.array([1.0, 0.0], dtype=np.float32))


def test_make_folds_covers_all_indices_once():
    folds = make_folds(n=10, k=3, seed=7)
    merged = np.concatenate(folds)

    assert sorted(merged.tolist()) == list(range(10))
    assert sum(len(fold) for fold in folds) == 10


def test_split_train_val_respects_requested_split():
    indices = np.arange(10)
    train_idx, val_idx = split_train_val(indices, val_split=0.2, seed=7)

    assert len(train_idx) == 8
    assert len(val_idx) == 2
    assert set(train_idx).isdisjoint(set(val_idx))
    assert sorted(np.concatenate([train_idx, val_idx]).tolist()) == list(range(10))
