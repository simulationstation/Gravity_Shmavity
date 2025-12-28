"""Regression utilities for model comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FitResult:
    coef: np.ndarray
    intercept: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + x @ self.coef


def _design_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x


def fit_linear(x: np.ndarray, y: np.ndarray) -> FitResult:
    x_mat = _design_matrix(x)
    x_aug = np.column_stack([np.ones(x_mat.shape[0]), x_mat])
    coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    return FitResult(coef=coef[1:], intercept=float(coef[0]))


def cv_mse(
    x: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray], FitResult],
    folds: int = 5,
) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(y)
    if n < 3:
        return float("nan")
    folds = min(folds, n)
    indices = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(indices)
    split = np.array_split(indices, folds)
    errors = []
    for i in range(folds):
        test_idx = split[i]
        train_idx = np.concatenate([split[j] for j in range(folds) if j != i])
        model = fit_fn(x[train_idx], y[train_idx])
        pred = model.predict(_design_matrix(x[test_idx]))
        errors.append(np.mean((y[test_idx] - pred) ** 2))
    return float(np.mean(errors))
