"""Equivalence and invariance utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EquivalenceResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    within_tolerance: bool


def bootstrap_equivalence(
    a: np.ndarray,
    b: np.ndarray,
    tolerance: float,
    samples: int = 2000,
    seed: int = 0,
) -> EquivalenceResult:
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size != b.size:
        raise ValueError("Paired samples must be same length.")
    diff = a - b
    if diff.size == 0:
        return EquivalenceResult(float("nan"), float("nan"), float("nan"), False)
    means = []
    for _ in range(samples):
        resample = rng.choice(diff, size=diff.size, replace=True)
        means.append(np.mean(resample))
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    mean_diff = float(np.mean(diff))
    within = ci_low >= -tolerance and ci_high <= tolerance
    return EquivalenceResult(mean_diff, float(ci_low), float(ci_high), within)
