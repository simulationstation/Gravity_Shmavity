"""Model selection utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ModelResult:
    k: int
    bic: float
    loglik: float
    a0: float | None = None


def select_best(results: List[ModelResult]) -> ModelResult:
    return min(results, key=lambda r: r.bic)


def delta_bic(results: List[ModelResult]) -> List[float]:
    base = results[0].bic
    return [r.bic - base for r in results]


def stop_rule(results: List[ModelResult], threshold: float) -> int:
    """Return max k supported by threshold (smaller BIC is better)."""
    best_k = results[0].k
    for prev, curr in zip(results[:-1], results[1:]):
        improvement = prev.bic - curr.bic
        if improvement <= threshold:
            break
        best_k = curr.k
    return best_k
