"""Model selection metrics."""
from __future__ import annotations

import math


def bic(n: int, sse: float, k: int) -> float:
    if n <= 0:
        return float("nan")
    return n * math.log(max(sse / n, 1e-12)) + k * math.log(n)


def aic(n: int, sse: float, k: int) -> float:
    if n <= 0:
        return float("nan")
    return n * math.log(max(sse / n, 1e-12)) + 2 * k
