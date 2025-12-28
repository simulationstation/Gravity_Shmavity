"""Alpha-ladder coupling definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

DEFAULT_ALPHA = 1 / 137.035999084


@dataclass(frozen=True)
class Ladder:
    alpha: float = DEFAULT_ALPHA
    include_m1: bool = True

    def epsilon(self, m: int) -> float:
        if m == 2:
            return 1.0
        if m == 1:
            return self.alpha
        if m < 1:
            raise ValueError("Layer index m must be >= 1")
        k = m - 2
        if k < 1:
            raise ValueError("Layer index for m>=3 expected")
        return self.alpha ** (2 * k)

    def ladder_weights(self, max_m: int) -> Dict[int, float]:
        layers = range(1, max_m + 1)
        return {m: self.epsilon(m) for m in layers if (self.include_m1 or m != 1)}

    def ladder_sequence(self, max_m: int) -> Iterable[float]:
        return [self.epsilon(m) for m in range(1, max_m + 1) if (self.include_m1 or m != 1)]
