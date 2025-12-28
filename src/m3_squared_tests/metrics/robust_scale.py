"""Robust scale estimators."""
from __future__ import annotations

import numpy as np


def mad(data: np.ndarray, scale: float = 1.4826) -> float:
    data = np.asarray(data)
    if data.size == 0:
        return float("nan")
    median = np.median(data)
    return scale * np.median(np.abs(data - median))


def trimmed_std(data: np.ndarray, proportion: float = 0.1) -> float:
    data = np.sort(np.asarray(data))
    if data.size == 0:
        return float("nan")
    trim = int(data.size * proportion)
    trimmed = data[trim : data.size - trim] if data.size > 2 * trim else data
    return float(np.std(trimmed, ddof=1)) if trimmed.size > 1 else 0.0
