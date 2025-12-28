"""Optional coherence metrics for time series."""
from __future__ import annotations

import numpy as np


def simple_coherence(values: np.ndarray) -> float:
    """Compute a bounded coherence proxy from variance ratio."""
    values = np.asarray(values)
    if values.size < 2:
        return float("nan")
    var = np.var(values, ddof=1)
    return float(1.0 / (1.0 + var))
