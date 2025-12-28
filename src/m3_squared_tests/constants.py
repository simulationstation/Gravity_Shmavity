"""Shared constants for M3 squared-coupling tests."""
from __future__ import annotations

import math

ALPHA2 = 5.325e-5
ALPHA = math.sqrt(ALPHA2)
PPM = 1e6


def fraction_to_ppm(value: float) -> float:
    return value * PPM


def ppm_to_fraction(value_ppm: float) -> float:
    return value_ppm / PPM


def alpha2_ppm() -> float:
    return fraction_to_ppm(ALPHA2)


def alpha4() -> float:
    return ALPHA2 * ALPHA2
