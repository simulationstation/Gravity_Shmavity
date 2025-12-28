"""Latent processes for layered models."""
from __future__ import annotations

import numpy as np


def simulate_linear_latent(phi: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Linear random effect X = phi @ w with w ~ N(0, sigma^2 I)."""
    w = rng.normal(scale=sigma, size=phi.shape[1])
    return phi @ w


def simulate_ou_process(t: np.ndarray, tau: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate an OU process with Euler-Maruyama discretization."""
    dt = np.diff(t, prepend=t[0])
    x = np.zeros_like(t)
    for i in range(1, len(t)):
        decay = np.exp(-dt[i] / tau)
        noise_scale = sigma * np.sqrt(1 - decay**2)
        x[i] = decay * x[i - 1] + noise_scale * rng.normal()
    return x
