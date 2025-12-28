"""Two-channel time series observable model and simulation."""
from __future__ import annotations

import numpy as np
from scipy import stats

from mverse_ladder.physics.ladder import Ladder
from mverse_ladder.physics.latent import simulate_ou_process


def simulate_timeseries(
    n: int,
    dt: float,
    ladder: Ladder,
    max_m: int,
    a0: float,
    tau_base: float,
    sigma_latent: float,
    sigma_noise: float,
    rng: np.random.Generator,
) -> dict:
    t = np.arange(n) * dt
    ch1 = np.zeros(n)
    ch2 = np.zeros(n)

    for idx, (m, eps) in enumerate(ladder.ladder_weights(max_m).items()):
        tau = tau_base * (1 + 0.5 * idx)
        x_m = simulate_ou_process(t, tau, sigma_latent, rng)
        weight = a0 * eps
        ch1 += weight * x_m
        ch2 += weight * x_m + rng.normal(scale=sigma_latent * 0.1, size=n)

    ch1 += rng.normal(scale=sigma_noise, size=n)
    ch2 += rng.normal(scale=sigma_noise, size=n)

    return {"t": t, "ch1": ch1, "ch2": ch2}


def summary_metrics(ch1: np.ndarray, ch2: np.ndarray) -> np.ndarray:
    coherence = np.corrcoef(ch1, ch2)[0, 1]
    non_gauss = stats.kurtosis(ch1 - ch2, fisher=True, bias=False)
    memory = np.corrcoef(ch1[1:], ch1[:-1])[0, 1]
    return np.array([coherence, non_gauss, memory])
