"""Gravity configuration observable model and simulation."""
from __future__ import annotations

import numpy as np

from mverse_ladder.physics.ladder import Ladder
from mverse_ladder.physics.latent import simulate_linear_latent


def build_phi(config: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(config.shape[0]), config])


def simulate_gravity(
    n: int,
    config_dim: int,
    ladder: Ladder,
    max_m: int,
    a0: float,
    sigma_latent: float,
    sigma_noise: float,
    rng: np.random.Generator,
) -> dict:
    config = rng.normal(size=(n, config_dim))
    temp = rng.normal(size=n)
    geom = rng.normal(size=n)
    phi = build_phi(config)

    layer_terms = []
    for m, eps in ladder.ladder_weights(max_m).items():
        x_m = simulate_linear_latent(phi, rng, sigma_latent)
        layer_terms.append(a0 * eps * x_m)

    nuisance = 0.5 * temp + 0.2 * geom + 0.1 * geom**2
    y = np.sum(layer_terms, axis=0) + nuisance + rng.normal(scale=sigma_noise, size=n)

    return {
        "config": config,
        "temp": temp,
        "geom": geom,
        "y": y,
    }


def nuisance_design(temp: np.ndarray, geom: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(temp), temp, geom, geom**2])


def model_matrix(config: np.ndarray, temp: np.ndarray, geom: np.ndarray) -> np.ndarray:
    phi = build_phi(config)
    base = nuisance_design(temp, geom)
    return base, phi
