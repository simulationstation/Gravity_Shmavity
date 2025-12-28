"""Model fitting routines."""
from __future__ import annotations

import numpy as np
from scipy import optimize

from mverse_ladder.physics.ladder import Ladder
from mverse_ladder.physics.latent import simulate_linear_latent, simulate_ou_process
from mverse_ladder.physics.observables.gravity_config import build_phi, nuisance_design


def _deterministic_latent_config(phi: np.ndarray, layer: int, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(10_000 + layer)
    return simulate_linear_latent(phi, rng, sigma)


def _deterministic_latent_timeseries(t: np.ndarray, layer: int, tau_base: float, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(20_000 + layer)
    tau = tau_base * (1 + 0.5 * (layer - 1))
    return simulate_ou_process(t, tau, sigma, rng)


def fit_gravity(data: dict, max_m: int, ladder: Ladder, sigma_latent: float) -> dict:
    y = data["y"]
    temp = data["temp"]
    geom = data["geom"]
    config = data["config"]

    phi = build_phi(config)
    nuisance = nuisance_design(temp, geom)

    layer_effect = np.zeros_like(y)
    for m, eps in ladder.ladder_weights(max_m).items():
        x_m = _deterministic_latent_config(phi, m, sigma_latent)
        layer_effect += eps * x_m

    design = np.column_stack([layer_effect, nuisance])

    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    a0 = coef[0]
    beta = coef[1:]
    resid = y - design @ coef
    sigma_hat = np.sqrt(np.mean(resid**2))
    n = y.size
    k = len(coef) + 1
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma_hat**2) + 1)
    bic = k * np.log(n) - 2 * loglik
    return {
        "a0": a0,
        "beta": beta,
        "sigma": sigma_hat,
        "loglik": loglik,
        "bic": bic,
        "resid": resid,
    }


def fit_gravity_m0(data: dict) -> dict:
    y = data["y"]
    nuisance = nuisance_design(data["temp"], data["geom"])
    coef, *_ = np.linalg.lstsq(nuisance, y, rcond=None)
    resid = y - nuisance @ coef
    sigma_hat = np.sqrt(np.mean(resid**2))
    n = y.size
    k = len(coef) + 1
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma_hat**2) + 1)
    bic = k * np.log(n) - 2 * loglik
    return {"beta": coef, "sigma": sigma_hat, "loglik": loglik, "bic": bic, "resid": resid}


def _timeseries_nuisance_matrix(t: np.ndarray) -> np.ndarray:
    t_norm = (t - t.mean()) / t.std()
    return np.column_stack([np.ones_like(t), t_norm, np.sin(2 * np.pi * t_norm), np.cos(2 * np.pi * t_norm)])


def fit_timeseries(data: dict, max_m: int, ladder: Ladder, tau_base: float, sigma_latent: float) -> dict:
    t = data["t"]
    signal = (data["ch1"] + data["ch2"]) / 2

    layer_effect = np.zeros_like(signal)
    for m, eps in ladder.ladder_weights(max_m).items():
        x_m = _deterministic_latent_timeseries(t, m, tau_base, sigma_latent)
        layer_effect += eps * x_m

    nuisance = _timeseries_nuisance_matrix(t)
    design = np.column_stack([layer_effect, nuisance])

    coef, *_ = np.linalg.lstsq(design, signal, rcond=None)
    a0 = coef[0]
    beta = coef[1:]
    resid = signal - design @ coef
    sigma_hat = np.sqrt(np.mean(resid**2))
    n = signal.size
    k = len(coef) + 1
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma_hat**2) + 1)
    bic = k * np.log(n) - 2 * loglik
    return {
        "a0": a0,
        "beta": beta,
        "sigma": sigma_hat,
        "loglik": loglik,
        "bic": bic,
        "resid": resid,
    }


def fit_timeseries_m0(data: dict) -> dict:
    t = data["t"]
    signal = (data["ch1"] + data["ch2"]) / 2
    nuisance = _timeseries_nuisance_matrix(t)
    coef, *_ = np.linalg.lstsq(nuisance, signal, rcond=None)
    resid = signal - nuisance @ coef
    sigma_hat = np.sqrt(np.mean(resid**2))
    n = signal.size
    k = len(coef) + 1
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma_hat**2) + 1)
    bic = k * np.log(n) - 2 * loglik
    return {"beta": coef, "sigma": sigma_hat, "loglik": loglik, "bic": bic, "resid": resid}


def fit_free_layers(data: dict, max_m: int, ladder: Ladder, sigma_latent: float) -> dict:
    """M2-only style: free per-layer weights without alpha ladder."""
    y = data["y"]
    temp = data["temp"]
    geom = data["geom"]
    config = data["config"]

    phi = build_phi(config)
    nuisance = nuisance_design(temp, geom)
    layer_terms = []
    for m in ladder.ladder_weights(max_m).keys():
        x_m = _deterministic_latent_config(phi, m, sigma_latent)
        layer_terms.append(x_m)

    design = np.column_stack(layer_terms + [*nuisance.T])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    sigma_hat = np.sqrt(np.mean(resid**2))
    n = y.size
    k = len(coef) + 1
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma_hat**2) + 1)
    bic = k * np.log(n) - 2 * loglik
    return {"coef": coef, "sigma": sigma_hat, "loglik": loglik, "bic": bic, "resid": resid}
