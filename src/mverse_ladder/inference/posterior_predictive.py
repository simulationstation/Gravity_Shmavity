"""Posterior predictive checks."""
from __future__ import annotations

import numpy as np

from mverse_ladder.physics.ladder import Ladder
from mverse_ladder.physics.observables.gravity_config import simulate_gravity
from mverse_ladder.physics.observables.timeseries_twochannel import simulate_timeseries


def _statistic(resid: np.ndarray) -> float:
    return float(np.mean(resid**2))


def ppc_gravity(data: dict, ladder: Ladder, max_m: int, a0: float, sigma_latent: float, sigma_noise: float, rng: np.random.Generator, draws: int) -> float:
    observed = _statistic(data["resid"])
    sims = []
    for _ in range(draws):
        sim = simulate_gravity(
            n=len(data["y"]),
            config_dim=data["config"].shape[1],
            ladder=ladder,
            max_m=max_m,
            a0=a0,
            sigma_latent=sigma_latent,
            sigma_noise=sigma_noise,
            rng=rng,
        )
        resid = sim["y"] - sim["y"].mean()
        sims.append(_statistic(resid))
    sims = np.array(sims)
    return float(np.mean(sims >= observed))


def ppc_timeseries(data: dict, ladder: Ladder, max_m: int, a0: float, tau_base: float, sigma_latent: float, sigma_noise: float, rng: np.random.Generator, draws: int) -> float:
    signal = (data["ch1"] + data["ch2"]) / 2
    observed = _statistic(signal - signal.mean())
    sims = []
    for _ in range(draws):
        sim = simulate_timeseries(
            n=len(data["t"]),
            dt=data["t"][1] - data["t"][0],
            ladder=ladder,
            max_m=max_m,
            a0=a0,
            tau_base=tau_base,
            sigma_latent=sigma_latent,
            sigma_noise=sigma_noise,
            rng=rng,
        )
        sim_signal = (sim["ch1"] + sim["ch2"]) / 2
        sims.append(_statistic(sim_signal - sim_signal.mean()))
    sims = np.array(sims)
    return float(np.mean(sims >= observed))
