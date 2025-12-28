"""Calibration routines for false positive rates."""
from __future__ import annotations

import numpy as np

from mverse_ladder.physics.ladder import Ladder
from mverse_ladder.physics.observables.gravity_config import simulate_gravity
from mverse_ladder.physics.observables.timeseries_twochannel import simulate_timeseries
from mverse_ladder.inference.fit import fit_gravity, fit_gravity_m0, fit_timeseries, fit_timeseries_m0
from mverse_ladder.inference.model_selection import ModelResult


def calibrate_null(
    preset: str,
    runs: int,
    max_k: int,
    ladder: Ladder,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    delta_bics = []

    for i in range(runs):
        sub_rng = np.random.default_rng(rng.integers(0, 1_000_000))
        if preset == "gravity_synth":
            data = simulate_gravity(
                n=120,
                config_dim=3,
                ladder=ladder,
                max_m=max_k,
                a0=0.0,
                sigma_latent=0.5,
                sigma_noise=0.4,
                rng=sub_rng,
            )
            m0 = fit_gravity_m0(data)
            mk = fit_gravity(data, max_k, ladder, sigma_latent=0.5)
        else:
            data = simulate_timeseries(
                n=1000,
                dt=0.1,
                ladder=ladder,
                max_m=max_k,
                a0=0.0,
                tau_base=2.0,
                sigma_latent=0.3,
                sigma_noise=0.3,
                rng=sub_rng,
            )
            m0 = fit_timeseries_m0(data)
            mk = fit_timeseries(data, max_k, ladder, tau_base=2.0, sigma_latent=0.3)

        delta_bics.append(m0["bic"] - mk["bic"])

    delta_bics = np.array(delta_bics)
    return {
        "delta_bics": delta_bics,
        "threshold_95": np.quantile(delta_bics, 0.95),
        "threshold_99": np.quantile(delta_bics, 0.99),
    }
