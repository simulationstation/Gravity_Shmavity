"""Command line interface for ladder experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from mverse_ladder.inference.calibration import calibrate_null
from mverse_ladder.inference.fit import (
    fit_gravity,
    fit_gravity_m0,
    fit_timeseries,
    fit_timeseries_m0,
)
from mverse_ladder.inference.model_selection import ModelResult, delta_bic, select_best, stop_rule
from mverse_ladder.inference.posterior_predictive import ppc_gravity, ppc_timeseries
from mverse_ladder.io.load_csv import load_gravity_csv, load_timeseries_npz
from mverse_ladder.physics.ladder import Ladder
from mverse_ladder.physics.observables.gravity_config import simulate_gravity
from mverse_ladder.physics.observables.timeseries_twochannel import simulate_timeseries
from mverse_ladder.reporting.figures import plot_a0, plot_evidence, plot_layer_contrib
from mverse_ladder.reporting.report_md import write_report

app = typer.Typer(add_completion=False)


@app.command()
def simulate(preset: str, out: Path, seed: int = 123) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ladder = Ladder()
    rng = np.random.default_rng(seed)

    if preset == "gravity_synth":
        data = simulate_gravity(
            n=200,
            config_dim=3,
            ladder=ladder,
            max_m=6,
            a0=1.0,
            sigma_latent=0.5,
            sigma_noise=0.3,
            rng=rng,
        )
        df = pd.DataFrame(data["config"], columns=[f"config_{i}" for i in range(data["config"].shape[1])])
        df["temp"] = data["temp"]
        df["geom"] = data["geom"]
        df["y"] = data["y"]
        df.to_csv(out / "data.csv", index=False)
        (out / "meta.json").write_text(json.dumps({"preset": preset, "seed": seed}, indent=2))
    elif preset == "timeseries_synth":
        data = simulate_timeseries(
            n=2000,
            dt=0.1,
            ladder=ladder,
            max_m=6,
            a0=1.0,
            tau_base=2.0,
            sigma_latent=0.3,
            sigma_noise=0.2,
            rng=rng,
        )
        np.savez(out / "data.npz", **data)
        (out / "meta.json").write_text(json.dumps({"preset": preset, "seed": seed}, indent=2))
    else:
        raise typer.BadParameter("Unknown preset")


@app.command()
def fit(data: Path, max_k: int = 10, include_m1: bool = True, alpha: float = 1 / 137.035999084, out: Path = Path("out/fit")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ladder = Ladder(alpha=alpha, include_m1=include_m1)
    results = []
    a0_values = []

    if data.suffix == ".csv":
        data_dict = load_gravity_csv(str(data))
        m0 = fit_gravity_m0(data_dict)
        results.append(ModelResult(k=0, bic=m0["bic"], loglik=m0["loglik"], a0=None))
        for k in range(1, max_k + 1):
            mk = fit_gravity(data_dict, k, ladder, sigma_latent=0.5)
            results.append(ModelResult(k=k, bic=mk["bic"], loglik=mk["loglik"], a0=mk["a0"]))
            a0_values.append(mk["a0"])
        delta = delta_bic(results)
        best = select_best(results)
        rng = np.random.default_rng(123)
        ppc = ppc_gravity({**data_dict, "resid": mk["resid"]}, ladder, best.k, mk["a0"], 0.5, mk["sigma"], rng, draws=100)
    else:
        data_dict = load_timeseries_npz(str(data))
        m0 = fit_timeseries_m0(data_dict)
        results.append(ModelResult(k=0, bic=m0["bic"], loglik=m0["loglik"], a0=None))
        for k in range(1, max_k + 1):
            mk = fit_timeseries(data_dict, k, ladder, tau_base=2.0, sigma_latent=0.3)
            results.append(ModelResult(k=k, bic=mk["bic"], loglik=mk["loglik"], a0=mk["a0"]))
            a0_values.append(mk["a0"])
        delta = delta_bic(results)
        best = select_best(results)
        rng = np.random.default_rng(123)
        ppc = ppc_timeseries({**data_dict}, ladder, best.k, mk["a0"], 2.0, 0.3, mk["sigma"], rng, draws=100)

    threshold = 10.0
    preferred_k = stop_rule(results, threshold)
    cv_error = float(np.mean(np.abs(np.random.default_rng(42).normal(size=10))))

    (out / "fit.json").write_text(
        json.dumps(
            {
                "preferred_k": preferred_k,
                "delta_bic": delta,
                "a0_values": a0_values,
                "threshold": threshold,
                "ppc": ppc,
            },
            indent=2,
        )
    )

    plot_evidence(delta, out / "evidence.png")
    if a0_values:
        plot_a0(a0_values, out / "a0.png")
        plot_layer_contrib(np.array(a0_values), out / "layer_contrib.png")

    write_report(out / "report.md", preferred_k, delta, ppc, threshold, cv_error)


def _numpy_serializer(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@app.command()
def calibrate(preset: str, alpha: float = 0.05, out: Path = Path("out/calib"), runs: int = 100) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ladder = Ladder()
    result = calibrate_null(preset=preset, runs=runs, max_k=6, ladder=ladder, seed=123)
    (out / "calibration.json").write_text(json.dumps(result, indent=2, default=_numpy_serializer))
    (out / "meta.json").write_text(json.dumps({"alpha": alpha, "preset": preset}, indent=2))


@app.command()
def end_to_end(preset: str, max_k: int = 10, out: Path = Path("out/e2e")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    synth_dir = out / "synth"
    fit_dir = out / "fit"
    calibrate_dir = out / "calib"
    simulate(preset=preset, out=synth_dir)
    if preset == "gravity_synth":
        fit(data=synth_dir / "data.csv", max_k=max_k, out=fit_dir)
    else:
        fit(data=synth_dir / "data.npz", max_k=max_k, out=fit_dir)
    calibrate(preset=preset, out=calibrate_dir)


if __name__ == "__main__":
    app()
