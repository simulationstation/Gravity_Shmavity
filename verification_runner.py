#!/usr/bin/env python3
"""
Comprehensive verification of M0..M10 ladder framework.
Tests calibration (FPR), power curves, and ladder falsification.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mverse_ladder.inference.fit import (
    fit_gravity,
    fit_gravity_m0,
    fit_timeseries,
    fit_timeseries_m0,
)
from mverse_ladder.inference.model_selection import ModelResult, delta_bic
from mverse_ladder.physics.ladder import Ladder
from mverse_ladder.physics.observables.gravity_config import simulate_gravity
from mverse_ladder.physics.observables.timeseries_twochannel import simulate_timeseries


# =============================================================================
# Calibration: Test FPR at a0=0
# =============================================================================

def run_single_null_gravity(args: tuple) -> dict:
    """Run single null simulation for gravity preset."""
    seed, max_k, ladder = args
    rng = np.random.default_rng(seed)
    data = simulate_gravity(
        n=200, config_dim=3, ladder=ladder, max_m=max_k,
        a0=0.0,  # NULL SIGNAL
        sigma_latent=0.5, sigma_noise=0.3, rng=rng
    )
    m0 = fit_gravity_m0(data)
    results = [ModelResult(k=0, bic=m0["bic"], loglik=m0["loglik"], a0=None)]
    for k in range(1, max_k + 1):
        mk = fit_gravity(data, k, ladder, sigma_latent=0.5)
        results.append(ModelResult(k=k, bic=mk["bic"], loglik=mk["loglik"], a0=mk["a0"]))
    return {"delta_bics": delta_bic(results), "bics": [r.bic for r in results]}


def run_single_null_timeseries(args: tuple) -> dict:
    """Run single null simulation for timeseries preset."""
    seed, max_k, ladder = args
    rng = np.random.default_rng(seed)
    data = simulate_timeseries(
        n=2000, dt=0.1, ladder=ladder, max_m=max_k,
        a0=0.0,  # NULL SIGNAL
        tau_base=2.0, sigma_latent=0.3, sigma_noise=0.2, rng=rng
    )
    m0 = fit_timeseries_m0(data)
    results = [ModelResult(k=0, bic=m0["bic"], loglik=m0["loglik"], a0=None)]
    for k in range(1, max_k + 1):
        mk = fit_timeseries(data, k, ladder, tau_base=2.0, sigma_latent=0.3)
        results.append(ModelResult(k=k, bic=mk["bic"], loglik=mk["loglik"], a0=mk["a0"]))
    return {"delta_bics": delta_bic(results), "bics": [r.bic for r in results]}


def calibrate_fpr(preset: str, n_null: int = 200, max_k: int = 10, alpha: float = 0.05) -> dict:
    """
    Calibrate false positive rate at a0=0.
    FPR should be ~alpha for each k if calibration is correct.
    """
    ladder = Ladder()
    n_workers = max(1, cpu_count() - 1)

    if preset == "gravity_synth":
        run_fn = run_single_null_gravity
    else:
        run_fn = run_single_null_timeseries

    args_list = [(seed, max_k, ladder) for seed in range(n_null)]

    with Pool(n_workers) as pool:
        results = pool.map(run_fn, args_list)

    # Aggregate delta BICs for each k
    all_delta_bics = np.array([r["delta_bics"] for r in results])  # (n_null, max_k+1)

    # Compute empirical thresholds at alpha quantile
    # delta_bic = BIC_k - BIC_0: negative means M_k preferred
    thresholds = {}
    fpr_at_each_k = {}
    for k in range(1, max_k + 1):
        delta_k = all_delta_bics[:, k]  # delta_BIC for this k vs M0
        # Under null, delta_BIC should be ~positive (M_k has more params, higher BIC)
        # Threshold: alpha percentile of null distribution (detect if delta < threshold)
        thresholds[f"k{k}"] = float(np.percentile(delta_k, alpha * 100))
        # FPR: fraction of null runs where delta_BIC < 0 (M_k incorrectly preferred)
        # Lower BIC is better, so negative delta = BIC_k < BIC_0 = M_k preferred
        fpr_at_each_k[f"k{k}"] = float(np.mean(delta_k < 0))

    return {
        "preset": preset,
        "n_null": n_null,
        "alpha": alpha,
        "max_k": max_k,
        "thresholds": thresholds,
        "fpr_per_k": fpr_at_each_k,
        "delta_bics_raw": all_delta_bics.tolist(),
        "mean_delta_bic_per_k": {f"k{k}": float(all_delta_bics[:, k].mean()) for k in range(1, max_k + 1)},
        "std_delta_bic_per_k": {f"k{k}": float(all_delta_bics[:, k].std()) for k in range(1, max_k + 1)},
    }


# =============================================================================
# Power Analysis: Detection probability vs A0
# =============================================================================

def run_single_power_gravity(args: tuple) -> dict:
    """Run single simulation at given a0 for power analysis."""
    seed, a0, max_k, ladder = args
    rng = np.random.default_rng(seed)
    data = simulate_gravity(
        n=200, config_dim=3, ladder=ladder, max_m=max_k,
        a0=a0, sigma_latent=0.5, sigma_noise=0.3, rng=rng
    )
    m0 = fit_gravity_m0(data)
    results = [ModelResult(k=0, bic=m0["bic"], loglik=m0["loglik"], a0=None)]
    for k in range(1, max_k + 1):
        mk = fit_gravity(data, k, ladder, sigma_latent=0.5)
        results.append(ModelResult(k=k, bic=mk["bic"], loglik=mk["loglik"], a0=mk["a0"]))
    deltas = delta_bic(results)
    best_k = int(np.argmin(deltas))  # k with lowest delta_BIC (most preferred over M0)
    return {"a0": a0, "delta_bics": deltas, "best_k": best_k}


def run_single_power_timeseries(args: tuple) -> dict:
    """Run single simulation at given a0 for power analysis."""
    seed, a0, max_k, ladder = args
    rng = np.random.default_rng(seed)
    data = simulate_timeseries(
        n=2000, dt=0.1, ladder=ladder, max_m=max_k,
        a0=a0, tau_base=2.0, sigma_latent=0.3, sigma_noise=0.2, rng=rng
    )
    m0 = fit_timeseries_m0(data)
    results = [ModelResult(k=0, bic=m0["bic"], loglik=m0["loglik"], a0=None)]
    for k in range(1, max_k + 1):
        mk = fit_timeseries(data, k, ladder, tau_base=2.0, sigma_latent=0.3)
        results.append(ModelResult(k=k, bic=mk["bic"], loglik=mk["loglik"], a0=mk["a0"]))
    deltas = delta_bic(results)
    best_k = int(np.argmax(deltas))
    return {"a0": a0, "delta_bics": deltas, "best_k": best_k}


def power_curve(preset: str, a0_values: np.ndarray, n_reps: int = 50, max_k: int = 10,
                thresholds: dict = None) -> dict:
    """
    Compute detection power curve: P(detect | a0) for each k.
    """
    ladder = Ladder()
    n_workers = max(1, cpu_count() - 1)

    if preset == "gravity_synth":
        run_fn = run_single_power_gravity
    else:
        run_fn = run_single_power_timeseries

    # Build args for all a0 x reps combinations
    args_list = []
    for a0 in a0_values:
        for rep in range(n_reps):
            args_list.append((1000 + rep + int(a0 * 1e6), a0, max_k, ladder))

    with Pool(n_workers) as pool:
        results = pool.map(run_fn, args_list)

    # Group results by a0
    power_by_a0 = {}
    for a0 in a0_values:
        a0_results = [r for r in results if np.isclose(r["a0"], a0)]
        detection_rates = {}
        for k in range(3, max_k + 1):  # k=3..10 as requested
            # Detection: delta_BIC[k] < threshold[k] (negative delta = M_k preferred)
            if thresholds and f"k{k}" in thresholds:
                thresh = thresholds[f"k{k}"]
            else:
                thresh = 0  # If no thresholds, use 0 (negative = prefer M_k)
            detected = sum(1 for r in a0_results if r["delta_bics"][k] < thresh)
            detection_rates[f"k{k}"] = detected / len(a0_results)
        power_by_a0[float(a0)] = detection_rates

    return {
        "preset": preset,
        "a0_values": a0_values.tolist(),
        "n_reps": n_reps,
        "max_k": max_k,
        "power_by_a0": power_by_a0,
    }


# =============================================================================
# Ladder Falsifier: Alpha-ladder vs Free-epsilon comparison
# =============================================================================

def fit_free_epsilon_gravity(data: dict, max_k: int) -> dict:
    """
    Fit model with unconstrained epsilon per layer (free scaling).
    This is a model comparison test: does the alpha-ladder constraint improve fit?
    """
    # Design: y = sum(eps_m * X_m) + nuisance + noise
    # Free model: each eps_m is independently estimated
    n = len(data["y"])
    config = data["config"]
    temp = data["temp"]
    geom = data["geom"]

    # Build design matrix with free coefficients for each layer
    rng = np.random.default_rng(42)  # Deterministic latent
    phi = np.column_stack([np.ones(n), config])

    # Generate fixed latent effects
    latent_effects = []
    for m in range(1, max_k + 1):
        rng_m = np.random.default_rng(m)
        beta_m = rng_m.normal(size=phi.shape[1])
        X_m = phi @ beta_m
        latent_effects.append(X_m)

    # Design matrix: [latent_1, latent_2, ..., latent_k, temp, geom, geom^2, intercept]
    X_design = np.column_stack([
        *latent_effects,
        temp, geom, geom**2, np.ones(n)
    ])

    # Least squares
    coeffs, resid, rank, s = np.linalg.lstsq(X_design, data["y"], rcond=None)
    pred = X_design @ coeffs
    residuals = data["y"] - pred
    sigma = np.sqrt(np.mean(residuals**2))

    n_params = X_design.shape[1]
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma**2) + 1)
    bic = n_params * np.log(n) - 2 * loglik

    return {
        "bic": bic,
        "loglik": loglik,
        "n_params": n_params,
        "sigma": sigma,
        "eps_values": coeffs[:max_k].tolist(),  # Free epsilon estimates
    }


def fit_free_epsilon_timeseries(data: dict, max_k: int) -> dict:
    """Fit free-epsilon model for timeseries."""
    n = len(data["t"])
    y = (data["ch1"] + data["ch2"]) / 2  # Use mean of channels

    # Generate OU processes as basis
    tau_base = 2.0
    sigma_latent = 0.3

    latent_effects = []
    for m in range(1, max_k + 1):
        tau_m = tau_base * (1 + 0.5 * (m - 1))
        rng_m = np.random.default_rng(m)
        ou = np.zeros(n)
        dt = data["t"][1] - data["t"][0] if len(data["t"]) > 1 else 0.1
        for i in range(1, n):
            ou[i] = ou[i-1] * np.exp(-dt / tau_m) + sigma_latent * rng_m.normal() * np.sqrt(1 - np.exp(-2*dt/tau_m))
        latent_effects.append(ou)

    # Nuisance: linear trend + seasonal
    t_norm = (data["t"] - data["t"].mean()) / (data["t"].std() + 1e-10)
    X_design = np.column_stack([
        *latent_effects,
        t_norm, np.sin(2*np.pi*data["t"]/10), np.cos(2*np.pi*data["t"]/10), np.ones(n)
    ])

    coeffs, resid, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
    pred = X_design @ coeffs
    residuals = y - pred
    sigma = np.sqrt(np.mean(residuals**2))

    n_params = X_design.shape[1]
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma**2) + 1)
    bic = n_params * np.log(n) - 2 * loglik

    return {
        "bic": bic,
        "loglik": loglik,
        "n_params": n_params,
        "sigma": sigma,
        "eps_values": coeffs[:max_k].tolist(),
    }


def ladder_falsifier_single(args: tuple) -> dict:
    """Compare alpha-ladder constrained vs free-epsilon model."""
    seed, preset, max_k, a0, ladder = args
    rng = np.random.default_rng(seed)

    if preset == "gravity_synth":
        data = simulate_gravity(
            n=200, config_dim=3, ladder=ladder, max_m=max_k,
            a0=a0, sigma_latent=0.5, sigma_noise=0.3, rng=rng
        )
        # Constrained fit (alpha-ladder)
        constrained = fit_gravity(data, max_k, ladder, sigma_latent=0.5)
        # Free-epsilon fit
        free = fit_free_epsilon_gravity(data, max_k)
    else:
        data = simulate_timeseries(
            n=2000, dt=0.1, ladder=ladder, max_m=max_k,
            a0=a0, tau_base=2.0, sigma_latent=0.3, sigma_noise=0.2, rng=rng
        )
        constrained = fit_timeseries(data, max_k, ladder, tau_base=2.0, sigma_latent=0.3)
        free = fit_free_epsilon_timeseries(data, max_k)

    # BIC comparison: lower is better
    # delta = BIC_constrained - BIC_free
    # If delta < 0: constrained (alpha-ladder) is preferred
    # If delta > 0: free model is preferred
    delta_bic = constrained["bic"] - free["bic"]

    return {
        "seed": seed,
        "a0": a0,
        "bic_constrained": constrained["bic"],
        "bic_free": free["bic"],
        "delta_bic": delta_bic,
        "prefers_alpha_ladder": delta_bic < 0,
    }


def run_ladder_falsifier(preset: str, n_reps: int = 100, max_k: int = 6) -> dict:
    """
    Test whether alpha-ladder constraint is preferred over free-epsilon.
    Run at a0=1.0 (true signal) and a0=0 (null).
    """
    ladder = Ladder()
    n_workers = max(1, cpu_count() - 1)

    # Test at both a0=0 (null) and a0=1.0 (signal)
    args_list = []
    for a0 in [0.0, 1.0]:
        for rep in range(n_reps):
            args_list.append((rep + int(a0 * 10000), preset, max_k, a0, ladder))

    with Pool(n_workers) as pool:
        results = pool.map(ladder_falsifier_single, args_list)

    # Aggregate by a0
    null_results = [r for r in results if r["a0"] == 0.0]
    signal_results = [r for r in results if r["a0"] == 1.0]

    return {
        "preset": preset,
        "n_reps": n_reps,
        "max_k": max_k,
        "null_a0": {
            "mean_delta_bic": float(np.mean([r["delta_bic"] for r in null_results])),
            "std_delta_bic": float(np.std([r["delta_bic"] for r in null_results])),
            "pct_prefers_alpha_ladder": float(np.mean([r["prefers_alpha_ladder"] for r in null_results])),
        },
        "signal_a0": {
            "mean_delta_bic": float(np.mean([r["delta_bic"] for r in signal_results])),
            "std_delta_bic": float(np.std([r["delta_bic"] for r in signal_results])),
            "pct_prefers_alpha_ladder": float(np.mean([r["prefers_alpha_ladder"] for r in signal_results])),
        },
        "raw_results": results,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_fpr_calibration(calib_gravity: dict, calib_ts: dict, out_path: Path):
    """Plot FPR per k for both presets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, calib, title in zip(axes, [calib_gravity, calib_ts], ["Gravity", "Timeseries"]):
        ks = list(range(1, calib["max_k"] + 1))
        fprs = [calib["fpr_per_k"][f"k{k}"] for k in ks]
        ax.bar(ks, fprs, alpha=0.7, color="steelblue")
        ax.axhline(calib["alpha"], color="red", linestyle="--", label=f"Target α={calib['alpha']}")
        ax.set_xlabel("Model k")
        ax.set_ylabel("False Positive Rate")
        ax.set_title(f"{title} - FPR at a0=0 (N={calib['n_null']})")
        ax.legend()
        ax.set_ylim(0, max(0.2, max(fprs) * 1.2))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_power_curves(power_gravity: dict, power_ts: dict, out_path: Path):
    """Plot power curves for both presets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, power, title in zip(axes, [power_gravity, power_ts], ["Gravity", "Timeseries"]):
        a0_vals = power["a0_values"]
        for k in [3, 5, 7, 10]:
            if k <= power["max_k"]:
                rates = [power["power_by_a0"][float(a0)][f"k{k}"] for a0 in a0_vals]
                ax.plot(a0_vals, rates, marker='o', label=f"k={k}")
        ax.set_xlabel("Signal Amplitude (a0)")
        ax.set_ylabel("Detection Probability")
        ax.set_title(f"{title} - Power Curve")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_evidence_vs_k(calib: dict, out_path: Path, title: str):
    """Plot mean delta BIC vs k with error bars."""
    ks = list(range(1, calib["max_k"] + 1))
    means = [calib["mean_delta_bic_per_k"][f"k{k}"] for k in ks]
    stds = [calib["std_delta_bic_per_k"][f"k{k}"] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ks, means, yerr=stds, marker='o', capsize=5, capthick=2)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Model k")
    ax.set_ylabel("Δ BIC (vs M0)")
    ax.set_title(f"{title} - Evidence vs k (a0=0)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    out_dir = Path("out/verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("M0..M10 LADDER FRAMEWORK VERIFICATION")
    print("="*60)

    # 1. Calibration (FPR at a0=0)
    print("\n[1/4] Running calibration checks (N=200)...")
    calib_gravity = calibrate_fpr("gravity_synth", n_null=200, max_k=10, alpha=0.05)
    calib_ts = calibrate_fpr("timeseries_synth", n_null=200, max_k=10, alpha=0.05)

    # Save calibration results
    with open(out_dir / "calibration_gravity.json", "w") as f:
        json.dump({k: v for k, v in calib_gravity.items() if k != "delta_bics_raw"}, f, indent=2)
    with open(out_dir / "calibration_timeseries.json", "w") as f:
        json.dump({k: v for k, v in calib_ts.items() if k != "delta_bics_raw"}, f, indent=2)

    # Thresholds JSON (as requested)
    thresholds_combined = {
        "gravity_synth": calib_gravity["thresholds"],
        "timeseries_synth": calib_ts["thresholds"],
    }
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(thresholds_combined, f, indent=2)

    print(f"  Gravity FPR per k: {calib_gravity['fpr_per_k']}")
    print(f"  Timeseries FPR per k: {calib_ts['fpr_per_k']}")

    # 2. Power curves
    print("\n[2/4] Running power analysis...")
    a0_values = np.logspace(-2, 1, 10)  # 0.01 to 10
    power_gravity = power_curve("gravity_synth", a0_values, n_reps=50, max_k=10,
                                 thresholds=calib_gravity["thresholds"])
    power_ts = power_curve("timeseries_synth", a0_values, n_reps=50, max_k=10,
                           thresholds=calib_ts["thresholds"])

    with open(out_dir / "power_gravity.json", "w") as f:
        json.dump(power_gravity, f, indent=2)
    with open(out_dir / "power_timeseries.json", "w") as f:
        json.dump(power_ts, f, indent=2)

    # 3. Ladder falsifiers
    print("\n[3/4] Running ladder falsifier (alpha-ladder vs free-epsilon)...")
    falsifier_gravity = run_ladder_falsifier("gravity_synth", n_reps=100, max_k=6)
    falsifier_ts = run_ladder_falsifier("timeseries_synth", n_reps=100, max_k=6)

    # Save without raw results (too large)
    with open(out_dir / "falsifier_gravity.json", "w") as f:
        json.dump({k: v for k, v in falsifier_gravity.items() if k != "raw_results"}, f, indent=2)
    with open(out_dir / "falsifier_timeseries.json", "w") as f:
        json.dump({k: v for k, v in falsifier_ts.items() if k != "raw_results"}, f, indent=2)

    print(f"  Gravity - Null a0=0: {falsifier_gravity['null_a0']['pct_prefers_alpha_ladder']*100:.1f}% prefer α-ladder")
    print(f"  Gravity - Signal a0=1: {falsifier_gravity['signal_a0']['pct_prefers_alpha_ladder']*100:.1f}% prefer α-ladder")
    print(f"  Timeseries - Null a0=0: {falsifier_ts['null_a0']['pct_prefers_alpha_ladder']*100:.1f}% prefer α-ladder")
    print(f"  Timeseries - Signal a0=1: {falsifier_ts['signal_a0']['pct_prefers_alpha_ladder']*100:.1f}% prefer α-ladder")

    # 4. Generate plots
    print("\n[4/4] Generating plots...")
    plot_fpr_calibration(calib_gravity, calib_ts, out_dir / "fpr_calibration.png")
    plot_power_curves(power_gravity, power_ts, out_dir / "power_curves.png")
    plot_evidence_vs_k(calib_gravity, out_dir / "evidence_gravity.png", "Gravity")
    plot_evidence_vs_k(calib_ts, out_dir / "evidence_timeseries.png", "Timeseries")

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print(f"\nArtifacts saved to: {out_dir.absolute()}")

    # Return summary for report generation
    return {
        "calibration": {"gravity": calib_gravity, "timeseries": calib_ts},
        "power": {"gravity": power_gravity, "timeseries": power_ts},
        "falsifier": {"gravity": falsifier_gravity, "timeseries": falsifier_ts},
    }


if __name__ == "__main__":
    summary = main()
