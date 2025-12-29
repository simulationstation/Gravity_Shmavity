#!/usr/bin/env python3
"""
Permutation Falsifier for Nature 2018 Li et al. Data

This script performs N=200 permutation tests to answer:
"How often does a random method-label assignment produce a
 delta_BIC as strong (negative) as the observed value?"

Procedure:
1. Load actual data with real method_id labels
2. Fit null model (intercept only) and full model (intercept + method)
3. Compute observed delta_BIC = BIC_full - BIC_null
4. Permute method_id labels N times
5. For each permutation, compute delta_BIC_perm
6. p-value = fraction of permutations with delta_BIC_perm <= observed

If p < 0.05, the method separation is unlikely to arise by chance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# Physical constant
ALPHA = 1 / 137.035999084
ALPHA2 = ALPHA ** 2  # ~5.325e-05

def numpy_safe_json(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_safe_json(v) for v in obj]
    return obj


def compute_bic(n: int, k: int, rss: float) -> float:
    """Compute BIC for a linear regression model."""
    if rss <= 0:
        return np.inf
    return n * np.log(rss / n) + k * np.log(n)


def fit_ols_rss(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Fit OLS and return only RSS for speed."""
    if weights is not None:
        W = np.diag(np.sqrt(weights))
        Xw = W @ X
        yw = W @ y
        coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        pred = X @ coef
        resid = y - pred
        rss = np.sum(weights * resid**2)
    else:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        resid = y - pred
        rss = np.sum(resid**2)
    return rss


def compute_delta_bic(y: np.ndarray, is_aaf: np.ndarray, weights: np.ndarray | None) -> float:
    """Compute delta_BIC = BIC_full - BIC_null for given method labels."""
    n = len(y)

    # Null model: intercept only
    X_null = np.ones((n, 1))
    rss_null = fit_ols_rss(X_null, y, weights)
    bic_null = compute_bic(n, 1, rss_null)

    # Full model: intercept + is_AAF
    X_full = np.column_stack([np.ones(n), is_aaf])
    rss_full = fit_ols_rss(X_full, y, weights)
    bic_full = compute_bic(n, 2, rss_full)

    return bic_full - bic_null


def run_single_permutation(args: tuple) -> float:
    """Run a single permutation and return delta_BIC."""
    y, is_aaf_orig, weights, seed = args
    rng = np.random.default_rng(seed)
    is_aaf_perm = rng.permutation(is_aaf_orig)
    return compute_delta_bic(y, is_aaf_perm, weights)


def main():
    N_PERM = 200  # Number of permutations

    # Load data
    data_path = Path("data/staged/nature2018_run/g_measurements_minimal.csv")
    df = pd.read_csv(data_path)

    # Prepare variables
    y = df['G_value_1e11'].to_numpy()
    n = len(y)
    mean_g = np.mean(y)

    # Weights: inverse variance
    sigma = df['G_sigma_1e11'].to_numpy()
    weights = 1.0 / sigma**2
    weights = weights / np.sum(weights) * n  # normalize

    # Binary method indicator
    is_aaf = (df['method_id'] == 'angular_acceleration_feedback').astype(float).to_numpy()

    # =========================================================================
    # OBSERVED delta_BIC
    # =========================================================================
    delta_bic_observed = compute_delta_bic(y, is_aaf, weights)

    # Also compute without weights for comparison
    delta_bic_observed_unweighted = compute_delta_bic(y, is_aaf, None)

    print(f"Observed delta_BIC (weighted):   {delta_bic_observed:.4f}")
    print(f"Observed delta_BIC (unweighted): {delta_bic_observed_unweighted:.4f}")
    print(f"\nRunning {N_PERM} permutations...")

    # =========================================================================
    # PERMUTATION TEST (parallelized)
    # =========================================================================
    n_workers = max(1, cpu_count() - 1)
    seeds = list(range(1000, 1000 + N_PERM))

    args_list = [(y, is_aaf, weights, seed) for seed in seeds]

    with Pool(n_workers) as pool:
        delta_bics_perm = pool.map(run_single_permutation, args_list)

    delta_bics_perm = np.array(delta_bics_perm)

    # =========================================================================
    # P-VALUE CALCULATION
    # =========================================================================
    # p-value: fraction of permutations with delta_BIC <= observed
    # (more negative = stronger evidence for method effect)
    p_value = np.mean(delta_bics_perm <= delta_bic_observed)

    # Also compute one-sided p-value for "as extreme or more"
    p_value_extreme = np.mean(np.abs(delta_bics_perm) >= np.abs(delta_bic_observed))

    print(f"\nPermutation results:")
    print(f"  Mean delta_BIC (null):     {np.mean(delta_bics_perm):.4f}")
    print(f"  Std delta_BIC (null):      {np.std(delta_bics_perm):.4f}")
    print(f"  Min delta_BIC (null):      {np.min(delta_bics_perm):.4f}")
    print(f"  Max delta_BIC (null):      {np.max(delta_bics_perm):.4f}")
    print(f"\n  Observed delta_BIC:        {delta_bic_observed:.4f}")
    print(f"  p-value (one-sided):       {p_value:.4f}")
    print(f"  p-value (|extreme|):       {p_value_extreme:.4f}")

    # =========================================================================
    # INTERPRET RESULTS
    # =========================================================================
    # Under the null (random labels), delta_BIC should be around 0 or slightly positive
    # (adding a useless parameter incurs BIC penalty of ~ln(n)/2)
    bic_penalty = np.log(n)
    expected_null_delta_bic = bic_penalty / 2  # ~1.8 for n=36

    # =========================================================================
    # METHOD EFFECT SIZE
    # =========================================================================
    # Compute the method separation directly
    g_tos = y[is_aaf == 0]
    g_aaf = y[is_aaf == 1]
    method_sep = np.mean(g_aaf) - np.mean(g_tos)
    method_sep_ppm = method_sep / mean_g * 1e6
    alpha2_ppm = ALPHA2 * 1e6

    results = {
        "analysis": "permutation_falsifier",
        "dataset": "nature2018_li",
        "n": n,
        "n_permutations": N_PERM,
        "alpha2": ALPHA2,
        "observed": {
            "delta_bic_weighted": float(delta_bic_observed),
            "delta_bic_unweighted": float(delta_bic_observed_unweighted),
            "method_separation_ppm": float(method_sep_ppm),
            "alpha2_ppm": float(alpha2_ppm),
            "separation_over_alpha2": float(method_sep_ppm / alpha2_ppm),
        },
        "permutation_null": {
            "mean_delta_bic": float(np.mean(delta_bics_perm)),
            "std_delta_bic": float(np.std(delta_bics_perm)),
            "min_delta_bic": float(np.min(delta_bics_perm)),
            "max_delta_bic": float(np.max(delta_bics_perm)),
            "percentile_5": float(np.percentile(delta_bics_perm, 5)),
            "percentile_95": float(np.percentile(delta_bics_perm, 95)),
        },
        "p_values": {
            "p_one_sided": float(p_value),
            "p_extreme": float(p_value_extreme),
            "n_more_extreme": int(np.sum(delta_bics_perm <= delta_bic_observed)),
        },
        "interpretation": {
            "expected_null_delta_bic": float(expected_null_delta_bic),
            "observed_is_significant": bool(p_value < 0.05),
            "method_effect_is_real": bool(p_value < 0.05),
        },
        "delta_bics_all_permutations": [float(x) for x in delta_bics_perm],
    }

    # =========================================================================
    # OUTPUT
    # =========================================================================
    out_dir = Path("outputs/nature2018/permutation_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(numpy_safe_json(results), f, indent=2)

    # Summary file without full permutation array
    results_summary = {k: v for k, v in results.items() if k != "delta_bics_all_permutations"}
    with open(out_dir / "results_summary.json", "w") as f:
        json.dump(numpy_safe_json(results_summary), f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Method separation: {method_sep_ppm:.1f} ppm")
    print(f"Alpha^2 scale:     {alpha2_ppm:.1f} ppm")
    print(f"Ratio:             {method_sep_ppm/alpha2_ppm:.2f}")
    print()
    print(f"Observed delta_BIC: {delta_bic_observed:.2f}")
    print(f"Null mean:          {np.mean(delta_bics_perm):.2f}")
    print(f"p-value:            {p_value:.4f}")
    print()
    if p_value < 0.05:
        print("RESULT: The method separation is STATISTICALLY SIGNIFICANT")
        print("        (unlikely to arise from random label assignment)")
    else:
        print("RESULT: The method separation is NOT statistically significant")
        print("        (could arise from random label assignment)")
    print()
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
