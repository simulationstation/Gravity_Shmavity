#!/usr/bin/env python3
"""
Method-as-Nuisance Test for Nature 2018 Li et al. Data

This script tests whether the alpha^2 term provides improvement AFTER
fully controlling for method_id (and optionally apparatus sub-configuration).

Models compared:
  A) Nuisance-only: G = intercept + beta_method * is_AAF
  B) Nuisance + alpha^2: G = intercept + beta_method * is_AAF + A0 * alpha^2
  C) Free-scale: G = intercept + beta_method * is_AAF + A * delta_frac

The key question: Does Model B or C improve BIC over Model A?
If yes, does the improvement persist when we add apparatus-level effects?
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

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
    """Compute BIC for a linear regression model.

    BIC = n * ln(RSS/n) + k * ln(n)

    where n = number of observations, k = number of parameters, RSS = residual sum of squares
    """
    if rss <= 0:
        return np.inf
    return n * np.log(rss / n) + k * np.log(n)


def compute_aic(n: int, k: int, rss: float) -> float:
    """Compute AIC for a linear regression model.

    AIC = n * ln(RSS/n) + 2k
    """
    if rss <= 0:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def fit_ols(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Fit OLS (or WLS if weights provided) and return coefficients + stats."""
    n, k = X.shape

    if weights is not None:
        # Weighted least squares: transform problem
        W = np.diag(np.sqrt(weights))
        Xw = W @ X
        yw = W @ y
        coef, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
        pred = X @ coef
        resid = y - pred
        # For RSS in WLS, use weighted residuals
        rss = np.sum(weights * resid**2)
    else:
        coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        resid = y - pred
        rss = np.sum(resid**2)

    bic = compute_bic(n, k, rss)
    aic = compute_aic(n, k, rss)

    # Estimate coefficient standard errors
    if weights is not None:
        sigma2 = rss / (n - k)
        XtWX = X.T @ np.diag(weights) @ X
        try:
            cov_matrix = sigma2 * np.linalg.inv(XtWX)
            se = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)
    else:
        sigma2 = rss / (n - k)
        try:
            cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)

    return {
        'coef': coef,
        'se': se,
        'rss': rss,
        'bic': bic,
        'aic': aic,
        'resid': resid,
        'pred': pred,
        'n': n,
        'k': k,
    }


def bootstrap_coefficient(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None,
                          coef_idx: int, n_boot: int = 2000) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a specific coefficient."""
    rng = np.random.default_rng(42)
    n = len(y)
    boot_coefs = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        w_boot = weights[idx] if weights is not None else None

        try:
            result = fit_ols(X_boot, y_boot, w_boot)
            boot_coefs.append(result['coef'][coef_idx])
        except:
            continue

    boot_coefs = np.array(boot_coefs)
    return (
        float(np.mean(boot_coefs)),
        float(np.percentile(boot_coefs, 2.5)),
        float(np.percentile(boot_coefs, 97.5))
    )


def main():
    # Load data
    data_path = Path("data/staged/nature2018_run/g_measurements_minimal.csv")
    df = pd.read_csv(data_path)

    # Prepare variables
    y = df['G_value_1e11'].to_numpy()
    n = len(y)
    mean_g = np.mean(y)

    # Weights: inverse variance (proportional to 1/sigma^2)
    sigma = df['G_sigma_1e11'].to_numpy()
    weights = 1.0 / sigma**2
    weights = weights / np.sum(weights) * n  # normalize for numerical stability

    # Binary method indicator
    is_aaf = (df['method_id'] == 'angular_acceleration_feedback').astype(float).to_numpy()

    # Extract apparatus from config_id (AAF-I, AAF-II, AAF-III, TOS-I, TOS-II)
    def extract_apparatus(config_id: str) -> str:
        if config_id.startswith('AAF_'):
            return config_id.replace('AAF_', '')  # AAF-I, AAF-II, AAF-III
        elif config_id.startswith('TOS_'):
            # ToS fibers 1-4 are TOS-I, 5-7 are TOS-II
            fiber_num = int(config_id.split('_')[-1].replace('fiber_', ''))
            return 'TOS-I' if fiber_num <= 4 else 'TOS-II'
        return 'unknown'

    df['apparatus'] = df['config_id'].apply(extract_apparatus)

    # Create dummy variables for apparatus (5 levels: AAF-I, AAF-II, AAF-III, TOS-I, TOS-II)
    apparatus_dummies = pd.get_dummies(df['apparatus'], prefix='app', drop_first=True)

    results = {
        "analysis": "method_nuisance_test",
        "dataset": "nature2018_li",
        "n": n,
        "alpha2": ALPHA2,
        "mean_G": float(mean_g),
    }

    # =========================================================================
    # TIER 1: Method-level models (is_AAF only)
    # =========================================================================

    # Model A: Nuisance-only (intercept + is_AAF)
    X_A = np.column_stack([np.ones(n), is_aaf])
    fit_A = fit_ols(X_A, y, weights)

    # Model B: Nuisance + alpha^2 term
    # The alpha^2 term acts as a "method × alpha^2" interaction
    # We model: G = intercept + beta_method*is_AAF + A0*is_AAF*alpha^2
    # But since alpha^2 is a constant, this is equivalent to adding is_AAF*alpha^2 as a regressor
    # Actually, to test if alpha^2 fits the METHOD DIFFERENCE, we should see if the coefficient
    # on is_AAF is close to alpha^2 * mean_G
    #
    # Better formulation: Model B tests if residuals from A have alpha^2 structure
    # We'll use: G = intercept + beta_method*is_AAF + A0*(is_AAF - mean(is_AAF))
    # where A0 is scaled so that if the method effect = alpha^2, A0 = 1

    # Actually, the simplest approach:
    # Model A: G = a + b*is_AAF
    # Model B: G = a + b*is_AAF + c*alpha2_indicator
    # where alpha2_indicator = (is_AAF - 0.5) * 2 * alpha^2 * mean_G
    # This tests if adding an alpha^2-scaled shift improves fit

    # Let me reformulate more clearly:
    # We want to test if the method separation equals alpha^2
    # Model A: G = intercept + delta * is_AAF
    # Model B: Same, but constrained so delta = alpha^2 * mean_G * A0

    # Clearer approach: Compare against the alpha^2 prediction
    # delta_observed = beta_AAF from Model A
    # delta_predicted = alpha^2 * mean_G
    # Test: is delta_observed / delta_predicted close to 1?

    delta_observed = fit_A['coef'][1]
    delta_predicted = ALPHA2 * mean_g
    delta_ratio = delta_observed / delta_predicted

    # Bootstrap CI for the ratio
    boot_mean, boot_ci_low, boot_ci_high = bootstrap_coefficient(X_A, y, weights, coef_idx=1)
    ratio_ci_low = boot_ci_low / delta_predicted
    ratio_ci_high = boot_ci_high / delta_predicted

    results["tier1_method_level"] = {
        "model_A_nuisance_only": {
            "intercept": float(fit_A['coef'][0]),
            "beta_AAF": float(fit_A['coef'][1]),
            "beta_AAF_se": float(fit_A['se'][1]),
            "bic": float(fit_A['bic']),
            "aic": float(fit_A['aic']),
            "rss": float(fit_A['rss']),
        },
        "alpha2_comparison": {
            "delta_observed": float(delta_observed),
            "delta_predicted_alpha2": float(delta_predicted),
            "ratio_observed_over_alpha2": float(delta_ratio),
            "ratio_bootstrap_mean": float(boot_mean / delta_predicted),
            "ratio_95ci_low": float(ratio_ci_low),
            "ratio_95ci_high": float(ratio_ci_high),
            "ratio_ci_includes_1": bool(ratio_ci_low <= 1.0 <= ratio_ci_high),
        }
    }

    # =========================================================================
    # TIER 2: Apparatus-level models (5 categories)
    # =========================================================================

    # Model C: Full apparatus dummies (intercept + 4 apparatus dummies)
    X_C = np.column_stack([np.ones(n), apparatus_dummies.to_numpy()])
    fit_C = fit_ols(X_C, y, weights)

    # Compare Model A vs Model C
    delta_bic_AC = fit_C['bic'] - fit_A['bic']  # negative means C is better

    results["tier2_apparatus_level"] = {
        "model_C_apparatus_dummies": {
            "n_params": fit_C['k'],
            "bic": float(fit_C['bic']),
            "aic": float(fit_C['aic']),
            "rss": float(fit_C['rss']),
            "coefficients": {
                "intercept": float(fit_C['coef'][0]),
                **{col: float(fit_C['coef'][i+1]) for i, col in enumerate(apparatus_dummies.columns)}
            }
        },
        "comparison_A_vs_C": {
            "delta_bic_C_minus_A": float(delta_bic_AC),
            "apparatus_improves_over_method": bool(delta_bic_AC < 0),
        }
    }

    # =========================================================================
    # TIER 3: Test residual structure after method correction
    # =========================================================================

    # After fitting Model A, do residuals show any alpha^2-scale structure?
    resid_A = fit_A['resid']
    resid_std = np.std(resid_A)
    resid_frac = resid_std / mean_g

    # Split residuals by method
    resid_tos = resid_A[is_aaf == 0]
    resid_aaf = resid_A[is_aaf == 1]

    results["tier3_residual_analysis"] = {
        "resid_std": float(resid_std),
        "resid_fractional": float(resid_frac),
        "resid_fractional_over_alpha2": float(resid_frac / ALPHA2),
        "resid_tos_std": float(np.std(resid_tos)),
        "resid_aaf_std": float(np.std(resid_aaf)),
        "shapiro_wilk_p": float(stats.shapiro(resid_A)[1]),
    }

    # =========================================================================
    # TIER 4: Key diagnostic - does alpha^2 FIT the method separation?
    # =========================================================================

    # The critical question: beta_AAF from Model A ≈ alpha^2 * mean_G?
    # H0: beta_AAF = alpha^2 * mean_G (ratio = 1)
    # We test this with the bootstrap CI

    # T-test for beta_AAF = delta_predicted
    t_stat = (fit_A['coef'][1] - delta_predicted) / fit_A['se'][1]
    p_value_two_sided = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

    results["tier4_alpha2_diagnostic"] = {
        "null_hypothesis": "beta_AAF = alpha^2 * mean_G",
        "beta_AAF_observed": float(fit_A['coef'][1]),
        "alpha2_x_mean_G": float(delta_predicted),
        "difference": float(fit_A['coef'][1] - delta_predicted),
        "t_statistic": float(t_stat),
        "p_value_two_sided": float(p_value_two_sided),
        "conclusion": "CONSISTENT" if p_value_two_sided > 0.05 else "INCONSISTENT",
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================

    results["summary"] = {
        "method_separation_ppm": float(delta_observed / mean_g * 1e6),
        "alpha2_scale_ppm": float(ALPHA2 * 1e6),
        "ratio_method_sep_to_alpha2": float(delta_ratio),
        "ratio_95ci": [float(ratio_ci_low), float(ratio_ci_high)],
        "ratio_ci_includes_1": bool(ratio_ci_low <= 1.0 <= ratio_ci_high),
        "apparatus_improves_fit": bool(delta_bic_AC < -2),
        "residual_scale_fraction_of_alpha2": float(resid_frac / ALPHA2),
    }

    # =========================================================================
    # OUTPUT
    # =========================================================================
    out_dir = Path("outputs/nature2018/method_nuisance_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(numpy_safe_json(results), f, indent=2)

    # Print summary
    print("=" * 60)
    print("Method-as-Nuisance Test Results")
    print("=" * 60)
    print(f"\nDataset: {n} measurements from Nature 2018 Li et al.")
    print(f"alpha^2 = {ALPHA2:.6e} ({ALPHA2*1e6:.1f} ppm)")
    print()
    print("TIER 1: Method-level Model")
    print(f"  beta_AAF (observed method separation): {delta_observed:.6e}")
    print(f"  alpha^2 * mean_G (predicted):          {delta_predicted:.6e}")
    print(f"  Ratio (observed / predicted):          {delta_ratio:.3f}")
    print(f"  95% CI for ratio:                      [{ratio_ci_low:.3f}, {ratio_ci_high:.3f}]")
    print(f"  CI includes 1.0:                       {ratio_ci_low <= 1.0 <= ratio_ci_high}")
    print()
    print("TIER 2: Apparatus-level Model")
    print(f"  Delta BIC (apparatus vs method):       {delta_bic_AC:.2f}")
    print(f"  Apparatus improves fit (BIC < -2):     {delta_bic_AC < -2}")
    print()
    print("TIER 3: Residual Analysis")
    print(f"  Residual std / alpha^2:                {resid_frac/ALPHA2:.2f}")
    print()
    print("TIER 4: Alpha^2 Diagnostic")
    print(f"  T-test p-value:                        {p_value_two_sided:.4f}")
    print(f"  Conclusion:                            {results['tier4_alpha2_diagnostic']['conclusion']}")
    print()
    print(f"Results saved to: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
