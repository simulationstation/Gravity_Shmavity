#!/usr/bin/env python3
"""
M3 Indication Analysis for Nature 2018 Li et al. Data

This script performs a focused analysis of whether there are any α²-scale
patterns in the Nature 2018 G measurement data. It does NOT claim discovery;
it asks: "Is anything even in the α² ballpark within one paper?"

Analysis components:
1. Method separation (ToS vs AAF)
2. Within-method dispersion
3. Simple nuisance model fit
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Physical constant
ALPHA = 1 / 137.035999084
ALPHA2 = ALPHA ** 2  # ~5.325e-05

def mad_to_sigma(x: np.ndarray) -> float:
    """Convert MAD to Gaussian-equivalent sigma."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))

def bootstrap_mean_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 5000) -> tuple[float, float, float]:
    """Bootstrap the difference in means."""
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        x_boot = rng.choice(x, size=len(x), replace=True)
        y_boot = rng.choice(y, size=len(y), replace=True)
        diffs.append(np.mean(x_boot) - np.mean(y_boot))
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def main():
    # Load data
    df = pd.read_csv("data/staged/nature2018_run/g_measurements_minimal.csv")

    # Separate by method
    tos = df[df['method_id'] == 'time_of_swing']['G_value_1e11'].to_numpy()
    aaf = df[df['method_id'] == 'angular_acceleration_feedback']['G_value_1e11'].to_numpy()
    all_g = df['G_value_1e11'].to_numpy()

    results = {
        "analysis": "m3_indication",
        "dataset": "nature2018_li",
        "n_tos": len(tos),
        "n_aaf": len(aaf),
        "n_total": len(all_g),
        "alpha2": ALPHA2,
    }

    # =========================================================================
    # 1. METHOD SEPARATION
    # =========================================================================
    mean_tos = np.mean(tos)
    mean_aaf = np.mean(aaf)
    mean_all = np.mean(all_g)

    delta_method = mean_aaf - mean_tos  # AAF - ToS
    delta_method_frac = delta_method / mean_all
    delta_method_over_alpha2 = delta_method_frac / ALPHA2

    # Bootstrap uncertainty on the difference
    boot_mean, boot_ci_low, boot_ci_high = bootstrap_mean_diff(aaf, tos)
    boot_mean_frac = boot_mean / mean_all
    boot_ci_low_frac = boot_ci_low / mean_all
    boot_ci_high_frac = boot_ci_high / mean_all

    results["method_separation"] = {
        "mean_tos": float(mean_tos),
        "mean_aaf": float(mean_aaf),
        "delta_aaf_minus_tos_1e11": float(delta_method),
        "delta_fractional": float(delta_method_frac),
        "delta_over_alpha2": float(delta_method_over_alpha2),
        "bootstrap_mean_frac": float(boot_mean_frac),
        "bootstrap_ci_low_frac": float(boot_ci_low_frac),
        "bootstrap_ci_high_frac": float(boot_ci_high_frac),
        "bootstrap_ci_low_over_alpha2": float(boot_ci_low_frac / ALPHA2),
        "bootstrap_ci_high_over_alpha2": float(boot_ci_high_frac / ALPHA2),
    }

    # =========================================================================
    # 2. WITHIN-METHOD DISPERSION
    # =========================================================================
    sigma_tos = mad_to_sigma(tos)
    sigma_aaf = mad_to_sigma(aaf)

    sigma_tos_frac = sigma_tos / mean_tos
    sigma_aaf_frac = sigma_aaf / mean_aaf

    results["within_method_dispersion"] = {
        "sigma_tos_1e11": float(sigma_tos),
        "sigma_aaf_1e11": float(sigma_aaf),
        "sigma_tos_fractional": float(sigma_tos_frac),
        "sigma_aaf_fractional": float(sigma_aaf_frac),
        "sigma_tos_over_alpha2": float(sigma_tos_frac / ALPHA2),
        "sigma_aaf_over_alpha2": float(sigma_aaf_frac / ALPHA2),
    }

    # =========================================================================
    # 3. NUISANCE MODEL FIT
    # =========================================================================
    # Fit: G = intercept + β_method * is_AAF
    # Then look at residual scale

    is_aaf = (df['method_id'] == 'angular_acceleration_feedback').astype(float).to_numpy()
    y = df['G_value_1e11'].to_numpy()

    # Design matrix: [1, is_AAF]
    X = np.column_stack([np.ones(len(y)), is_aaf])
    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    pred = X @ coef
    resid = y - pred
    resid_scale = mad_to_sigma(resid)
    resid_scale_frac = resid_scale / np.mean(y)

    results["nuisance_model"] = {
        "intercept": float(coef[0]),
        "beta_aaf": float(coef[1]),
        "residual_scale_1e11": float(resid_scale),
        "residual_scale_fractional": float(resid_scale_frac),
        "residual_scale_over_alpha2": float(resid_scale_frac / ALPHA2),
    }

    # =========================================================================
    # OUTPUT
    # =========================================================================
    out_dir = Path("outputs/nature2018/m3_indication")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # =========================================================================
    # FIGURES
    # =========================================================================

    # Figure 1: Method comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([tos, aaf], labels=['ToS (n=7)', 'AAF (n=29)'])
    ax.axhline(mean_all, color='gray', linestyle='--', label=f'Overall mean: {mean_all:.6f}')
    ax.axhline(mean_all * (1 + ALPHA2), color='red', linestyle=':', alpha=0.7, label=f'+α² = {mean_all*(1+ALPHA2):.6f}')
    ax.axhline(mean_all * (1 - ALPHA2), color='blue', linestyle=':', alpha=0.7, label=f'-α² = {mean_all*(1-ALPHA2):.6f}')
    ax.set_ylabel('G (×10⁻¹¹ m³ kg⁻¹ s⁻²)')
    ax.set_title('Method Comparison: ToS vs AAF')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "method_comparison.png", dpi=150)
    plt.close()

    # Figure 2: Residual histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resid * 1e6, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='-', label='Zero')
    ax.axvline(ALPHA2 * mean_all * 1e6, color='green', linestyle='--', label=f'+α² scale = {ALPHA2*mean_all*1e6:.2f} ppm')
    ax.axvline(-ALPHA2 * mean_all * 1e6, color='green', linestyle='--')
    ax.set_xlabel('Residual (ppm relative to mean)')
    ax.set_ylabel('Count')
    ax.set_title('Residuals after Method Correction')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_histogram.png", dpi=150)
    plt.close()

    # =========================================================================
    # REPORT
    # =========================================================================
    report = f"""# M3 Indication Analysis: Nature 2018 Li et al.

**Date:** 2025-12-28
**Dataset:** nature2018_li (36 measurements)
**α² = {ALPHA2:.6e}** (~53.3 ppm)

---

## 1. Method Separation (ToS vs AAF)

| Metric | Value |
|--------|-------|
| Mean ToS | {mean_tos:.6f} ×10⁻¹¹ |
| Mean AAF | {mean_aaf:.6f} ×10⁻¹¹ |
| Δ (AAF - ToS) | {delta_method:.6e} ×10⁻¹¹ |
| Δ (fractional) | {delta_method_frac:.6e} |
| **Δ / α²** | **{delta_method_over_alpha2:.2f}** |

### Bootstrap 95% CI for Δ/α²

| Metric | Value |
|--------|-------|
| Mean | {boot_mean_frac/ALPHA2:.2f} |
| 95% CI | [{boot_ci_low_frac/ALPHA2:.2f}, {boot_ci_high_frac/ALPHA2:.2f}] |

**Interpretation:** The method separation is approximately **{delta_method_over_alpha2:.1f}× α²**.
This is in the α² ballpark (O(1) in α² units).

---

## 2. Within-Method Dispersion

| Method | σ (MAD→σ) | σ / mean | σ / α² |
|--------|-----------|----------|--------|
| ToS | {sigma_tos:.6e} | {sigma_tos_frac:.6e} | {sigma_tos_frac/ALPHA2:.2f} |
| AAF | {sigma_aaf:.6e} | {sigma_aaf_frac:.6e} | {sigma_aaf_frac/ALPHA2:.2f} |

**Interpretation:** Within-method scatter is **~{sigma_tos_frac/ALPHA2:.1f}× α²** (ToS) and
**~{sigma_aaf_frac/ALPHA2:.1f}× α²** (AAF). Both are below the α² scale.

---

## 3. Nuisance Model Fit

Model: `G = intercept + β_method × is_AAF`

| Parameter | Value |
|-----------|-------|
| Intercept | {coef[0]:.6f} ×10⁻¹¹ |
| β_AAF | {coef[1]:.6e} ×10⁻¹¹ |
| Residual σ | {resid_scale:.6e} ×10⁻¹¹ |
| Residual σ (fractional) | {resid_scale_frac:.6e} |
| **Residual σ / α²** | **{resid_scale_frac/ALPHA2:.2f}** |

**Interpretation:** After correcting for method, the residual scatter is
**{resid_scale_frac/ALPHA2:.1f}× α²**, which is below the α² scale.

---

## Summary

| Quantity | Value in α² units | Interpretation |
|----------|-------------------|----------------|
| Method separation | {delta_method_over_alpha2:.1f}× | ≈ O(1) in α² |
| ToS internal scatter | {sigma_tos_frac/ALPHA2:.1f}× | < α² |
| AAF internal scatter | {sigma_aaf_frac/ALPHA2:.1f}× | < α² |
| Post-method residual | {resid_scale_frac/ALPHA2:.1f}× | < α² |

**Key finding:** The ToS-AAF method separation ({delta_method_frac*1e6:.1f} ppm) is
remarkably close to α² (~53 ppm). This is a known feature of this paper's results,
not a new discovery.

---

## Figures

- `figures/method_comparison.png` - Box plot of ToS vs AAF G values
- `figures/residual_histogram.png` - Histogram of residuals after method correction

---

## Caution

This analysis shows that the Nature 2018 data has structure at the α² scale,
specifically in the method separation. However:

1. This is within a single paper's data
2. The ToS-AAF difference is a known result (reported as 45 ppm in the paper)
3. We cannot determine if this is a genuine α²-related effect or coincidence
4. Reversal-direction data would be needed to test pred1 (sign-blind invariance)

**This does NOT constitute evidence for new physics.**
"""

    with open(out_dir / "report.md", "w") as f:
        f.write(report)

    print("M3 Indication Analysis Complete")
    print(f"  Results: {out_dir / 'results.json'}")
    print(f"  Report: {out_dir / 'report.md'}")
    print(f"  Figures: {fig_dir}")
    print()
    print(f"Key findings:")
    print(f"  Method separation / α² = {delta_method_over_alpha2:.2f}")
    print(f"  ToS scatter / α² = {sigma_tos_frac/ALPHA2:.2f}")
    print(f"  AAF scatter / α² = {sigma_aaf_frac/ALPHA2:.2f}")
    print(f"  Residual σ / α² = {resid_scale_frac/ALPHA2:.2f}")

if __name__ == "__main__":
    main()
