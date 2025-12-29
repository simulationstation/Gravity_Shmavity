#!/usr/bin/env python3
"""
Quick "is this enough?" check for reversal-separated data.

Outputs:
- Number of datasets acquired
- Number of reversal-separated rows
- Number of reversal pairs
- Typical uncertainty scale (ppm)
- Whether we can run pred1 meaningfully
"""
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("REVERSAL DATA READINESS CHECK")
    print("=" * 60)
    print()

    # Load the staged data
    pred1_path = Path("data/staged/reversal_pred1.csv")
    pairs_path = Path("data/staged/reversal_pred1_pairs.csv")

    if not pred1_path.exists():
        print("ERROR: reversal_pred1.csv not found")
        return

    # Read CSV, skipping comment lines
    df = pd.read_csv(pred1_path, comment='#')

    print("1. DATASETS ACQUIRED")
    print("-" * 40)
    datasets = df['dataset_id'].unique()
    print(f"   Number of datasets: {len(datasets)}")
    for ds in datasets:
        n = len(df[df['dataset_id'] == ds])
        print(f"   - {ds}: {n} rows")
    print()

    print("2. REVERSAL-SEPARATED ROWS")
    print("-" * 40)
    n_total = len(df)
    n_averaged = len(df[df['reversal_flag'] == 'averaged'])
    n_separated = n_total - n_averaged
    print(f"   Total rows: {n_total}")
    print(f"   Reversal-averaged: {n_averaged}")
    print(f"   Reversal-separated: {n_separated}")
    print()

    print("3. REVERSAL PAIRS")
    print("-" * 40)
    # Count actual pairs (would need opposite reversal_flags for same config/run)
    reversal_flags = df['reversal_flag'].unique()
    print(f"   Reversal flags present: {list(reversal_flags)}")

    # Try to read pairs file
    try:
        with open(pairs_path, 'r') as f:
            lines = [l for l in f.readlines() if not l.startswith('#') and l.strip()]
        n_pairs = len(lines) - 1  # subtract header
        if n_pairs < 0:
            n_pairs = 0
    except:
        n_pairs = 0

    print(f"   Number of reversal pairs: {n_pairs}")
    print()

    print("4. UNCERTAINTY SCALE")
    print("-" * 40)
    if 'G_sigma_1e11' in df.columns:
        sigma = df['G_sigma_1e11'].dropna()
        g_mean = df['G_value_1e11'].mean()
        sigma_ppm = sigma / g_mean * 1e6
        print(f"   Mean G: {g_mean:.6f} x 10^-11 m^3 kg^-1 s^-2")
        print(f"   Median uncertainty: {sigma.median():.6e} x 10^-11")
        print(f"   Median uncertainty: {sigma_ppm.median():.1f} ppm")
        print(f"   Range: [{sigma_ppm.min():.1f}, {sigma_ppm.max():.1f}] ppm")
    else:
        print("   No uncertainty column found")
    print()

    print("5. PRED1 READINESS")
    print("-" * 40)
    alpha2 = (1/137.035999084)**2
    alpha2_ppm = alpha2 * 1e6
    print(f"   Alpha^2 scale: {alpha2_ppm:.1f} ppm")

    MIN_PAIRS_REQUIRED = 5
    can_run_pred1 = n_pairs >= MIN_PAIRS_REQUIRED

    print(f"   Minimum pairs required: {MIN_PAIRS_REQUIRED}")
    print(f"   Pairs available: {n_pairs}")
    print()
    if can_run_pred1:
        print("   STATUS: READY - Can run pred1")
    else:
        print("   STATUS: NOT READY - Insufficient reversal pairs")
        print()
        print("   REASON: All publicly available G measurements are")
        print("           reversal-averaged. No CW/CCW or near/far")
        print("           separated G values are published.")
    print()

    print("6. ALTERNATIVE ANALYSES POSSIBLE")
    print("-" * 40)
    print("   a) Method-separation analysis (ToS vs AAF)")
    print("      - Nature 2018 has 7 ToS + 29 AAF measurements")
    print("      - Already analyzed: separation ~ 0.84 x alpha^2")
    print()
    print("   b) Temporal/configuration analysis")
    print("      - TR&D-96 has 26 runs over 10 years")
    print("      - Can test for temporal structure")
    print()
    print("   c) Cross-experiment analysis")
    print("      - Schlamminger compilation has 21 experiments")
    print("      - Can compare method-level structure")
    print()

    return {
        'n_datasets': len(datasets),
        'n_rows': n_total,
        'n_reversal_separated': n_separated,
        'n_pairs': n_pairs,
        'can_run_pred1': can_run_pred1,
    }


if __name__ == "__main__":
    results = main()
