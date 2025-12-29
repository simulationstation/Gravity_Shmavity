#!/usr/bin/env python3
"""Build unified reversal observables and pairs datasets."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT_DIR = Path("data/staged/nature2018_li_preavg")

def build_reversal_observables():
    """Build reversal_observables.csv from extracted tables."""
    print("Building reversal_observables.csv...")

    rows = []

    # Load extdata_fig5
    df1 = pd.read_csv(OUT_DIR / "extdata_fig5__c.csv")
    for i, row in df1.iterrows():
        # Near observation
        rows.append({
            "dataset_id": "nature2018_li",
            "method_id": "ToS",
            "observable_type": "period_T",
            "run_id": f"extdata_fig5_{row['pair_id']}",
            "condition_label": "near",
            "value": row["T_near_s"],
            "value_sigma": np.nan,  # Not provided
            "units": "s",
            "source": f"source_data_extdata_fig5.xlsx/c/row{i+2}",
            "notes": f"condition={row['condition']}, time={row['time_near_day']:.1f}d"
        })
        # Far observation
        rows.append({
            "dataset_id": "nature2018_li",
            "method_id": "ToS",
            "observable_type": "period_T",
            "run_id": f"extdata_fig5_{row['pair_id']}",
            "condition_label": "far",
            "value": row["T_far_s"],
            "value_sigma": np.nan,
            "units": "s",
            "source": f"source_data_extdata_fig5.xlsx/c/row{i+2}",
            "notes": f"condition={row['condition']}, time={row['time_far_day']:.1f}d"
        })

    # Load fig2 sheets a and b
    for sheet, fiber_id in [("a", 1), ("b", 2)]:
        df = pd.read_csv(OUT_DIR / f"fig2__{sheet}.csv")
        for i, row in df.iterrows():
            # Near observation
            rows.append({
                "dataset_id": "nature2018_li",
                "method_id": "ToS",
                "observable_type": "period_T",
                "run_id": f"fig2_{sheet}_{row['pair_id']}",
                "condition_label": "near",
                "value": row["T_near_s"],
                "value_sigma": np.nan,
                "units": "s",
                "source": f"source_data_fig2.xlsx/{sheet}/row{i+2}",
                "notes": f"fiber={fiber_id}, time={row['time_near_day']:.1f}d"
            })
            # Far observation
            rows.append({
                "dataset_id": "nature2018_li",
                "method_id": "ToS",
                "observable_type": "period_T",
                "run_id": f"fig2_{sheet}_{row['pair_id']}",
                "condition_label": "far",
                "value": row["T_far_s"],
                "value_sigma": np.nan,
                "units": "s",
                "source": f"source_data_fig2.xlsx/{sheet}/row{i+2}",
                "notes": f"fiber={fiber_id}, time={row['time_far_day']:.1f}d"
            })

    df_obs = pd.DataFrame(rows)
    df_obs.to_csv(OUT_DIR / "reversal_observables.csv", index=False)
    print(f"  Saved {len(df_obs)} observations to reversal_observables.csv")
    return df_obs


def build_reversal_pairs():
    """Build reversal_pairs.csv from observables."""
    print("Building reversal_pairs.csv...")

    rows = []

    # Process extdata_fig5
    df1 = pd.read_csv(OUT_DIR / "extdata_fig5__c.csv")
    for i, row in df1.iterrows():
        delta = row["T_near_s"] - row["T_far_s"]
        rows.append({
            "dataset_id": "nature2018_li",
            "run_id": f"extdata_fig5_{row['pair_id']}",
            "observable_type": "period_T",
            "value_A": row["T_near_s"],
            "value_B": row["T_far_s"],
            "condition_A": "near",
            "condition_B": "far",
            "delta": delta,
            "abs_delta": abs(delta),
            "sigma_delta": np.nan,
            "pairing_confidence": "HIGH",
            "source": f"extdata_fig5/c/{row['condition']}",
            "notes": f"Same-row pairing, condition={row['condition']}"
        })

    # Process fig2 sheets
    for sheet, fiber_id in [("a", 1), ("b", 2)]:
        df = pd.read_csv(OUT_DIR / f"fig2__{sheet}.csv")
        for i, row in df.iterrows():
            delta = row["T_near_s"] - row["T_far_s"]
            rows.append({
                "dataset_id": "nature2018_li",
                "run_id": f"fig2_{sheet}_{row['pair_id']}",
                "observable_type": "period_T",
                "value_A": row["T_near_s"],
                "value_B": row["T_far_s"],
                "condition_A": "near",
                "condition_B": "far",
                "delta": delta,
                "abs_delta": abs(delta),
                "sigma_delta": np.nan,
                "pairing_confidence": "HIGH",
                "source": f"fig2/{sheet}/fiber{fiber_id}",
                "notes": f"Temporal pairing (~3d offset), fiber={fiber_id}"
            })

    df_pairs = pd.DataFrame(rows)
    df_pairs.to_csv(OUT_DIR / "reversal_pairs.csv", index=False)
    print(f"  Saved {len(df_pairs)} pairs to reversal_pairs.csv")
    return df_pairs


def main():
    print("=" * 60)
    print("Building unified reversal datasets")
    print("=" * 60)

    df_obs = build_reversal_observables()
    df_pairs = build_reversal_pairs()

    print()
    print("Summary:")
    print(f"  Total observations: {len(df_obs)}")
    print(f"  Total pairs: {len(df_pairs)}")
    print(f"  HIGH confidence pairs: {(df_pairs['pairing_confidence'] == 'HIGH').sum()}")

    # Quick stats on delta
    print()
    print("Delta (T_near - T_far) statistics:")
    print(f"  Mean: {df_pairs['delta'].mean():.6f} s")
    print(f"  Std:  {df_pairs['delta'].std():.6f} s")
    print(f"  Min:  {df_pairs['delta'].min():.6f} s")
    print(f"  Max:  {df_pairs['delta'].max():.6f} s")


if __name__ == "__main__":
    main()
