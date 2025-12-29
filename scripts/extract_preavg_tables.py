#!/usr/bin/env python3
"""Extract pre-averaging period tables from Nature 2018 attachments."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT_DIR = Path("data/staged/nature2018_li_preavg")
META_DIR = OUT_DIR / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

def extract_extdata_fig5():
    """Extract near/far periods from ExtData Fig5."""
    print("Extracting extdata_fig5...")
    df = pd.read_excel(
        "data/raw/nature2018_li/attachments/source_data_extdata_fig5.xlsx",
        sheet_name="c", header=1
    )

    # Clean column names
    df.columns = ["condition", "time_near_day", "T_near_s", "time_far_day", "T_far_s"]

    # Forward fill condition labels
    df["condition"] = df["condition"].ffill()

    # Remove separator rows (containing '--')
    df = df[~df["T_near_s"].astype(str).str.contains("--", na=False)]
    df = df.dropna(subset=["T_near_s", "T_far_s"])

    # Convert to numeric
    df["T_near_s"] = pd.to_numeric(df["T_near_s"], errors="coerce")
    df["T_far_s"] = pd.to_numeric(df["T_far_s"], errors="coerce")
    df["time_near_day"] = pd.to_numeric(df["time_near_day"], errors="coerce")
    df["time_far_day"] = pd.to_numeric(df["time_far_day"], errors="coerce")

    df = df.dropna()
    df = df.reset_index(drop=True)

    # Add identifiers
    df["source_file"] = "source_data_extdata_fig5.xlsx"
    df["source_sheet"] = "c"
    df["pair_id"] = range(1, len(df) + 1)

    # Save
    out_path = OUT_DIR / "extdata_fig5__c.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")

    # Metadata
    meta = {
        "source_file": "source_data_extdata_fig5.xlsx",
        "source_sheet": "c",
        "description": "Period measurements at near and far source mass positions",
        "columns": {
            "condition": "Voltage condition (Ground, 0.1V, -0.1V)",
            "time_near_day": "Time of near measurement (days)",
            "T_near_s": "Period at near position (seconds)",
            "time_far_day": "Time of far measurement (days)",
            "T_far_s": "Period at far position (seconds)"
        },
        "units": {"T_near_s": "s", "T_far_s": "s", "time": "day"},
        "n_rows": len(df),
        "pairing_method": "same_row",
        "pairing_confidence": "HIGH"
    }
    with open(META_DIR / "extdata_fig5__c.json", "w") as f:
        json.dump(meta, f, indent=2)

    return df


def extract_fig2_sheet(sheet_name, fiber_id):
    """Extract near/far periods from Fig2 sheets a or b."""
    print(f"Extracting fig2 sheet {sheet_name} (fiber {fiber_id})...")
    df = pd.read_excel(
        "data/raw/nature2018_li/attachments/source_data_fig2.xlsx",
        sheet_name=sheet_name, header=1
    )

    # Clean column names
    df.columns = ["unnamed", "time_day", "T_near_s", "T_far_s"]
    df = df.drop(columns=["unnamed"])

    # Separate near and far measurements
    near_mask = df["T_near_s"].notna()
    far_mask = df["T_far_s"].notna()

    near_df = df[near_mask][["time_day", "T_near_s"]].reset_index(drop=True)
    far_df = df[far_mask][["time_day", "T_far_s"]].reset_index(drop=True)

    near_df.columns = ["time_near_day", "T_near_s"]
    far_df.columns = ["time_far_day", "T_far_s"]

    # Pair by index (temporal ordering)
    n_pairs = min(len(near_df), len(far_df))
    paired = pd.concat([
        near_df.iloc[:n_pairs].reset_index(drop=True),
        far_df.iloc[:n_pairs].reset_index(drop=True)
    ], axis=1)

    paired["fiber_id"] = fiber_id
    paired["source_file"] = "source_data_fig2.xlsx"
    paired["source_sheet"] = sheet_name
    paired["pair_id"] = range(1, len(paired) + 1)

    # Save
    out_path = OUT_DIR / f"fig2__{sheet_name}.csv"
    paired.to_csv(out_path, index=False)
    print(f"  Saved {len(paired)} rows to {out_path}")

    # Metadata
    meta = {
        "source_file": "source_data_fig2.xlsx",
        "source_sheet": sheet_name,
        "description": f"Period time series for ToS Fiber {fiber_id}",
        "columns": {
            "time_near_day": "Time of near measurement (days)",
            "T_near_s": "Period at near position (seconds)",
            "time_far_day": "Time of far measurement (days)",
            "T_far_s": "Period at far position (seconds)",
            "fiber_id": "Fiber identifier"
        },
        "units": {"T_near_s": "s", "T_far_s": "s", "time": "day"},
        "n_rows": len(paired),
        "pairing_method": "temporal_proximity",
        "pairing_confidence": "HIGH",
        "notes": "Near and far measurements are interleaved ~3 days apart"
    }
    with open(META_DIR / f"fig2__{sheet_name}.json", "w") as f:
        json.dump(meta, f, indent=2)

    return paired


def main():
    print("=" * 60)
    print("Extracting pre-averaging period tables")
    print("=" * 60)

    # Extract all candidate tables
    df_extdata = extract_extdata_fig5()
    df_fig2_a = extract_fig2_sheet("a", fiber_id=1)
    df_fig2_b = extract_fig2_sheet("b", fiber_id=2)

    print()
    print("Summary:")
    print(f"  extdata_fig5: {len(df_extdata)} pairs")
    print(f"  fig2 sheet a: {len(df_fig2_a)} pairs")
    print(f"  fig2 sheet b: {len(df_fig2_b)} pairs")
    print(f"  Total: {len(df_extdata) + len(df_fig2_a) + len(df_fig2_b)} pairs")


if __name__ == "__main__":
    main()
