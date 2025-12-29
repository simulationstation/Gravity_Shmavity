#!/usr/bin/env python3
"""Inspect XLSX files and extract sheet structures."""
import sys
import pandas as pd
from pathlib import Path

def inspect_xlsx(filepath):
    """Inspect an XLSX file and print structure."""
    print(f"\n{'='*60}")
    print(f"FILE: {filepath}")
    print(f"{'='*60}")

    try:
        xl = pd.ExcelFile(filepath)
        print(f"Sheets: {xl.sheet_names}")

        for sheet in xl.sheet_names:
            print(f"\n--- Sheet: '{sheet}' ---")
            df = pd.read_excel(filepath, sheet_name=sheet, header=None, nrows=10)
            print(f"Shape (first 10 rows): {df.shape}")
            print("First 10 rows (raw):")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(df.to_string())
            print()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    for fp in sys.argv[1:]:
        inspect_xlsx(fp)
