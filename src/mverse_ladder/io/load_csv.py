"""Load CSV data for gravity configs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_gravity_csv(path: str) -> dict:
    df = pd.read_csv(path)
    config_cols = [c for c in df.columns if c.startswith("config_")]
    return {
        "y": df["y"].to_numpy(),
        "temp": df["temp"].to_numpy(),
        "geom": df["geom"].to_numpy(),
        "config": df[config_cols].to_numpy(),
    }


def load_timeseries_npz(path: str) -> dict:
    data = np.load(path)
    return {"t": data["t"], "ch1": data["ch1"], "ch2": data["ch2"]}
