"""Shared utilities for test suites."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


def numpy_safe_json(obj: Any) -> str:
    """JSON dumps with numpy type handling."""
    def _default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    return json.dumps(obj, indent=2, default=_default)


@dataclass(frozen=True)
class ResidualData:
    residuals: np.ndarray
    fractional: np.ndarray
    df: pd.DataFrame


def compute_residuals(df: pd.DataFrame) -> ResidualData:
    df = df.copy()
    # Handle both column naming conventions
    value_col = "value" if "value" in df.columns else "G_value_1e11"
    values = df[value_col].astype(float).to_numpy()
    means = df.groupby("dataset_id")[value_col].transform("mean").astype(float)
    residuals = values - means.to_numpy()
    fractional = residuals / means.to_numpy()
    df["residual"] = residuals
    df["fractional_residual"] = fractional
    return ResidualData(residuals=residuals, fractional=fractional, df=df)


def safe_group_mean(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols)[value_col].mean().reset_index()
    return grouped


def ensure_column(df: pd.DataFrame, name: str, default: Optional[str] = None) -> pd.DataFrame:
    if name in df.columns:
        return df
    if default is None:
        df[name] = "unknown"
    else:
        df[name] = default
    return df
