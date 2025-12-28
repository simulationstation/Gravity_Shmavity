"""Shared utilities for test suites."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResidualData:
    residuals: np.ndarray
    fractional: np.ndarray
    df: pd.DataFrame


def compute_residuals(df: pd.DataFrame) -> ResidualData:
    df = df.copy()
    values = df["value"].astype(float).to_numpy()
    means = df.groupby("dataset_id")["value"].transform("mean").astype(float)
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
