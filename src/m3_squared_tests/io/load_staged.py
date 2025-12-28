"""Load staged CSV data for M3 squared tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import validate_schema


@dataclass(frozen=True)
class StagedData:
    measurements: pd.DataFrame
    configs: pd.DataFrame
    merged: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing staged file: {path}")
    return pd.read_csv(path)


def _coalesce_values(df: pd.DataFrame) -> pd.Series:
    if "G_value_1e11" in df.columns:
        return df["G_value_1e11"].astype(float)
    return df["G_value"].astype(float)


def load_staged(staged_dir: str | Path) -> StagedData:
    staged_path = Path(staged_dir)
    measurements = _read_csv(staged_path / "g_measurements_minimal.csv")
    configs = _read_csv(staged_path / "g_configs_minimal.csv")
    result = validate_schema(measurements, configs)
    if not result.ok:
        raise ValueError("; ".join(result.errors))
    measurements = measurements.copy()
    measurements["value"] = _coalesce_values(measurements)
    merged = measurements.merge(configs, on=["dataset_id", "config_id"], how="left")
    return StagedData(measurements=measurements, configs=configs, merged=merged)


def load_mapping(path: Optional[str | Path], key: str, value: str) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    mapping = pd.read_csv(path)
    if key not in mapping.columns or value not in mapping.columns:
        raise ValueError(f"Mapping file must include columns {key} and {value}.")
    return mapping[[key, value]].copy()
