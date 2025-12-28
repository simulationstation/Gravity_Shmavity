"""Lightweight schema validation for staged CSVs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SchemaResult:
    ok: bool
    errors: list[str]


MEASUREMENTS_REQUIRED = {"dataset_id", "config_id"}
MEASUREMENTS_OPTIONAL_VALUE = ["G_value_1e11", "G_value"]

CONFIGS_REQUIRED = {"dataset_id", "config_id"}


def _missing_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [col for col in required if col not in df.columns]


def validate_measurements(df: pd.DataFrame) -> SchemaResult:
    errors: list[str] = []
    missing = _missing_columns(df, MEASUREMENTS_REQUIRED)
    if missing:
        errors.append(f"Missing required measurement columns: {missing}")
    if not any(col in df.columns for col in MEASUREMENTS_OPTIONAL_VALUE):
        errors.append(
            "Measurement data must include at least one of G_value_1e11 or G_value."
        )
    return SchemaResult(ok=not errors, errors=errors)


def validate_configs(df: pd.DataFrame) -> SchemaResult:
    errors: list[str] = []
    missing = _missing_columns(df, CONFIGS_REQUIRED)
    if missing:
        errors.append(f"Missing required config columns: {missing}")
    return SchemaResult(ok=not errors, errors=errors)


def validate_schema(measurements: pd.DataFrame, configs: pd.DataFrame) -> SchemaResult:
    errors: list[str] = []
    result_measurements = validate_measurements(measurements)
    result_configs = validate_configs(configs)
    errors.extend(result_measurements.errors)
    errors.extend(result_configs.errors)
    return SchemaResult(ok=not errors, errors=errors)
