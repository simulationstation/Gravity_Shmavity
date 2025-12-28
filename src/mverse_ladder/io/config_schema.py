"""Pydantic configuration schemas."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LadderConfig(BaseModel):
    alpha: float = Field(default=1 / 137.035999084)
    include_m1: bool = True


class GravitySimConfig(BaseModel):
    n: int = 200
    config_dim: int = 3
    max_m: int = 6
    a0: float = 1.0
    sigma_latent: float = 0.5
    sigma_noise: float = 0.3
    seed: int = 123


class TimeSeriesSimConfig(BaseModel):
    n: int = 2000
    dt: float = 0.1
    max_m: int = 6
    a0: float = 1.0
    tau_base: float = 2.0
    sigma_latent: float = 0.3
    sigma_noise: float = 0.2
    seed: int = 123


class FitConfig(BaseModel):
    max_k: int = 10
    include_m1: bool = True
    alpha: float = 1 / 137.035999084
    seed: int = 123
    model_type: Literal["gravity", "timeseries"]
    nested_sampling: bool = False
    bootstrap_ppc: int = 200


class CalibrateConfig(BaseModel):
    alpha: float = 0.05
    runs: int = 100
    preset: Literal["gravity_synth", "timeseries_synth"] = "gravity_synth"
    max_k: int = 6
    include_m1: bool = True
    seed: int = 123
    nested_sampling: bool = False
    out: Optional[str] = None
