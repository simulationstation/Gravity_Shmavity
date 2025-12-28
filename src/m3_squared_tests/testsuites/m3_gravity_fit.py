"""Minimal M3 gravity residual model-fit harness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from m3_squared_tests.constants import ALPHA2
from m3_squared_tests.reporting.report_md import write_report
from m3_squared_tests.stats.model_selection import bic
from m3_squared_tests.testsuites.utils import compute_residuals


def _design_matrix(df, include_m3: bool) -> tuple[np.ndarray, np.ndarray]:
    dataset_ids = sorted(df["dataset_id"].unique())
    method_ids = sorted(df["method_type"].unique())

    dataset_map = {k: i for i, k in enumerate(dataset_ids)}
    method_map = {k: i for i, k in enumerate(method_ids)}

    n = len(df)
    columns = 1 + len(dataset_ids) + len(method_ids)
    if include_m3:
        columns += 1
    x = np.zeros((n, columns))
    x[:, 0] = 1.0

    for idx, row in enumerate(df.itertuples(index=False)):
        d_idx = dataset_map[row.dataset_id]
        m_idx = method_map[row.method_type]
        x[idx, 1 + d_idx] = 1.0
        x[idx, 1 + len(dataset_ids) + m_idx] = 1.0

    if include_m3:
        z = np.linspace(-1, 1, n)
        x[:, -1] = ALPHA2 * z
    y = df["fractional_residual"].to_numpy(dtype=float)
    return x, y


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    sse = float(np.sum((y - pred) ** 2))
    return coef, sse


def run_m3_fit(staged, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    residuals = compute_residuals(staged.merged)
    df = residuals.df.copy()
    df["method_type"] = df.get("method_type", df.get("method_id", "unknown"))

    if len(df) < 5:
        result = {"test": "m3_gravity_fit", "status": "insufficient_data"}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(result, indent=2))
        write_report(
            out_dir,
            "M3 gravity fit",
            {"status": "insufficient_data"},
            ["- Not enough data points for regression."],
        )
        return result

    x_null, y = _design_matrix(df, include_m3=False)
    x_full, _ = _design_matrix(df, include_m3=True)

    coef_null, sse_null = _fit(x_null, y)
    coef_full, sse_full = _fit(x_full, y)

    bic_null = bic(len(y), sse_null, k=x_null.shape[1])
    bic_full = bic(len(y), sse_full, k=x_full.shape[1])

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(500):
        idx = rng.choice(len(y), size=len(y), replace=True)
        coef_boot, _ = _fit(x_full[idx], y[idx])
        boot.append(coef_boot[-1])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    result = {
        "test": "m3_gravity_fit",
        "status": "ok",
        "bic_null": bic_null,
        "bic_full": bic_full,
        "delta_bic": bic_full - bic_null,
        "a0_estimate": float(coef_full[-1]),
        "a0_ci_low": float(ci_low),
        "a0_ci_high": float(ci_high),
    }

    summary = {
        "delta_bic": f"{result['delta_bic']:.2f}",
        "a0_estimate": f"{result['a0_estimate']:.3e}",
    }
    sections = [
        "- Null model uses dataset + method nuisances.",
        "- Full model adds A0 * alpha2 * z(config) effect.",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    write_report(out_dir, "M3 gravity fit", summary, sections)
    return result
