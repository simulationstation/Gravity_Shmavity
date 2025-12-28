"""Prediction 2: topology-squared scaling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m3_squared_tests.constants import ALPHA2
from m3_squared_tests.reporting.figures import save_scatter_with_fit
from m3_squared_tests.reporting.report_md import write_report
from m3_squared_tests.stats.model_selection import bic
from m3_squared_tests.stats.regression import cv_mse, fit_linear
from m3_squared_tests.testsuites.utils import compute_residuals, safe_group_mean


def _bootstrap_c(x: np.ndarray, y: np.ndarray, samples: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    coefs = []
    for _ in range(samples):
        idx = rng.choice(len(y), size=len(y), replace=True)
        model = fit_linear(x[idx], y[idx])
        coefs.append(model.coef[0])
    return float(np.percentile(coefs, 2.5)), float(np.percentile(coefs, 97.5))


def run_pred2(staged, out_dir: Path, topology_map: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    residuals = compute_residuals(staged.merged)
    df = residuals.df

    if "topology_proxy" not in df.columns and topology_map is not None:
        mapping = pd.read_csv(topology_map)
        if "config_id" not in mapping.columns or "topology_proxy" not in mapping.columns:
            raise ValueError("topology_map must include config_id and topology_proxy columns.")
        df = df.copy()
        df = df.merge(mapping[["config_id", "topology_proxy"]], on="config_id", how="left")

    if "topology_proxy" not in df.columns:
        result = {"test": "pred2_topology_sq", "status": "insufficient_data"}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(result, indent=2))
        write_report(
            out_dir,
            "Prediction 2: Topology-squared scaling",
            {"status": "insufficient_data"},
            ["- No topology proxy available; provide topology_map.csv."],
        )
        return result

    grouped = safe_group_mean(df, ["config_id", "topology_proxy"], "fractional_residual")
    grouped = grouped.dropna(subset=["topology_proxy"])
    if len(grouped) < 3:
        result = {"test": "pred2_topology_sq", "status": "insufficient_data"}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(result, indent=2))
        write_report(
            out_dir,
            "Prediction 2: Topology-squared scaling",
            {"status": "insufficient_data"},
            ["- Not enough topology groups for regression."],
        )
        return result

    x = grouped["topology_proxy"].to_numpy(dtype=float)
    y = grouped["fractional_residual"].to_numpy(dtype=float)

    linear_model = fit_linear(x[:, None], y)
    quad_model = fit_linear((x**2)[:, None], y)
    linear_pred = linear_model.predict(x[:, None])
    quad_pred = quad_model.predict((x**2)[:, None])

    sse_linear = float(np.sum((y - linear_pred) ** 2))
    sse_quad = float(np.sum((y - quad_pred) ** 2))
    bic_linear = bic(len(y), sse_linear, k=2)
    bic_quad = bic(len(y), sse_quad, k=2)

    cv_linear = cv_mse(x[:, None], y, fit_linear)
    cv_quad = cv_mse((x**2)[:, None], y, fit_linear)

    ci_low, ci_high = _bootstrap_c((x**2)[:, None], y)
    c_est = float(quad_model.coef[0])

    preferred = "quadratic" if bic_quad < bic_linear else "linear"

    result = {
        "test": "pred2_topology_sq",
        "status": "ok",
        "preferred": preferred,
        "bic_linear": bic_linear,
        "bic_quadratic": bic_quad,
        "cv_mse_linear": cv_linear,
        "cv_mse_quadratic": cv_quad,
        "c_estimate": c_est,
        "c_over_alpha2": c_est / ALPHA2,
        "c_ci_low": ci_low,
        "c_ci_high": ci_high,
    }

    save_scatter_with_fit(
        figures_dir,
        "topology_quadratic_fit",
        x,
        y,
        quad_pred,
        "Topology proxy T",
        "Fractional residual",
        "Quadratic fit",
    )

    summary = {
        "preferred": preferred,
        "c/alpha2": f"{c_est / ALPHA2:.2f}",
    }
    sections = [
        f"- Linear BIC: {bic_linear:.2f}; Quadratic BIC: {bic_quad:.2f}.",
        f"- Quadratic coefficient CI: [{ci_low:.3e}, {ci_high:.3e}].",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    write_report(out_dir, "Prediction 2: Topology-squared scaling", summary, sections)
    return result
