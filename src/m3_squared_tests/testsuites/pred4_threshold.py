"""Prediction 4: threshold turn-on."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m3_squared_tests.reporting.figures import save_scatter_with_fit
from m3_squared_tests.reporting.report_md import write_report
from m3_squared_tests.stats.model_selection import bic
from m3_squared_tests.stats.regression import cv_mse, fit_linear
from m3_squared_tests.testsuites.utils import compute_residuals


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _fit_threshold(c: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    best = (float("inf"), None, None, None)
    c_grid = np.percentile(c, [20, 40, 60, 80])
    w_grid = np.linspace(np.std(c) * 0.25, np.std(c) * 1.5, 4)
    for c0 in c_grid:
        for w in w_grid:
            if w <= 0:
                continue
            s = _sigmoid((c - c0) / w) ** 2
            model = fit_linear(s[:, None], y)
            pred = model.predict(s[:, None])
            sse = float(np.sum((y - pred) ** 2))
            if sse < best[0]:
                best = (sse, c0, w, pred)
    return best[1], best[2], best[3], best[0]


def run_pred4(staged, out_dir: Path, proxy_map: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    residuals = compute_residuals(staged.merged)
    df = residuals.df.copy()

    if "quality_proxy" not in df.columns and proxy_map is not None:
        mapping = pd.read_csv(proxy_map)
        if "config_id" not in mapping.columns or "quality_proxy" not in mapping.columns:
            raise ValueError(\"proxy_map must include config_id and quality_proxy columns.\")
        df = df.merge(mapping[[\"config_id\", \"quality_proxy\"]], on=\"config_id\", how=\"left\")

    if "quality_proxy" not in df.columns:
        if "uncertainty_ppm" in df.columns:
            df["quality_proxy"] = 1.0 / df["uncertainty_ppm"].astype(float)
        else:
            result = {"test": "pred4_threshold", "status": "insufficient_data"}
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "results.json").write_text(json.dumps(result, indent=2))
            write_report(
                out_dir,
                "Prediction 4: Threshold turn-on",
                {"status": "insufficient_data"},
                ["- No quality proxy available; supply proxy_map.csv."],
            )
            return result

    df = df.dropna(subset=["quality_proxy"])
    if len(df) < 5:
        result = {"test": "pred4_threshold", "status": "insufficient_data"}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(result, indent=2))
        write_report(
            out_dir,
            "Prediction 4: Threshold turn-on",
            {"status": "insufficient_data"},
            ["- Not enough quality-proxy points."],
        )
        return result

    c = df["quality_proxy"].to_numpy(dtype=float)
    y = df["fractional_residual"].to_numpy(dtype=float)

    linear_model = fit_linear(c[:, None], y)
    quad_model = fit_linear((c**2)[:, None], y)
    linear_pred = linear_model.predict(c[:, None])
    quad_pred = quad_model.predict((c**2)[:, None])

    sse_linear = float(np.sum((y - linear_pred) ** 2))
    sse_quad = float(np.sum((y - quad_pred) ** 2))
    bic_linear = bic(len(y), sse_linear, k=2)
    bic_quad = bic(len(y), sse_quad, k=2)

    c_star, width, thresh_pred, sse_thresh = _fit_threshold(c, y)
    bic_thresh = bic(len(y), sse_thresh, k=4)

    cv_linear = cv_mse(c[:, None], y, fit_linear)
    cv_quad = cv_mse((c**2)[:, None], y, fit_linear)

    preferred = "threshold" if bic_thresh < min(bic_linear, bic_quad) else "smooth"

    save_scatter_with_fit(
        figures_dir,
        "threshold_fit",
        c,
        y,
        thresh_pred,
        "Quality proxy",
        "Fractional residual",
        "Threshold fit",
    )

    result = {
        "test": "pred4_threshold",
        "status": "ok",
        "preferred": preferred,
        "bic_linear": bic_linear,
        "bic_quadratic": bic_quad,
        "bic_threshold": bic_thresh,
        "cv_mse_linear": cv_linear,
        "cv_mse_quadratic": cv_quad,
        "c_star": float(c_star),
        "width": float(width),
    }

    summary = {
        "preferred": preferred,
        "c_star": f"{c_star:.3e}",
        "width": f"{width:.3e}",
    }
    sections = [
        f"- Smooth models: BIC linear {bic_linear:.2f}, quadratic {bic_quad:.2f}.",
        f"- Threshold BIC {bic_thresh:.2f}.",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    write_report(out_dir, "Prediction 4: Threshold turn-on", summary, sections)
    return result
