"""Prediction 3: symmetry null suppression."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from m3_squared_tests.constants import alpha4
from m3_squared_tests.metrics.robust_scale import mad
from m3_squared_tests.reporting.figures import save_bar_plot
from m3_squared_tests.reporting.report_md import write_report
from m3_squared_tests.testsuites.utils import compute_residuals, numpy_safe_json


def run_pred3(staged, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    residuals = compute_residuals(staged.merged)
    df = residuals.df.copy()
    df["method_type"] = df.get("method_type", df.get("method_id", "unknown"))
    df["geometry_mode"] = df.get("geometry_mode", "unknown")

    group_cols = ["dataset_id", "method_type", "geometry_mode"]
    suppression = []
    labels = []
    nulled_values = []
    for key, group in df.groupby(group_cols):
        if group["config_id"].nunique() < 2:
            continue
        raw_amp = float(np.mean(np.abs(group["fractional_residual"])))
        nulled = float(np.abs(group["fractional_residual"].mean()))
        nulled_values.append(nulled)
        suppression.append(raw_amp / max(nulled, 1e-12))
        labels.append("|".join(str(k) for k in key))

    if not suppression:
        result = {"test": "pred3_symmetry_null", "status": "insufficient_data"}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(numpy_safe_json(result))
        write_report(
            out_dir,
            "Prediction 3: Symmetry null suppression",
            {"status": "insufficient_data"},
            ["- Not enough symmetric pairs for nulling."],
        )
        return result

    suppression = np.array(suppression)
    nulled_values = np.array(nulled_values)
    nulled_scale = mad(nulled_values)
    alpha4_target = alpha4()

    result = {
        "test": "pred3_symmetry_null",
        "status": "ok",
        "median_suppression": float(np.median(suppression)),
        "nulled_scale": float(nulled_scale),
        "alpha4_target": alpha4_target,
        "scale_over_alpha4": float(nulled_scale / alpha4_target) if alpha4_target > 0 else float("nan"),
    }

    save_bar_plot(
        figures_dir,
        "suppression_factors",
        labels,
        suppression.tolist(),
        "Suppression factors",
        "Raw / nulled",
    )

    summary = {
        "median_suppression": f"{result['median_suppression']:.2f}",
        "nulled_scale": f"{nulled_scale:.3e}",
    }
    sections = [
        "- Suppression computed as mean(|raw|) / |mean(nulled)|.",
        f"- Nulled MAD scale vs alpha4 ({alpha4_target:.3e}).",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(numpy_safe_json(result))
    write_report(out_dir, "Prediction 3: Symmetry null suppression", summary, sections)
    return result
