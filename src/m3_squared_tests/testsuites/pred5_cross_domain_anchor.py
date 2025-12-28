"""Prediction 5: cross-domain alpha2 anchor (exploratory)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from m3_squared_tests.constants import ALPHA2
from m3_squared_tests.metrics.robust_scale import mad
from m3_squared_tests.reporting.figures import save_bar_plot
from m3_squared_tests.reporting.report_md import write_report
from m3_squared_tests.testsuites.utils import compute_residuals


def _closeness(scales: np.ndarray, anchor: float) -> float:
    return float(np.mean(np.abs(np.log10(scales / anchor))))


def run_pred5(staged, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    residuals = compute_residuals(staged.merged)
    df = residuals.df.copy()

    groups = df.groupby("dataset_id")
    scales = {}
    for dataset, group in groups:
        if len(group) < 3:
            continue
        scale = mad(group["fractional_residual"].to_numpy())
        scales[dataset] = scale

    if len(scales) < 2:
        result = {"test": "pred5_cross_domain_anchor", "status": "insufficient_data"}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(result, indent=2))
        write_report(
            out_dir,
            "Prediction 5: Cross-domain anchor",
            {"status": "insufficient_data"},
            ["- Not enough dataset groups for anchor comparison."],
        )
        return result

    datasets = list(scales.keys())
    scale_values = np.array([scales[d] for d in datasets])
    ratios = scale_values / ALPHA2

    rng = np.random.default_rng(0)
    anchors = 10 ** rng.uniform(-6, -3, size=1000)
    actual = _closeness(scale_values, ALPHA2)
    permuted = np.array([_closeness(scale_values, anchor) for anchor in anchors])
    p_value = float(np.mean(permuted <= actual))

    save_bar_plot(
        figures_dir,
        "anchor_ratios",
        datasets,
        ratios.tolist(),
        "Scale / alpha2 ratios",
        "Ratio",
    )

    result = {
        "test": "pred5_cross_domain_anchor",
        "status": "ok",
        "datasets": datasets,
        "scale_values": scale_values.tolist(),
        "ratio_to_alpha2": ratios.tolist(),
        "closeness": actual,
        "permutation_p": p_value,
        "note": "Exploratory anchor comparison; not evidence of new physics.",
    }

    summary = {
        "datasets": ", ".join(datasets),
        "permutation_p": f"{p_value:.3f}",
    }
    sections = [
        "- Uses MAD scale of fractional residuals per dataset.",
        f"- Closeness to alpha2 compared against log-uniform anchors (1e-6..1e-3).",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    write_report(out_dir, "Prediction 5: Cross-domain anchor", summary, sections)
    return result
