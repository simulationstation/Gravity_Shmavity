"""Prediction 1: sign-blind intensity law (reversal invariance)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from m3_squared_tests.constants import ALPHA2
from m3_squared_tests.reporting.figures import save_paired_plot
from m3_squared_tests.reporting.report_md import write_report
from m3_squared_tests.stats.invariance import bootstrap_equivalence
from m3_squared_tests.testsuites.utils import compute_residuals, numpy_safe_json


def run_pred1(staged, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    residuals = compute_residuals(staged.merged)
    df = residuals.df.copy()
    df["reversal_indicator"] = df.get("reversal_indicator", "unknown")
    df["method_type"] = df.get("method_type", df.get("method_id", "unknown"))
    df["geometry_mode"] = df.get("geometry_mode", "unknown")

    group_cols = ["dataset_id", "method_type", "geometry_mode"]
    pairs = []
    for _, group in df.groupby(group_cols):
        categories = sorted(group["reversal_indicator"].unique())
        if len(categories) < 2:
            continue
        cat_a, cat_b = categories[:2]
        mag_a = np.mean(np.abs(group.loc[group["reversal_indicator"] == cat_a, "fractional_residual"]))
        mag_b = np.mean(np.abs(group.loc[group["reversal_indicator"] == cat_b, "fractional_residual"]))
        label = f"{group.iloc[0]['dataset_id']}|{group.iloc[0]['method_type']}"
        pairs.append((label, mag_a, mag_b))

    result = {
        "test": "pred1_sign_blind",
        "tolerance": ALPHA2,
        "pairs": len(pairs),
        "status": "ok" if pairs else "insufficient_data",
    }
    sections = []

    if pairs:
        labels, mags_a, mags_b = zip(*pairs)
        mags_a = np.array(mags_a)
        mags_b = np.array(mags_b)
        equiv = bootstrap_equivalence(mags_a, mags_b, tolerance=ALPHA2)
        result.update(
            {
                "mean_diff": equiv.mean_diff,
                "ci_low": equiv.ci_low,
                "ci_high": equiv.ci_high,
                "within_tolerance": equiv.within_tolerance,
            }
        )
        save_paired_plot(figures_dir, "paired_magnitudes", labels, mags_a, mags_b, "Paired magnitudes")
        sections.append(
            "- Paired comparison uses groupings by dataset/method/geometry; "
            "magnitudes are absolute fractional residuals."
        )
        sections.append(
            f"- Equivalence CI: [{equiv.ci_low:.3e}, {equiv.ci_high:.3e}] vs tolerance {ALPHA2:.3e}."
        )
    else:
        sections.append("- Insufficient paired reversal groupings to compute equivalence.")

    summary = {
        "status": result["status"],
        "pairs": result["pairs"],
        "tolerance": f"{ALPHA2:.3e}",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(numpy_safe_json(result))
    write_report(out_dir, "Prediction 1: Sign-blind intensity law", summary, sections)
    return result
