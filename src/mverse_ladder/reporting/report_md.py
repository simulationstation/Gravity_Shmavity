"""Markdown report generation."""
from __future__ import annotations

from pathlib import Path
from typing import List


def write_report(
    out_path: Path,
    preferred_k: int,
    delta_bic: List[float],
    ppc_value: float,
    threshold: float,
    cv_error: float,
) -> None:
    lines = [
        "# Ladder model report",
        "",
        f"Preferred k: **{preferred_k}**",
        "",
        "## Evidence",
        "",
        "| k | ΔBIC vs M0 |",
        "|---|---|",
    ]
    for k, dbic in enumerate(delta_bic):
        lines.append(f"| {k} | {dbic:.3f} |")
    lines.extend(
        [
            "",
            "## Decision rules",
            f"ΔBIC threshold: {threshold:.3f}",
            f"Posterior predictive p-value: {ppc_value:.3f}",
            f"Cross-validated error (MAE): {cv_error:.3f}",
            "",
            "Stop rule applied: stop when ΔBIC improvement ≤ threshold.",
        ]
    )
    out_path.write_text("\n".join(lines))
