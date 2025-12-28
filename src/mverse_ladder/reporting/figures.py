"""Plotting utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_evidence(delta_bic: List[float], out_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(range(len(delta_bic)), delta_bic, marker="o")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("k (model index)")
    ax.set_ylabel("ΔBIC vs M0")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_a0(a0_values: List[float], out_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(range(1, len(a0_values) + 1), a0_values, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("A0 estimate")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_layer_contrib(weights: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(range(1, len(weights) + 1), weights, marker="o", label="fitted")
    ax.set_xlabel("Layer m")
    ax.set_ylabel("Contribution scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
