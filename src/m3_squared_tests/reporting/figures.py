"""Figure generation for reports (matplotlib only)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def save_paired_plot(
    out_dir: Path,
    name: str,
    labels: Iterable[str],
    values_a: np.ndarray,
    values_b: np.ndarray,
    title: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(len(labels))
    ax.scatter(idx - 0.1, values_a, label="A", color="tab:blue")
    ax.scatter(idx + 0.1, values_b, label="B", color="tab:orange")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_scatter_with_fit(
    out_dir: Path,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    y_fit: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color="tab:blue", label="data")
    order = np.argsort(x)
    ax.plot(x[order], y_fit[order], color="tab:red", label="fit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_bar_plot(
    out_dir: Path,
    name: str,
    labels: Iterable[str],
    values: Iterable[float],
    title: str,
    ylabel: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(labels)
    values = list(values)
    x = np.arange(len(labels))
    ax.bar(x, values, color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
