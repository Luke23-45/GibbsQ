"""
Figure generation for the convergence study (H1/H2).

Produces multi-panel convergence summary showing:
- Terminal pairwise diameter (attractor uniqueness)
- Maximum residual norm at terminal time
- Convergence rate across systems
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from analysis.common.io import find_latest_csv, read_csv_rows
from analysis.common.style import save_figure, get_status_color

logger = logging.getLogger(__name__)


def fig_reflected_ode_convergence(
    data_dir: Path,
    figure_dir: Path,
) -> List[Path]:
    """Generate the reflected-ODE convergence summary figure.

    Produces a 3-panel figure showing convergence metrics per system.
    """
    import matplotlib.pyplot as plt

    csv_path = find_latest_csv(data_dir, "reflected_ode_convergence_summary")
    if csv_path is None:
        logger.warning("No reflected_ode_convergence_summary CSV found — skipping")
        return []

    rows = read_csv_rows(csv_path)
    systems = [r["system_id"] for r in rows]
    diameters = [float(r["terminal_diameter"]) for r in rows]
    residuals = [float(r["max_residual_norm"]) for r in rows]
    conv_rates = [float(r["convergence_rate"]) for r in rows]
    statuses = [r["status"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Terminal diameter
    colors = [get_status_color(s) for s in statuses]
    x = np.arange(len(systems))
    axes[0].bar(x, diameters, color=colors, edgecolor="#333", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(systems, rotation=25, ha="right")
    axes[0].set_ylabel("Terminal pairwise diameter")
    axes[0].set_title("Attractor Uniqueness")
    axes[0].set_yscale("symlog", linthresh=1e-16)

    # Panel 2: Max residual norm
    axes[1].bar(x, residuals, color=colors, edgecolor="#333", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(systems, rotation=25, ha="right")
    axes[1].set_ylabel(r"$\max \|\dot{q}(T)\|_\infty$")
    axes[1].set_title("Residual at Terminal Time")
    axes[1].set_yscale("symlog", linthresh=1e-16)

    # Panel 3: Convergence rate
    axes[2].bar(x, conv_rates, color=colors, edgecolor="#333", alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(systems, rotation=25, ha="right")
    axes[2].set_ylabel("Convergence rate")
    axes[2].set_title("Fraction Converged")

    fig.suptitle("Reflected-ODE Multi-Start Convergence (H1/H2)", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = figure_dir / "h1h2_reflected_ode_convergence"
    save_figure(fig, out_path)
    return [out_path]
