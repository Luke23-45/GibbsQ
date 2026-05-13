"""
Figure generation for the mismatch study (H3).

Produces a 2-panel figure showing:
- Positive boundary term count per system
- Maximum CTMC-ODE drift gap per system
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from analysis.common.io import find_latest_csv, read_csv_rows
from analysis.common.style import save_figure, get_status_color

logger = logging.getLogger(__name__)


def fig_boundary_mismatch(
    data_dir: Path,
    figure_dir: Path,
) -> List[Path]:
    """Generate the boundary-mismatch demonstration figure.

    Shows the decomposition of (L H)(Q) into interior, boundary,
    and remainder terms for a representative system.
    """
    import matplotlib.pyplot as plt

    csv_path = find_latest_csv(data_dir, "ctmc_boundary_mismatch_summary")
    if csv_path is None:
        logger.warning("No ctmc_boundary_mismatch_summary CSV found — skipping")
        return []

    rows = read_csv_rows(csv_path)
    systems = [r["system_id"] for r in rows]
    positive_counts = [int(r["positive_boundary_count"]) for r in rows]
    max_gaps = [float(r["max_gap"]) for r in rows]
    boundary_counts = [int(r["boundary_states"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(systems))
    colors_gap = ["#d62728" if g > 0.01 else "#2ca02c" for g in max_gaps]

    # Panel 1: Positive boundary term count
    axes[0].bar(x, positive_counts, color="#e07020", edgecolor="#333", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(systems, rotation=25, ha="right")
    axes[0].set_ylabel("States with positive boundary term")
    axes[0].set_title("Boundary Obstruction Count")

    # Panel 2: Max gap
    axes[1].bar(x, max_gaps, color=colors_gap, edgecolor="#333", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(systems, rotation=25, ha="right")
    axes[1].set_ylabel("max(CTMC drift − reflected-ODE drift)")
    axes[1].set_title("Generator–ODE Drift Gap")

    fig.suptitle(
        "CTMC Generator Boundary Mismatch (H3)\n"
        "Demonstrates why the old H-based proof route fails",
        fontsize=13, y=1.04,
    )
    fig.tight_layout()
    out_path = figure_dir / "h3_boundary_mismatch_demo"
    save_figure(fig, out_path)
    return [out_path]
