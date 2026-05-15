"""
Figure generation for the equilibrium study (H2).

Produces a bar chart showing max_discrepancy per system, with a
horizontal tolerance line.  Systems that PASS are green; FAIL are red.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from studies.analysis.common.io import find_latest_csv, read_csv_rows
from studies.analysis.common.style import save_figure, get_status_color

logger = logging.getLogger(__name__)


def fig_boundary_equilibrium(
    data_dir: Path,
    figure_dir: Path,
) -> List[Path]:
    """Generate the boundary-equilibrium verification figure."""
    import matplotlib.pyplot as plt

    csv_path = find_latest_csv(data_dir, "boundary_equilibrium_verification")
    if csv_path is None:
        logger.warning("No boundary_equilibrium_verification CSV found — skipping")
        return []

    rows = read_csv_rows(csv_path)
    systems = [r["system_id"] for r in rows]
    discrepancies = [float(r["max_discrepancy"]) for r in rows]
    statuses = [r["status"] for r in rows]
    colors = [get_status_color(s) for s in statuses]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(systems))
    bars = ax.bar(x, discrepancies, color=colors, edgecolor="#333", alpha=0.85)
    ax.axhline(1e-4, color="#888", linestyle="--", linewidth=1, label="Tolerance (1e-4)")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=25, ha="right")
    ax.set_ylabel(r"$\max_i |q^*_{\mathrm{exact}} - q^*_{\mathrm{ODE}}|$")
    ax.set_title("Boundary Equilibrium Verification (H2)")
    ax.set_yscale("log")
    ax.legend(loc="upper right")

    # Annotate status
    for bar, status in zip(bars, statuses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            status,
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    fig.tight_layout()
    out_path = figure_dir / "h2_boundary_equilibrium_verification"
    save_figure(fig, out_path)
    return [out_path]
