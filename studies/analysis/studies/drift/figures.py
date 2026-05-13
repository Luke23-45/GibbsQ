"""
Figure generation for the drift study (H4).

Produces:
- Bar chart of exhaustive drift audit results (max residual per system)
- Heatmap of theorem-constant sweep (ε across β,γ grid)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from analysis.common.io import find_latest_csv, read_csv_rows
from analysis.common.style import save_figure, get_status_color

logger = logging.getLogger(__name__)


def fig_exhaustive_drift_audit(
    data_dir: Path,
    figure_dir: Path,
) -> List[Path]:
    """Generate the exhaustive drift audit summary figure.

    Produces a bar chart showing the maximum residual per system,
    annotated with violation count and ε value.
    """
    import matplotlib.pyplot as plt

    csv_path = find_latest_csv(data_dir, "exhaustive_drift_summary")
    if csv_path is None:
        logger.warning("No exhaustive_drift_summary CSV found — skipping")
        return []

    rows = read_csv_rows(csv_path)
    systems = [r["system_id"] for r in rows]
    max_residuals = [float(r["max_residual"]) for r in rows]
    violations = [int(r["violations"]) for r in rows]
    epsilons = [float(r["epsilon"]) for r in rows]
    statuses = [r["status"] for r in rows]
    colors = [get_status_color(s) for s in statuses]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(systems))
    bars = ax.bar(x, max_residuals, color=colors, edgecolor="#333", alpha=0.85)
    ax.axhline(0, color="#888", linestyle="-", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=25, ha="right")
    ax.set_ylabel("max(LV − theorem RHS)")
    ax.set_title("Exhaustive CTMC Drift Audit (H4)")

    # Annotate with violations and ε
    for i, (bar, viol, eps) in enumerate(zip(bars, violations, epsilons)):
        y_pos = bar.get_height()
        label = f"viol={viol}\nε={eps:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos + 0.02 * abs(ax.get_ylim()[1] - ax.get_ylim()[0]),
            label,
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    out_path = figure_dir / "h4_exhaustive_drift_audit"
    save_figure(fig, out_path)
    return [out_path]


def fig_theorem_constant_sweep(
    data_dir: Path,
    figure_dir: Path,
) -> List[Path]:
    """Generate the theorem-constant sweep heatmap.

    Shows ε values across the (beta, gamma) parameter grid, with
    certified points in green and non-certified in red.
    """
    import matplotlib.pyplot as plt

    csv_path = find_latest_csv(data_dir, "theorem_constant_sweep")
    if csv_path is None:
        logger.warning("No theorem_constant_sweep CSV found — skipping")
        return []

    rows = read_csv_rows(csv_path)

    # Extract unique β and γ values
    betas = sorted(set(float(r["beta"]) for r in rows))
    gammas = sorted(set(float(r["gamma"]) for r in rows))

    if len(betas) < 2 or len(gammas) < 2:
        logger.warning("Insufficient sweep range for heatmap — skipping")
        return []

    # Build ε matrix (average over c values for each β,γ pair)
    eps_matrix = np.full((len(gammas), len(betas)), np.nan)
    for row in rows:
        b_idx = betas.index(float(row["beta"]))
        g_idx = gammas.index(float(row["gamma"]))
        eps_val = float(row["epsilon"])
        if np.isnan(eps_matrix[g_idx, b_idx]):
            eps_matrix[g_idx, b_idx] = eps_val
        else:
            eps_matrix[g_idx, b_idx] = max(eps_matrix[g_idx, b_idx], eps_val)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        eps_matrix, aspect="auto", origin="lower",
        cmap="RdYlGn", interpolation="nearest",
    )
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b:.2f}" for b in betas], rotation=45, ha="right")
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f"{g:.2f}" for g in gammas])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(r"Theorem Drift-Rate $\varepsilon$ Across Parameter Grid (H4)")
    fig.colorbar(im, ax=ax, label=r"$\varepsilon$")

    fig.tight_layout()
    out_path = figure_dir / "h4_theorem_constant_sweep_heatmap"
    save_figure(fig, out_path)
    return [out_path]
