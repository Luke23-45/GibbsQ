"""
Publication-quality plotting for the queueing network simulator.

Features a dark theme out-of-the-box (configurable via rcParams),
vector-graphic ready output, and clean axis labeling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from gibbsq.core.drift import DriftResult
from gibbsq.analysis.metrics import running_average, total_queue_trajectory
from gibbsq.engines.numpy_engine import SimResult

log = logging.getLogger(__name__)

__all__ = [
    "plot_trajectory",
    "plot_drift_landscape",
    "plot_drift_vs_norm",
    "plot_policy_comparison",
    "plot_alpha_sweep",
    "plot_convergence",
]

# ──────────────────────────────────────────────────────────────
#  Global theme configuration
# ──────────────────────────────────────────────────────────────

def _apply_theme() -> None:
    """Set publication-ready dark theme."""
    preferred_fonts = ["Inter", "Roboto", "Helvetica Neue", "Arial"]
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
    available_preferred = [f for f in preferred_fonts if f in installed_fonts]
    sans_serif_stack = available_preferred + ["DejaVu Sans"]

    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   sans_serif_stack,
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "legend.frameon":    False,
        "axes.linewidth":    1.2,
        "axes.grid":         True,
        "grid.alpha":        0.15,
        "grid.linestyle":    "--",
        "lines.linewidth":   1.5,
        "figure.figsize":    (8, 5),
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


# ──────────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────────

def plot_trajectory(
    result: SimResult,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot total queue length over time."""
    _apply_theme()
    fig, ax = plt.subplots()

    t, q_tot = total_queue_trajectory(result)

    # Subsample if too large for sensible plotting
    max_pts = 10000
    if len(t) > max_pts:
        step = len(t) // max_pts
        t = t[::step]
        q_tot = q_tot[::step]

    ax.plot(t, q_tot, color="#00E5FF", alpha=0.9)
    ax.set_title("Total Queue Length Trajectory")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"$|Q(t)|_1$")

    # Add marginal histogram
    ax_hist = ax.inset_axes([1.02, 0, 0.2, 1], sharey=ax)
    ax_hist.hist(q_tot, bins=40, orientation="horizontal", color="#00E5FF", alpha=0.5)
    ax_hist.axis("off")

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_drift_landscape(
    drift_res: DriftResult,
    alpha: float,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Heatmap of 𝓛V(Q) for N=2.
    Requires grid evaluation from `evaluate_grid`.
    """
    _apply_theme()
    states = drift_res.states
    if states.shape[1] != 2:
        raise ValueError("Heatmap only supported for N=2 systems.")

    # Infer grid dimensions
    q_max = int(states.max())
    grid_shape = (q_max + 1, q_max + 1)
    drift_grid = drift_res.exact_drifts.reshape(grid_shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        drift_grid.T,
        origin="lower",
        cmap="coolwarm",
        vmin=-5.0,
        vmax=5.0,  # Clip extremes for visual clarity
        interpolation="none",
    )
    plt.colorbar(im, ax=ax, label=r"Generator Drift ${\cal L}V(Q)$")

    ax.set_title(rf"Drift Landscape ($\alpha = {alpha}$)")
    ax.set_xlabel(r"$Q_1$")
    ax.set_ylabel(r"$Q_2$")

    # Region of positive drift contour
    ax.contour(drift_grid.T, levels=[0.0], colors="white", linewidths=2.0)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_drift_vs_norm(
    drift_res: DriftResult,
    eps: float,
    R: float,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Scatter plot:  Exact Drift  vs  L1 Norm.
    Overlays the theoretical simplified bound:  −ε|Q|₁ + R.
    """
    _apply_theme()
    fig, ax = plt.subplots()

    norms = drift_res.norms
    drifts = drift_res.exact_drifts

    # Random subsample to avoid overplotting if trajectory-based
    max_pts = 5000
    if len(norms) > max_pts:
        idx = np.random.choice(len(norms), max_pts, replace=False)
        norms = norms[idx]
        drifts = drifts[idx]

    ax.scatter(norms, drifts, color="#B388FF", s=10, alpha=0.5, label="Exact Drift")

    # Theoretical line
    x_line = np.array([0, float(norms.max())])
    y_line = -eps * x_line + R
    ax.plot(x_line, y_line, color="#FF1744", linewidth=2.5, label=r"Bound: $-\varepsilon|Q|_1 + R$")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_title("Drift Verification")
    ax.set_xlabel(r"State Norm $|Q|_1$")
    ax.set_ylabel(r"Generator Drift ${\cal L}V(Q)$")
    ax.legend(loc="upper right")

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_policy_comparison(
    results_dict: dict[str, list[float]],
    metric_name: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Bar chart comparing policies on a specific metric.
    `results_dict` maps "Policy Name" -> List of replication values.
    """
    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = list(results_dict.keys())
    means = [np.mean(vals) for vals in results_dict.values()]
    errs = [np.std(vals) / max(np.sqrt(len(vals)), 1) for vals in results_dict.values()]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=errs, capsize=5, color="#00B0FF", alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Policy Comparison: {metric_name}")

    # Value annotations
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.05, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9, color="white")

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_alpha_sweep(
    alpha_values: np.ndarray,
    mean_q_matrix: np.ndarray,
    rho_labels: list[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Line plot:  Mean Q vs α, stratified by ρ.
    """
    _apply_theme()
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(rho_labels)))

    for i, rho_label in enumerate(rho_labels):
        ax.plot(alpha_values, mean_q_matrix[i, :], marker="o", color=colors[i],
                linewidth=2, markersize=6, label=rho_label)

    ax.set_xscale("log")
    ax.set_title(r"System Performance vs Routing Temperature ($\alpha$)")
    ax.set_xlabel(r"Inverse Temperature $\alpha$")
    ax.set_ylabel(r"Expected Total Queue Length $\mathbb{E}[|Q|_1]$")
    ax.legend(title="Load Factor")

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_convergence(
    result: SimResult,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot the running average E[Q(t)] converging to steady state.
    """
    _apply_theme()
    fig, ax = plt.subplots(figsize=(8, 4))
    t, cum_avg = running_average(result)

    # Subsample points
    if len(t) > 5000:
        step = len(t) // 5000
        t = t[::step]
        cum_avg = cum_avg[::step]

    ax.plot(t, cum_avg, color="#00E676", linewidth=2)
    ax.set_title("Running Average of Total Queue Length")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"$\frac{1}{t}\int_0^t |Q(s)|_1 ds$")

    # Draw final average line
    final_val = cum_avg[-1]
    ax.axhline(final_val, color="white", linestyle="--", alpha=0.5,
               label=rf"Final: {final_val:.2f}")
    ax.legend(loc="lower right")

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig
