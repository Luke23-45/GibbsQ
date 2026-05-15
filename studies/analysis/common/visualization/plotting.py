"""
Core plotting routines for the GibbsQ analysis pipeline.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from .theme import apply_theme

def create_base_plot(figsize: Tuple[float, float] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Create a standard figure with the GibbsQ theme applied."""
    apply_theme()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def save_plot(fig: plt.Figure, path: str, dpi: int = 300):
    """Save plot to both PDF and PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p.with_suffix(".pdf")), format="pdf", bbox_inches="tight")
    fig.savefig(str(p.with_suffix(".png")), format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)

def plot_alpha_sweep(alphas, q_matrix, labels, stationary_matrix=None, save_path=None, theme="publication", formats=None):
    """Premium alpha-sweep plot for Gibbs bound verification."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        mask = stationary_matrix[i] if stationary_matrix is not None else np.ones_like(alphas, dtype=bool)
        ax.plot(alphas[mask], q_matrix[i][mask], marker='o', label=label)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\mathbb{E}[|Q|_1]$")
    ax.set_title("Alpha Sweep: Convergence vs Load")
    ax.legend()
    if save_path:
        save_plot(fig, str(save_path))

def plot_ablation_dual_panel(variant_names, mean_values, se_values, save_path=None, theme="publication", formats=None):
    """Premium dual-panel ablation study visualization."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(variant_names))
    ax.bar(x, mean_values, yerr=se_values, capsize=5, color="#332288", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=45, ha='right')
    ax.set_ylabel("Mean Queue Length")
    ax.set_title("Ablation Study Results")
    if save_path:
        save_plot(fig, str(save_path))

def plot_policy_dual_panel(labels, q_values, q_errors, tiers, save_path=None, theme="publication", formats=None):
    """Premium policy comparison visualization."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    colors = ['#332288' if t == 'baseline' else '#117733' for t in tiers]
    ax.bar(x, q_values, yerr=q_errors, capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Mean Queue Length")
    ax.set_title("Policy Performance Comparison")
    if save_path:
        save_plot(fig, str(save_path))

def plot_critical_load(rho_values, neural_eq, gibbs_eq, save_path=None, theme="publication", formats=None):
    """Premium critical load verification plot."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rho_values, neural_eq, 'o-', label="Neural Policy")
    ax.plot(rho_values, gibbs_eq, 's--', label="Reflected-UAS (Theory)")
    ax.set_xlabel(r"$\rho$ (Load Factor)")
    ax.set_ylabel("Equilibrium Queue Length")
    ax.set_title("Critical Load Stability Boundary")
    ax.legend()
    if save_path:
        save_plot(fig, str(save_path))

def plot_improvement_heatmap(grid, x_labels, y_labels, save_path=None, theme="publication", formats=None):
    """Premium improvement heatmap for generalization studies."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, cmap="RdYlGn")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Load (rho)")
    ax.set_ylabel("Scale Factor")
    ax.set_title("Generalization Improvement Heatmap")
    fig.colorbar(im, ax=ax, label="Improvement %")
    if save_path:
        save_plot(fig, str(save_path))
