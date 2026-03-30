"""
Publication-quality plotting for the queueing network simulator.

Features:
- Modular theme system (dark/publication)
- Multi-format export (PNG/PDF/SVG)
- Colorblind-safe color palettes
- Vector-graphic ready output
- Clean axis labeling for academic papers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from gibbsq.core.drift import DriftResult
from gibbsq.analysis.metrics import running_average, total_queue_trajectory
from gibbsq.engines.numpy_engine import SimResult
from gibbsq.analysis.theme import apply_theme, get_current_theme, THEMES
from gibbsq.analysis.chart_styles import (
    ChartType,
    ChartStyleSpec,
    get_chart_style,
    get_semantic_colors,
    resolve_colormap,
    PAIR_GIBBSQ,
    PAIR_NEURAL,
    TRAINING_PRIMARY,
    TRAINING_SECONDARY,
    TRAINING_CRITIC,
    TRAINING_GRADIENT,
    TRAINING_ENTROPY,
)
from gibbsq.utils.chart_exporter import save_chart, ChartConfig

log = logging.getLogger(__name__)

__all__ = [
    "plot_trajectory",
    "plot_drift_landscape",
    "plot_drift_vs_norm",
    "plot_policy_comparison",
    "plot_alpha_sweep",
    "plot_convergence",
    "plot_gradient_scatter",
    "plot_stress_dashboard",
    "plot_training_dashboard",
    "plot_raincloud",
    "plot_improvement_heatmap",
    "plot_ablation_bars",
    "plot_critical_load",
]

def _get_plot_colors(theme: str) -> dict:
    """Get color palette based on theme (legacy compatibility)."""
    config = THEMES.get(theme, THEMES["publication"])
    palette = config.color_palette
    return {
        "primary": palette[0],
        "secondary": palette[1],
        "tertiary": palette[2],
        "accent": palette[5],
        "scatter": palette[1],
        "line": palette[0],
        "error": palette[5],
        "histogram": palette[1],
        "contour": "white" if theme == "dark" else "black",
        "text": config.text_color,
    }

def _get_chart_style(
    chart_type: ChartType,
    theme: Optional[str] = None,
) -> tuple:
    """Get chart style spec + theme name for a chart type.

    Returns (spec, theme_name, contour_color).
    """
    theme = theme or get_current_theme() or "publication"
    apply_theme(theme)
    spec = get_chart_style(chart_type, theme=theme)
    contour_color = "white" if theme == "dark" else "black"
    return spec, theme, contour_color

def _setup_plot(theme: Optional[str] = None) -> str:
    """Set up plot with appropriate theme."""
    if theme is None:
        theme = get_current_theme() or "publication"
    apply_theme(theme)
    return theme

def _apply_theme() -> None:
    """Legacy function for backward compatibility. Applies dark theme."""
    apply_theme("dark")

def plot_trajectory(
    result: SimResult,
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot total queue length over time.
    
    Args:
        result: Simulation result
        save_path: Base path for saving (without extension)
        theme: Theme name ('dark' or 'publication')
        formats: Output formats ['png', 'pdf', 'svg']
    
    Returns:
        Matplotlib Figure object
    """
    theme = _setup_plot(theme)
    colors = _get_plot_colors(theme)
    
    fig, ax = plt.subplots()

    t, q_tot = total_queue_trajectory(result)

    max_pts = 10000
    if len(t) > max_pts:
        step = len(t) // max_pts
        t = t[::step]
        q_tot = q_tot[::step]

    ax.plot(t, q_tot, color=colors["primary"], alpha=0.9)
    ax.set_title("Total Queue Length Trajectory")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"$|Q(t)|_1$")

    ax_hist = ax.inset_axes([1.02, 0, 0.2, 1], sharey=ax)
    ax_hist.hist(q_tot, bins=40, orientation="horizontal", color=colors["histogram"], alpha=0.5)
    ax_hist.axis("off")

    if save_path:
        save_chart(fig, Path(save_path), formats or ['png'])
    
    return fig

def plot_drift_landscape(
    drift_res: DriftResult,
    alpha: float,
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Heatmap of 𝓛V(Q) for N=2.
    Requires grid evaluation from `evaluate_grid`.
    
    Args:
        drift_res: Drift evaluation result
        alpha: Routing temperature parameter
        save_path: Base path for saving (without extension)
        theme: Theme name ('dark' or 'publication')
        formats: Output formats ['png', 'pdf', 'svg']
    
    Returns:
        Matplotlib Figure object
    """
    theme = _setup_plot(theme)
    colors = _get_plot_colors(theme)
    
    states = drift_res.states
    if states.shape[1] != 2:
        raise ValueError("Heatmap only supported for N=2 systems.")

    q_max = int(states.max())
    grid_shape = (q_max + 1, q_max + 1)
    drift_grid = drift_res.exact_drifts.reshape(grid_shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        drift_grid.T,
        origin="lower",
        cmap="RdBu_r",  # Diverging colormap (blue=negative, red=positive)
        vmin=-5.0,
        vmax=5.0,
        interpolation="none",
    )
    cbar = plt.colorbar(im, ax=ax, label=r"Generator Drift ${\cal L}V(Q)$")
    
    cbar.ax.yaxis.label.set_color(colors["text"])
    cbar.ax.tick_params(colors=colors["text"])

    ax.set_title(rf"Drift Landscape ($\alpha = {alpha}$)")
    ax.set_xlabel(r"$Q_1$")
    ax.set_ylabel(r"$Q_2$")

    ax.contour(drift_grid.T, levels=[0.0], colors=colors["contour"], linewidths=2.0)

    if save_path:
        save_chart(fig, Path(save_path), formats or ['png'])
    
    return fig

def plot_drift_vs_norm(
    drift_res: DriftResult,
    eps: float,
    R: float,
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Scatter plot: Exact Drift vs L1 Norm.
    Overlays the theoretical simplified bound: −ε|Q|₁ + R.
    
    Args:
        drift_res: Drift evaluation result
        eps: Drift rate epsilon
        R: Drift constant R
        save_path: Base path for saving (without extension)
        theme: Theme name ('dark' or 'publication')
        formats: Output formats ['png', 'pdf', 'svg']
    
    Returns:
        Matplotlib Figure object
    """
    theme = _setup_plot(theme)
    colors = _get_plot_colors(theme)
    
    fig, ax = plt.subplots()

    norms = drift_res.norms
    drifts = drift_res.exact_drifts

    max_pts = 5000
    if len(norms) > max_pts:
        idx = np.random.choice(len(norms), max_pts, replace=False)
        norms = norms[idx]
        drifts = drifts[idx]

    ax.scatter(norms, drifts, color=colors["scatter"], s=10, alpha=0.5, label="Exact Drift")

    x_line = np.array([0, float(norms.max())])
    y_line = -eps * x_line + R
    ax.plot(x_line, y_line, color=colors["error"], linewidth=2.5, 
            label=r"Bound: $-\varepsilon|Q|_1 + R$")

    ax.axhline(0, color=colors["text"], linestyle="--", alpha=0.3)

    ax.set_title("Drift Verification")
    ax.set_xlabel(r"State Norm $|Q|_1$")
    ax.set_ylabel(r"Generator Drift ${\cal L}V(Q)$")
    ax.legend(loc="upper right")

    if save_path:
        save_chart(fig, Path(save_path), formats or ['png'])
    
    return fig

def plot_policy_comparison(
    results_dict: dict[str, list[float]],
    metric_name: str,
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Bar chart comparing policies on a specific metric.
    
    Args:
        results_dict: Maps "Policy Name" -> List of replication values
        metric_name: Name of the metric being compared
        save_path: Base path for saving (without extension)
        theme: Theme name ('dark' or 'publication')
        formats: Output formats ['png', 'pdf', 'svg']
    
    Returns:
        Matplotlib Figure object
    """
    theme = _setup_plot(theme)
    colors = _get_plot_colors(theme)
    
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = list(results_dict.keys())
    means = [np.mean(vals) for vals in results_dict.values()]
    errs = [np.std(vals) / max(np.sqrt(len(vals)), 1) for vals in results_dict.values()]

    x = np.arange(len(labels))
    
    palette = THEMES[theme].color_palette
    bar_colors = [palette[i % len(palette)] for i in range(len(labels))]
    
    bars = ax.bar(x, means, yerr=errs, capsize=5, color=bar_colors, alpha=0.85,
                  edgecolor=colors["contour"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Policy Comparison: {metric_name}")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.05, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9, color=colors["text"])

    if save_path:
        save_chart(fig, Path(save_path), formats or ['png'])
    
    return fig

def plot_alpha_sweep(
    alpha_values: np.ndarray,
    mean_q_matrix: np.ndarray,
    rho_labels: list[str],
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Line plot: Mean Q vs α, stratified by ρ.
    
    Args:
        alpha_values: Array of alpha (inverse temperature) values
        mean_q_matrix: Matrix of mean queue lengths (rho x alpha)
        rho_labels: Labels for each load factor
        save_path: Base path for saving (without extension)
        theme: Theme name ('dark' or 'publication')
        formats: Output formats ['png', 'pdf', 'svg']
    
    Returns:
        Matplotlib Figure object
    """
    theme = _setup_plot(theme)
    
    fig, ax = plt.subplots(figsize=(9, 6))

    palette = THEMES[theme].color_palette
    
    for i, rho_label in enumerate(rho_labels):
        color = palette[i % len(palette)]
        ax.plot(alpha_values, mean_q_matrix[i, :], marker="o", color=color,
                linewidth=2, markersize=6, label=rho_label)

    ax.set_xscale("log")
    ax.set_title(r"System Performance vs Routing Temperature ($\alpha$)")
    ax.set_xlabel(r"Inverse Temperature $\alpha$")
    ax.set_ylabel(r"Expected Total Queue Length $\mathbb{E}[|Q|_1]$")
    ax.legend(title="Load Factor")

    if save_path:
        save_chart(fig, Path(save_path), formats or ['png'])
    
    return fig

def plot_convergence(
    result: SimResult,
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot the running average E[Q(t)] converging to steady state.
    
    Args:
        result: Simulation result
        save_path: Base path for saving (without extension)
        theme: Theme name ('dark' or 'publication')
        formats: Output formats ['png', 'pdf', 'svg']
    
    Returns:
        Matplotlib Figure object
    """
    theme = _setup_plot(theme)
    colors = _get_plot_colors(theme)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    t, cum_avg = running_average(result)

    if len(t) > 5000:
        step = len(t) // 5000
        t = t[::step]
        cum_avg = cum_avg[::step]

    ax.plot(t, cum_avg, color=colors["tertiary"], linewidth=2)
    ax.set_title("Running Average of Total Queue Length")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"$\frac{1}{t}\int_0^t |Q(s)|_1 ds$")

    final_val = cum_avg[-1]
    ax.axhline(final_val, color=colors["contour"], linestyle="--", alpha=0.5,
               label=rf"Final: {final_val:.2f}")
    ax.legend(loc="lower right")

    if save_path:
        save_chart(fig, Path(save_path), formats or ['png'])
    
    return fig

def plot_gradient_scatter(
    fd_grads: np.ndarray,
    rf_grads: np.ndarray,
    *,
    z_scores: Optional[np.ndarray] = None,
    summary_stats: Optional[dict] = None,
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """Per-parameter scatter: REINFORCE vs finite-difference gradients.

    Each dot is one parameter.  Points near y=x indicate agreement.

    Parameters
    ----------
    fd_grads : array (n_params,)
        Finite-difference gradient (ground truth).
    rf_grads : array (n_params,)
        REINFORCE gradient estimates.
    z_scores : array (n_params,), optional
        Per-parameter Z-score for colour encoding.
    summary_stats : dict, optional
        Keys ``cosine_similarity``, ``relative_error``, ``passed``
        for the annotation scoreboard.
    """
    spec, theme, contour = _get_chart_style(ChartType.SCATTER, theme)

    fig, ax = plt.subplots(figsize=spec.figsize)

    c = z_scores if z_scores is not None else np.abs(rf_grads - fd_grads)
    sc = ax.scatter(
        fd_grads, rf_grads,
        c=c, cmap=spec.colormap, s=spec.marker_size,
        alpha=spec.line_alpha, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Z-score magnitude" if z_scores is not None else "|error|")

    lims = [
        min(fd_grads.min(), rf_grads.min()),
        max(fd_grads.max(), rf_grads.max()),
    ]
    ax.plot(lims, lims, "--", color=contour, linewidth=spec.line_width, alpha=0.6, label="y = x")

    ax.set_xlabel("Finite-Difference Gradient (ground truth)")
    ax.set_ylabel("REINFORCE Gradient (estimate)")
    ax.set_title("Gradient Estimator Agreement")
    ax.legend(loc="upper left")

    if summary_stats:
        text_parts = []
        if "cosine_similarity" in summary_stats:
            text_parts.append(f"cos = {summary_stats['cosine_similarity']:.4f}")
        if "relative_error" in summary_stats:
            text_parts.append(f"rel err = {summary_stats['relative_error']:.4f}")
        if "passed" in summary_stats:
            text_parts.append("PASSED ✓" if summary_stats["passed"] else "FAILED ✗")
        ax.text(
            0.02, 0.98, "\n".join(text_parts),
            transform=ax.transAxes, fontsize=spec.annotation_fontsize,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        )

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])

    return fig

def plot_stress_dashboard(
    scaling_data: dict,
    critical_data: dict,
    hetero_data: dict,
    *,
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """Three-panel stress-test dashboard.

    Parameters
    ----------
    scaling_data : dict
        Keys ``n_values`` (list), ``mean_q`` (list), ``gini`` (list).
    critical_data : dict
        Keys ``rho_values`` (list), ``mean_q`` (list),
        ``stationary`` (list of bool).
    hetero_data : dict
        Keys ``scenario_names`` (list), ``mean_q`` (list),
        ``gini`` (list).
    """
    spec, theme, contour = _get_chart_style(ChartType.STRESS_DASHBOARD, theme)

    fig, axes = plt.subplots(1, 3, figsize=spec.figsize)

    ax_a = axes[0]
    ax_a.plot(
        scaling_data["n_values"], scaling_data["mean_q"],
        marker=spec.marker_style, markersize=spec.marker_size,
        linewidth=spec.line_width, color=spec.palette[0],
    )
    ax_a.set_xlabel("Number of Servers (N)")
    ax_a.set_ylabel(r"$\mathbb{E}[|Q|_1]$")
    ax_a.set_title("(a) Scaling Test")
    ax_a.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    ax_b = axes[1]
    rho = critical_data["rho_values"]
    eq = critical_data["mean_q"]
    stationary = critical_data.get("stationary", [True] * len(rho))
    colors_b = [spec.palette[2] if s else spec.palette[5] for s in stationary]
    ax_b.scatter(rho, eq, c=colors_b, s=spec.marker_size * 8, zorder=3)
    ax_b.plot(rho, eq, linewidth=spec.line_width, color=spec.palette[4], alpha=0.7)
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"Load Factor $\rho$")
    ax_b.set_ylabel(r"$\mathbb{E}[|Q|_1]$ (log)")
    ax_b.set_title("(b) Critical Load")
    ax_b.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    ax_c = axes[2]
    x_pos = np.arange(len(hetero_data["scenario_names"]))
    bar_colors = [spec.palette[i % len(spec.palette)] for i in range(len(x_pos))]
    ax_c.bar(x_pos, hetero_data["mean_q"], color=bar_colors,
             alpha=spec.fill_alpha, edgecolor=contour)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(hetero_data["scenario_names"], rotation=30, ha="right", fontsize=7)
    ax_c.set_ylabel(r"$\mathbb{E}[|Q|_1]$")
    ax_c.set_title("(c) Heterogeneity")
    ax_c.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    fig.suptitle("Stress Test Dashboard", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])

    return fig

def plot_training_dashboard(
    metrics: dict,
    *,
    jsq_baseline: Optional[float] = None,
    random_baseline: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """Four-panel REINFORCE training dashboard.

    Parameters
    ----------
    metrics : dict
        Keys: ``epoch`` (list), ``performance_index`` (list),
        ``performance_index_ema`` (list), ``policy_loss`` (list),
        ``value_loss`` (list), ``ev_ema`` (list), ``corr_ema`` (list),
        ``policy_grad_norm`` (list), ``value_grad_norm`` (list),
        ``entropy`` (list).
    jsq_baseline : float, optional
        100% reference line on performance panel.
    random_baseline : float, optional
        0% reference line on performance panel.
    """
    spec, theme, contour = _get_chart_style(ChartType.TRAINING_DASHBOARD, theme)

    fig, axes = plt.subplots(
        spec.subplot_layout[0], spec.subplot_layout[1],
        figsize=spec.figsize,
    )
    perf_key = "base_regime_index" if "base_regime_index" in metrics else "performance_index"
    perf_ema_key = "base_regime_index_ema" if "base_regime_index_ema" in metrics else "performance_index_ema"
    epochs = metrics.get("epoch", list(range(len(metrics.get(perf_key, [])))))

    ax_a = axes[0, 0]
    if perf_key in metrics:
        ax_a.plot(epochs, metrics[perf_key],
                  alpha=0.35, linewidth=0.8, color=TRAINING_PRIMARY, label="Raw Base-Regime PI")
    if perf_ema_key in metrics:
        ax_a.plot(epochs, metrics[perf_ema_key],
                  linewidth=spec.line_width, color=TRAINING_PRIMARY, label="EMA Base-Regime PI")
    if jsq_baseline is not None:
        ax_a.axhline(jsq_baseline, color=contour, linestyle="--", alpha=0.4, label="JSQ (100%)")
    if random_baseline is not None:
        ax_a.axhline(random_baseline, color="#999999", linestyle=":", alpha=0.4, label="Random (0%)")
    ax_a.set_ylabel("Base-Regime PI Proxy (%)")
    ax_a.set_title("(a) Base-Regime Diagnostic")
    ax_a.legend(loc="lower right", fontsize=7)
    ax_a.grid(True, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    ax_b = axes[0, 1]
    if "policy_loss" in metrics:
        ax_b.plot(epochs, metrics["policy_loss"], linewidth=spec.line_width,
                  color=TRAINING_SECONDARY, label="Policy Loss")
    if "value_loss" in metrics:
        ax_b2 = ax_b.twinx()
        ax_b2.plot(epochs, metrics["value_loss"], linewidth=spec.line_width,
                   color=TRAINING_CRITIC, alpha=0.8, label="Value Loss")
        ax_b2.set_ylabel("Value Loss", color=TRAINING_CRITIC)
        ax_b2.tick_params(axis="y", labelcolor=TRAINING_CRITIC)
    ax_b.set_ylabel("Policy Loss", color=TRAINING_SECONDARY)
    ax_b.tick_params(axis="y", labelcolor=TRAINING_SECONDARY)
    ax_b.set_title("(b) Loss Curves")
    ax_b.grid(True, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    ax_c = axes[1, 0]
    if "ev_ema" in metrics:
        ax_c.plot(epochs, metrics["ev_ema"], linewidth=spec.line_width,
                  color=TRAINING_CRITIC, label="Explained Var EMA")
    if "corr_ema" in metrics:
        ax_c.plot(epochs, metrics["corr_ema"], linewidth=spec.line_width,
                  color=TRAINING_GRADIENT, label="Correlation EMA")
    ax_c.set_ylabel("Critic Diagnostics")
    ax_c.set_xlabel("Epoch")
    ax_c.set_title("(c) Critic Quality")
    ax_c.legend(loc="lower right", fontsize=7)
    ax_c.grid(True, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    ax_d = axes[1, 1]
    if "policy_grad_norm" in metrics:
        ax_d.plot(epochs, metrics["policy_grad_norm"], linewidth=spec.line_width,
                  color=TRAINING_GRADIENT, label="Policy grad norm")
    if "value_grad_norm" in metrics:
        ax_d.plot(epochs, metrics["value_grad_norm"], linewidth=spec.line_width,
                  color=TRAINING_CRITIC, alpha=0.8, label="Value grad norm")
    ax_d.set_ylabel("Gradient Norm")
    if "entropy" in metrics:
        ax_d2 = ax_d.twinx()
        ax_d2.plot(epochs, metrics["entropy"], linewidth=1.0,
                   color=TRAINING_ENTROPY, alpha=0.7, label="Entropy")
        ax_d2.set_ylabel("Entropy", color=TRAINING_ENTROPY)
        ax_d2.tick_params(axis="y", labelcolor=TRAINING_ENTROPY)
    ax_d.set_xlabel("Epoch")
    ax_d.set_title("(d) Gradient Health")
    ax_d.legend(loc="upper right", fontsize=7)
    ax_d.grid(True, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    fig.suptitle("REINFORCE Training Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])

    return fig

def plot_raincloud(
    group_a_data: np.ndarray,
    group_b_data: np.ndarray,
    group_a_label: str = "GibbsQ",
    group_b_label: str = "N-GibbsQ",
    *,
    stats: Optional[dict] = None,
    y_label: str = r"$\mathbb{E}[|Q|_1]$",
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """Raincloud plot: half-violin + jittered scatter + mini-boxplot.

    Uses seaborn for the violin component and matplotlib for scatter
    and box overlays.  This is the gold-standard visualisation for
    comparing two groups with n≈30 (Allen et al. 2019).

    Parameters
    ----------
    group_a_data, group_b_data : array (n_samples,)
        Per-seed metric values for each group.
    stats : dict, optional
        Keys ``p_value``, ``cohen_d``, ``ci_low``, ``ci_high``,
        ``improvement_pct`` for the annotation bracket.
    """
    spec, theme, contour = _get_chart_style(ChartType.RAINCLOUD, theme)

    try:
        import seaborn as sns
        _has_seaborn = True
    except ImportError:
        _has_seaborn = False
        log.warning("seaborn not installed; falling back to box+scatter only")

    fig, ax = plt.subplots(figsize=spec.figsize)

    colors = [PAIR_GIBBSQ, PAIR_NEURAL]
    data_groups = [group_a_data, group_b_data]
    labels = [group_a_label, group_b_label]
    positions = [0, 1]

    for i, (data, label, pos) in enumerate(zip(data_groups, labels, positions)):
        color = colors[i]

        if _has_seaborn:
            parts = ax.violinplot(
                data, positions=[pos], showmeans=False,
                showmedians=False, showextrema=False,
                widths=0.6,
            )
            for body in parts["bodies"]:
                m = np.mean(body.get_paths()[0].vertices[:, 0])
                body.get_paths()[0].vertices[:, 0] = np.clip(
                    body.get_paths()[0].vertices[:, 0],
                    -np.inf if i == 0 else m,
                    m if i == 0 else np.inf,
                )
                body.set_facecolor(color)
                body.set_edgecolor(contour)
                body.set_alpha(spec.fill_alpha)

        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(data))
        offset = 0.15 if i == 0 else -0.15
        ax.scatter(
            np.full_like(data, pos + offset) + jitter, data,
            s=spec.marker_size * 6, color=color, alpha=0.7,
            edgecolors="white", linewidth=0.5, zorder=3,
        )

        bp = ax.boxplot(
            data, positions=[pos], widths=0.12,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="white", linewidth=1.5),
            whiskerprops=dict(color=contour),
            capprops=dict(color=contour),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(y_label)
    ax.set_title(f"{group_a_label} vs {group_b_label}: Distribution Comparison")
    ax.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha, axis="y")

    if stats:
        y_max = max(group_a_data.max(), group_b_data.max())
        bracket_y = y_max * 1.08
        ax.plot([0, 0, 1, 1], [bracket_y * 0.98, bracket_y, bracket_y, bracket_y * 0.98],
                color=contour, linewidth=1.2)
        parts = []
        if "p_value" in stats:
            p = stats["p_value"]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            parts.append(f"p = {p:.4f} {stars}")
        if "cohen_d" in stats:
            parts.append(f"d = {stats['cohen_d']:.2f}")
        if "improvement_pct" in stats:
            parts.append(f"Δ = {stats['improvement_pct']:.1f}%")
        ax.text(0.5, bracket_y * 1.02, "  |  ".join(parts),
                ha="center", fontsize=spec.annotation_fontsize,
                fontweight="bold")

    fig.tight_layout()

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])

    return fig

def plot_improvement_heatmap(
    grid: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    *,
    x_axis_name: str = r"Load Factor $\rho$",
    y_axis_name: str = "Service Rate Scale",
    cmap: str = "RdYlGn",
    center: float = 1.0,
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """Annotated diverging heatmap for improvement-ratio grids.

    Used by gen_sweep.py for GibbsQ/Neural improvement ratios.

    Parameters
    ----------
    grid : array (n_y, n_x)
        Improvement ratio values.  >1.0 means Neural wins.
    x_labels, y_labels : lists of str
        Tick labels for each axis.
    center : float
        Colormap normalisation midpoint (1.0 = break-even).
    """
    spec, theme, contour = _get_chart_style(ChartType.HEATMAP_DIVERGING, theme)
    spec.colormap = cmap
    spec.colormap_center = center

    fig, ax = plt.subplots(figsize=spec.figsize)

    vmax = max(abs(grid.max() - center), abs(grid.min() - center))
    norm = mcolors.TwoSlopeNorm(vmin=center - vmax, vcenter=center, vmax=center + vmax)

    im = ax.imshow(
        grid, cmap=spec.colormap, norm=norm,
        interpolation="none", aspect="auto",
    )
    cbar = plt.colorbar(im, ax=ax, label="Improvement Ratio (GibbsQ / Neural)")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            text_color = "white" if abs(val - center) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:{spec.value_format}}x",
                    ha="center", va="center",
                    fontsize=spec.annotation_fontsize, color=text_color,
                    fontweight="bold")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    ax.set_title("Generalisation Sweep: Improvement Ratio")

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])

    return fig

def plot_ablation_bars(
    variant_names: List[str],
    mean_values: List[float],
    se_values: List[float],
    *,
    metric_name: str = r"$\mathbb{E}[Q_{total}]$",
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """Ablation bar chart with SE error bars and delta-% annotations.

    Bars are ordered from best (Full Model) to worst (Baseline),
    with delta-% labels showing degradation from the best variant.

    Parameters
    ----------
    variant_names : list of str
        Variant labels (e.g. ["Full", "No Log-Norm", ...]).
    mean_values : list of float
        Mean metric value per variant.
    se_values : list of float
        Standard error per variant.
    """
    spec, theme, contour = _get_chart_style(ChartType.BAR_COMPARISON, theme)

    fig, ax = plt.subplots(figsize=spec.figsize)
    x = np.arange(len(variant_names))

    palette = spec.palette
    bar_colors = [palette[i % len(palette)] for i in range(len(variant_names))]

    bars = ax.bar(
        x, mean_values, yerr=se_values,
        capsize=spec.error_bar_capsize, color=bar_colors,
        alpha=spec.fill_alpha, edgecolor=contour,
    )

    best_val = mean_values[0]
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + se_values[i] + 0.02 * best_val,
            f"{h:{spec.value_format}}",
            ha="center", va="bottom", fontsize=spec.annotation_fontsize,
        )
        if i > 0 and best_val > 0:
            delta = ((h - best_val) / best_val) * 100
            sign = "+" if delta > 0 else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + se_values[i] + 0.08 * best_val,
                f"{sign}{delta:.1f}%",
                ha="center", va="bottom",
                fontsize=spec.annotation_fontsize - 1,
                color="#D55E00" if delta > 0 else "#009E73",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=20, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title("Ablation Study: Component Contributions")
    ax.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha, axis="y")

    fig.tight_layout()

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])

    return fig

def plot_critical_load(
    rho_values: np.ndarray,
    neural_eq: np.ndarray,
    gibbs_eq: np.ndarray,
    *,
    critical_rho: float = 0.95,
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """Dual-line log-scale divergence plot for critical load comparison.

    Compares N-GibbsQ vs GibbsQ E[Q] as ρ→1 with an annotated
    near-critical regime shading.

    Parameters
    ----------
    rho_values : array (n_rho,)
        Load factor values.
    neural_eq, gibbs_eq : arrays (n_rho,)
        Expected queue lengths for each method.
    critical_rho : float
        Threshold for the "near-critical" shaded region.
    """
    spec, theme, contour = _get_chart_style(ChartType.LINE_SERIES, theme)

    fig, ax = plt.subplots(figsize=spec.figsize)

    ax.plot(rho_values, gibbs_eq,
            marker="o", markersize=spec.marker_size,
            linewidth=spec.line_width, color=PAIR_GIBBSQ,
            label="GibbsQ", alpha=spec.line_alpha)
    ax.plot(rho_values, neural_eq,
            marker="s", markersize=spec.marker_size,
            linewidth=spec.line_width, color=PAIR_NEURAL,
            label="N-GibbsQ", alpha=spec.line_alpha)

    ax.set_yscale("log")
    ax.set_xlabel(r"Load Factor $\rho$")
    ax.set_ylabel(r"$\mathbb{E}[|Q|_1]$ (log scale)")
    ax.set_title(r"Critical Load: E[Q] vs $\rho$ Near Stability Boundary")

    ax.axvspan(critical_rho, rho_values.max() * 1.01,
               alpha=0.08, color="#D55E00", label=f"Near-critical (ρ > {critical_rho})")

    ax.legend(loc="upper left")
    ax.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    fig.tight_layout()

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])

    return fig

def plot_tier_comparison_bars(
    labels: List[str],
    q_values: List[float],
    q_errors: List[float],
    tiers: List[int],
    *,
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """Tier-colored comparison bar chart for baseline policies.
    
    Parameters
    ----------
    labels : list of str
        Names of the baselines/policies.
    q_values : list of float
        Expected queue length means.
    q_errors : list of float
        Expected queue length standard errors.
    tiers : list of int
        Tier grouping (1-5) used for categorical coloring.
    """
    spec, theme, contour = _get_chart_style(ChartType.BAR_COMPARISON, theme)
    palette = spec.palette
    
    tier_colors = {
        1: palette[min(7, len(palette) - 1)],
        2: palette[min(1, len(palette) - 1)],
        3: palette[min(2, len(palette) - 1)],
        4: palette[min(4, len(palette) - 1)],
        5: palette[min(5, len(palette) - 1)],
    }
    colors = [tier_colors.get(t, palette[0]) for t in tiers]

    fig, ax = plt.subplots(figsize=spec.figsize)
    x = np.arange(len(labels))

    ax.bar(x, q_values, yerr=q_errors, color=colors, alpha=spec.fill_alpha,
           capsize=spec.error_bar_capsize, edgecolor=palette[0])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Expected Total Queue Length E[Q_total]')
    ax.set_title('Corrected Policy Comparison')
    ax.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha, axis='y')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f'Tier {t}', alpha=spec.fill_alpha) 
                       for t, c in sorted(tier_colors.items()) if t in tiers]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.tight_layout()

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])
        
    return fig

def plot_platinum_grid(
    rho_values: np.ndarray,
    uniform_eq: np.ndarray,
    neural_eq: np.ndarray,
    jsq_eq: np.ndarray,
    performance_index: np.ndarray,
    *,
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> plt.Figure:
    """Platinum grid analysis: Performance Envelope and Generalization Efficiency.
    
    Dual-panel plot. Left: Log-scale E[Q] comparison vs rho. 
    Right: Performance Index (%) relative to JSQ and Uniform vs rho.
    """
    spec, theme, contour = _get_chart_style(ChartType.LINE_SERIES, theme)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(rho_values, uniform_eq, linestyle='--', marker='x', color='#95a5a6',
             linewidth=spec.line_width, label='Uniform')
    ax1.plot(rho_values, neural_eq, linestyle='-', marker='o',
             linewidth=spec.line_width + 0.5, color=PAIR_NEURAL, label='N-GibbsQ (Platinum)')
    ax1.plot(rho_values, jsq_eq, linestyle='-.', marker='s',
             linewidth=spec.line_width, color=PAIR_GIBBSQ, label='JSQ (Optimal)')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Load Factor $\rho$')
    ax1.set_ylabel(r'$\mathbb{E}[|Q|_1]$ (Log Scale)')
    ax1.set_title('Performance Envelope')
    ax1.legend()
    ax1.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)
    
    ax2.plot(rho_values, performance_index, linestyle='-', marker='D',
             linewidth=spec.line_width + 0.5, color=spec.palette[0])
    ax2.axhline(100, color=PAIR_GIBBSQ, linestyle='--', alpha=0.5, label='JSQ Parity')
    ax2.axhline(0, color='#95a5a6', linestyle='--', alpha=0.5, label='Uniform Parity')
    ax2.set_ylim(-10, 110)
    ax2.set_xlabel(r'Load Factor $\rho$')
    ax2.set_ylabel('Performance Index (%)')
    ax2.set_title('Generalization Efficiency')
    ax2.legend()
    ax2.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)
    
    fig.tight_layout()
    
    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])
        
    return fig

