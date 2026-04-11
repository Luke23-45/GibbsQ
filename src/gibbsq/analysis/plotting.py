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
from matplotlib.lines import Line2D

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
from gibbsq.analysis.plot_profiles import (
    ExperimentPlotContext,
    ExperimentPlotProfile,
    resolve_experiment_plot_profile,
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
    "plot_ablation_training_curve",
    "plot_raincloud",
    "plot_improvement_heatmap",
    "plot_ablation_bars",
    "plot_ablation_dual_panel",
    "plot_tier_comparison_bars",
    "plot_policy_dual_panel",
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


def _as_1d_array(values: Union[np.ndarray, list, tuple], name: str) -> np.ndarray:
    """Convert a sequence-like value to a non-empty 1D ndarray."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _validate_equal_lengths(**arrays: np.ndarray) -> int:
    """Ensure one or more 1D arrays all share the same length."""
    lengths = {name: int(arr.shape[0]) for name, arr in arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        lengths_text = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Input lengths must match; got {lengths_text}")
    size = unique_lengths.pop()
    if size == 0:
        raise ValueError("Input arrays must be non-empty")
    return size

def plot_trajectory(
    result: SimResult,
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    plot_profile = resolve_experiment_plot_profile(
        None,
        "plot_trajectory",
        context=context,
        profile=profile,
    )
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
    ax.set_title(plot_profile.figure_title or "Total Queue Length Trajectory")
    ax.set_xlabel(plot_profile.axis_labels.get("x", "Time (t)"))
    ax.set_ylabel(plot_profile.axis_labels.get("y", r"$|Q(t)|_1$"))

    ax_hist = ax.inset_axes([1.02, 0, 0.2, 1], sharey=ax)
    ax_hist.hist(q_tot, bins=40, orientation="horizontal", color=colors["histogram"], alpha=0.5)
    ax_hist.axis("off")

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))
    
    return fig

def plot_drift_landscape(
    drift_res: DriftResult,
    alpha: float,
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    plot_profile = resolve_experiment_plot_profile(
        "verification",
        "plot_drift_landscape",
        context=context,
        profile=profile,
    )
    theme = _setup_plot(theme)
    colors = _get_plot_colors(theme)
    
    states = drift_res.states
    if states.shape[1] != 2:
        raise ValueError("Heatmap only supported for N=2 systems.")

    q_max = int(states.max())
    grid_shape = (q_max + 1, q_max + 1)
    expected_states = {(i, j) for i in range(q_max + 1) for j in range(q_max + 1)}
    observed_states = {tuple(row) for row in states.tolist()}
    if observed_states != expected_states or len(states) != np.prod(grid_shape):
        raise ValueError("plot_drift_landscape requires a full dense 2D grid of states")
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
    cbar = plt.colorbar(
        im,
        ax=ax,
        label=plot_profile.axis_labels.get("colorbar", r"Generator Drift ${\cal L}V(Q)$"),
    )
    
    cbar.ax.yaxis.label.set_color(colors["text"])
    cbar.ax.tick_params(colors=colors["text"])

    ax.set_title(plot_profile.figure_title or rf"Drift Landscape ($\alpha = {alpha}$)")
    ax.set_xlabel(plot_profile.axis_labels.get("x", r"$Q_1$"))
    ax.set_ylabel(plot_profile.axis_labels.get("y", r"$Q_2$"))

    ax.contour(drift_grid.T, levels=[0.0], colors=colors["contour"], linewidths=2.0)

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))
    
    return fig

def plot_drift_vs_norm(
    drift_res: DriftResult,
    eps: float,
    R: float,
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    plot_profile = resolve_experiment_plot_profile(
        "verification",
        "plot_drift_vs_norm",
        context=context,
        profile=profile,
    )
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

    ax.set_title(plot_profile.figure_title or "Drift Verification")
    ax.set_xlabel(plot_profile.axis_labels.get("x", r"State Norm $|Q|_1$"))
    ax.set_ylabel(plot_profile.axis_labels.get("y", r"Generator Drift ${\cal L}V(Q)$"))
    ax.legend(loc="upper right")

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))
    
    return fig

def plot_policy_comparison(
    results_dict: dict[str, list[float]],
    metric_name: str,
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    plot_profile = resolve_experiment_plot_profile(
        None,
        "plot_policy_comparison",
        context=context,
        profile=profile,
    )
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
    ax.set_ylabel(plot_profile.axis_labels.get("y", metric_name))
    ax.set_title(plot_profile.figure_title or f"Policy Comparison: {metric_name}")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.05, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9, color=colors["text"])

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))
    
    return fig

def plot_alpha_sweep(
    alpha_values: np.ndarray,
    mean_q_matrix: np.ndarray,
    rho_labels: list[str],
    stationary_matrix: Optional[np.ndarray] = None,
    *,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
        stationary_matrix: Optional boolean matrix (rho x alpha), where False
            marks non-stationary cells that should be called out visually.
        save_path: Base path for saving (without extension)
        theme: Theme name ('dark' or 'publication')
        formats: Output formats ['png', 'pdf', 'svg']
    
    Returns:
        Matplotlib Figure object
    """
    plot_profile = resolve_experiment_plot_profile(
        "sweep",
        "plot_alpha_sweep",
        context=context,
        profile=profile,
    )
    theme = _setup_plot(theme)

    alpha_values = _as_1d_array(alpha_values, "alpha_values").astype(np.float64, copy=False)
    mean_q_matrix = np.asarray(mean_q_matrix, dtype=np.float64)
    if mean_q_matrix.ndim != 2:
        raise ValueError(f"mean_q_matrix must be 2D, got shape {mean_q_matrix.shape}")
    if mean_q_matrix.shape != (len(rho_labels), alpha_values.size):
        raise ValueError(
            "mean_q_matrix shape must match (len(rho_labels), len(alpha_values)); "
            f"got {mean_q_matrix.shape} vs ({len(rho_labels)}, {alpha_values.size})"
        )

    if stationary_matrix is None:
        stationary_matrix = np.ones_like(mean_q_matrix, dtype=bool)
    else:
        stationary_matrix = np.asarray(stationary_matrix, dtype=bool)
        if stationary_matrix.shape != mean_q_matrix.shape:
            raise ValueError(
                "stationary_matrix shape must match mean_q_matrix; "
                f"got {stationary_matrix.shape} vs {mean_q_matrix.shape}"
            )
    
    fig, ax = plt.subplots(figsize=(9, 6))

    palette = THEMES[theme].color_palette
    
    for i, rho_label in enumerate(rho_labels):
        color = palette[i % len(palette)]
        y_vals = mean_q_matrix[i, :]
        stable_mask = stationary_matrix[i, :]
        unstable_mask = ~stable_mask

        ax.plot(
            alpha_values,
            y_vals,
            marker="o",
            color=color,
            linewidth=2,
            markersize=6,
            label=rho_label,
            alpha=0.9,
        )

        if np.any(unstable_mask):
            ax.scatter(
                alpha_values[unstable_mask],
                y_vals[unstable_mask],
                s=64,
                marker="o",
                facecolors="white",
                edgecolors=color,
                linewidths=2.0,
                zorder=4,
            )
            ax.scatter(
                alpha_values[unstable_mask],
                y_vals[unstable_mask],
                s=36,
                marker="x",
                color=color,
                linewidths=1.5,
                zorder=5,
            )

    ax.set_xscale("log")
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.set_yscale("log")
    ax.set_title(plot_profile.figure_title or r"System Performance vs Routing Temperature ($\alpha$)")
    ax.set_xlabel(plot_profile.axis_labels.get("x", r"Inverse Temperature $\alpha$"))
    ax.set_ylabel(plot_profile.axis_labels.get("y", r"Expected Total Queue Length $\mathbb{E}[|Q|_1]$ (log scale)"))
    
    valid_y = mean_q_matrix[np.isfinite(mean_q_matrix) & (mean_q_matrix > 0)]
    if valid_y.size > 0:
        min_y = np.nanmin(valid_y)
        max_y = np.nanmax(valid_y)
        ax.set_ylim(min_y * 0.85, max_y * 3.8) # Significant log headroom to clear top legends
    else:
        ax.set_ylim(4.0, 400.0)

    import matplotlib.ticker as ticker
    target_ticks = [5, 10, 20, 50, 100, 200, 400]
    ax.set_yticks(target_ticks)
    ax.set_yticklabels([str(t) for t in target_ticks])
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    load_factor_legend = ax.legend(
        title=plot_profile.axis_labels.get("legend_title", "Load Factor"),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
    )

    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerBase

    class _OverlayMarkerHandler(HandlerBase):
        """Draw a centered circle+cross overlay for legend keys."""

        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            cx = xdescent + width / 2.0
            cy = ydescent + height / 2.0
            color = orig_handle.get("color", "black")

            circle = Line2D(
                [cx],
                [cy],
                marker="o",
                markersize=orig_handle.get("circle_size", 8),
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=orig_handle.get("circle_edgewidth", 1.5),
                linestyle="None",
                transform=trans,
            )
            cross = Line2D(
                [cx],
                [cy],
                marker="x",
                markersize=orig_handle.get("cross_size", 7),
                color=color,
                markeredgewidth=orig_handle.get("cross_edgewidth", 1.5),
                linestyle="None",
                transform=trans,
            )
            return [circle, cross]
    
    # Matching the visual weight (s=64 vs markersize=6)
    stationarity_handles = [
        Line2D([0], [0], marker="o", color="black", markersize=6, linewidth=0),
        {
            "color": "black",
            "circle_size": 8,
            "circle_edgewidth": 1.5,
            "cross_size": 7,
            "cross_edgewidth": 1.5,
        },
    ]
    stationarity_labels = ["Stationary", "Non-stationary"]
    status_legend = ax.legend(
        handles=stationarity_handles,
        labels=stationarity_labels,
        loc="upper right",
        bbox_to_anchor=(0.87, 1.0),
        title="Cell Status",
        handler_map={dict: _OverlayMarkerHandler()},
    )
    ax.add_artist(load_factor_legend)
    ax.add_artist(status_legend)

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))
    
    return fig

def plot_convergence(
    result: SimResult,
    save_path: str | Path | None = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    plot_profile = resolve_experiment_plot_profile(
        None,
        "plot_convergence",
        context=context,
        profile=profile,
    )
    theme = _setup_plot(theme)
    colors = _get_plot_colors(theme)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    t, cum_avg = running_average(result)

    if len(t) > 5000:
        step = len(t) // 5000
        t = t[::step]
        cum_avg = cum_avg[::step]

    ax.plot(t, cum_avg, color=colors["tertiary"], linewidth=2)
    ax.set_title(plot_profile.figure_title or "Running Average of Total Queue Length")
    ax.set_xlabel(plot_profile.axis_labels.get("x", "Time (t)"))
    ax.set_ylabel(plot_profile.axis_labels.get("y", r"$\frac{1}{t}\int_0^t |Q(s)|_1 ds$"))

    final_val = cum_avg[-1]
    ax.axhline(final_val, color=colors["contour"], linestyle="--", alpha=0.5,
               label=rf"Final: {final_val:.2f}")
    ax.legend(loc="lower right")

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))
    
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
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    plot_profile = resolve_experiment_plot_profile(
        "reinforce_check",
        "plot_gradient_scatter",
        context=context,
        profile=profile,
    )
    spec, theme, contour = _get_chart_style(ChartType.SCATTER, theme)

    fig, ax = plt.subplots(figsize=spec.figsize)

    c = z_scores if z_scores is not None else np.abs(rf_grads - fd_grads)
    sc = ax.scatter(
        fd_grads, rf_grads,
        c=c, cmap=spec.colormap, s=spec.marker_size,
        alpha=spec.line_alpha, edgecolors="none",
    )
    plt.colorbar(
        sc,
        ax=ax,
        label=plot_profile.axis_labels.get(
            "colorbar",
            "Z-score magnitude" if z_scores is not None else "|error|",
        ),
    )

    lims = [
        min(fd_grads.min(), rf_grads.min()),
        max(fd_grads.max(), rf_grads.max()),
    ]
    ax.plot(lims, lims, "--", color=contour, linewidth=spec.line_width, alpha=0.6, label="y = x")

    ax.set_xlabel(plot_profile.axis_labels.get("x", "Finite-Difference Gradient (ground truth)"))
    ax.set_ylabel(plot_profile.axis_labels.get("y", "REINFORCE Gradient (estimate)"))
    ax.set_title(plot_profile.figure_title or "Gradient Estimator Agreement")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")

    if summary_stats:
        text_parts = []
        if "cosine_similarity" in summary_stats:
            text_parts.append(f"cos = {summary_stats['cosine_similarity']:.4f}")
        if "relative_error" in summary_stats:
            text_parts.append(f"rel err = {summary_stats['relative_error']:.4f}")
        if "passed" in summary_stats:
            text_parts.append("PASSED" if summary_stats["passed"] else "FAILED")
        ax.text(
            0.02, 0.98, "\n".join(text_parts),
            transform=ax.transAxes, fontsize=spec.annotation_fontsize,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        )

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))

    return fig

def plot_stress_dashboard(
    scaling_data: dict,
    critical_data: dict,
    hetero_data: dict,
    *,
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    plot_profile = resolve_experiment_plot_profile(
        "stress",
        "plot_stress_dashboard",
        context=context,
        profile=profile,
    )
    spec, theme, contour = _get_chart_style(ChartType.STRESS_DASHBOARD, theme)

    fig, axes = plt.subplots(1, 3, figsize=spec.figsize)

    ax_a = axes[0]
    n_values = _as_1d_array(scaling_data["n_values"], "scaling_data.n_values").astype(np.float64, copy=False)
    mean_q_scaling = _as_1d_array(scaling_data["mean_q"], "scaling_data.mean_q").astype(np.float64, copy=False)
    _validate_equal_lengths(n_values=n_values, mean_q_scaling=mean_q_scaling)
    ax_a.plot(
        n_values, mean_q_scaling,
        marker=spec.marker_style, markersize=spec.marker_size,
        linewidth=spec.line_width, color=spec.palette[0],
    )
    ax_a.set_xlabel(plot_profile.axis_labels.get("scaling_x", "Number of Servers (N)"))
    ax_a.set_ylabel(plot_profile.axis_labels.get("scaling_y", r"$\mathbb{E}[|Q|_1]$"))
    ax_a.set_title(plot_profile.panel_titles.get("scaling", "(a) Scaling Test"))
    ax_a.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)
    if n_values.size <= 2:
        ax_a.text(
            0.03,
            0.95,
            f"{n_values.size}-point debug check",
            transform=ax_a.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=contour,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor=contour),
        )

    ax_b = axes[1]
    rho = _as_1d_array(critical_data["rho_values"], "critical_data.rho_values").astype(np.float64, copy=False)
    eq = _as_1d_array(critical_data["mean_q"], "critical_data.mean_q").astype(np.float64, copy=False)
    stationary = _as_1d_array(
        critical_data.get("stationary", [True] * len(rho)),
        "critical_data.stationary",
    ).astype(bool, copy=False)
    _validate_equal_lengths(rho=rho, eq=eq, stationary=stationary)
    colors_b = [spec.palette[2] if s else spec.palette[5] for s in stationary]
    ax_b.scatter(rho, eq, c=colors_b, s=spec.marker_size * 8, zorder=3)
    if rho.size > 1:
        ax_b.plot(rho, eq, linewidth=spec.line_width, color=spec.palette[4], alpha=0.7)
    ax_b.set_yscale("log")
    ax_b.set_xlabel(plot_profile.axis_labels.get("critical_x", r"Load Factor $\rho$"))
    ax_b.set_ylabel(plot_profile.axis_labels.get("critical_y", r"$\mathbb{E}[|Q|_1]$ (log)"))
    ax_b.set_title(plot_profile.panel_titles.get("critical", "(b) Critical Load"))
    ax_b.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)
    if rho.size == 1:
        ax_b.text(
            0.03,
            0.95,
            "single-rho debug check",
            transform=ax_b.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=contour,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor=contour),
        )

    ax_c = axes[2]
    scenario_names = hetero_data["scenario_names"]
    hetero_mean_q = _as_1d_array(hetero_data["mean_q"], "hetero_data.mean_q").astype(np.float64, copy=False)
    hetero_gini = _as_1d_array(hetero_data.get("gini", []), "hetero_data.gini").astype(np.float64, copy=False)
    if len(scenario_names) != hetero_mean_q.size or hetero_gini.size != hetero_mean_q.size:
        raise ValueError("hetero_data fields must have matching lengths")
    x_pos = np.arange(len(scenario_names))
    bar_colors = [spec.palette[i % len(spec.palette)] for i in range(len(x_pos))]
    bars = ax_c.bar(x_pos, hetero_mean_q, color=bar_colors,
             alpha=spec.fill_alpha, edgecolor=contour)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(scenario_names, rotation=15, ha="right", fontsize=7)
    ax_c.set_ylabel(plot_profile.axis_labels.get("heterogeneity_y", r"$\mathbb{E}[|Q|_1]$"))
    ax_c.set_title(plot_profile.panel_titles.get("heterogeneity", "(c) Heterogeneity"))
    ax_c.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)
    
    # Ensure vertical headroom for Gini labels
    ax_c.set_ylim(0, max(hetero_mean_q.max() * 1.25, 1.0))

    if plot_profile.semantic_flags.get("annotate_heterogeneity_gini", True):
        for idx, bar in enumerate(bars):
            ax_c.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(hetero_mean_q.max(), 1.0) * 0.02,
                f"Gini: {hetero_gini[idx]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=contour,
            )

    if plot_profile.legend_mode == "figure":
        legend_handles = [
            Line2D([0], [0], color=spec.palette[0], marker=spec.marker_style, linewidth=spec.line_width, label="Mean queue"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=spec.palette[2], markersize=8, label="Stationary"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=spec.palette[5], markersize=8, label="Unstable"),
        ]
        if rho.size > 1:
            legend_handles.append(
                Line2D([0], [0], color=spec.palette[4], linewidth=spec.line_width, label="Critical-load trend")
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=spec.palette[0],
                alpha=spec.fill_alpha,
                markersize=10,
                label="Scenario E[Q]",
            )
        )
        # Positioning legend below the suptitle to avoid overlap
        fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), 
                   bbox_to_anchor=(0.5, 0.93), frameon=True, fontsize=9)

    fig.suptitle(plot_profile.figure_title or "Stress Test Dashboard", y=0.98, fontsize=13, fontweight="bold")
    # Subplots start at 0.85 to clear both the title and the legend row
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))

    return fig

def plot_training_dashboard(
    metrics: dict,
    *,
    jsq_baseline: Optional[float] = None,
    random_baseline: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    plot_profile = resolve_experiment_plot_profile(
        "training",
        "plot_training_dashboard",
        context=context,
        profile=profile,
    )
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
    ax_a.set_title(plot_profile.panel_titles.get("performance", "(a) Base-Regime Diagnostic"))
    if plot_profile.legend_mode != "figure":
        ax_a.legend(loc="lower right", fontsize=7)
    ax_a.grid(True, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    ax_b = axes[0, 1]
    ax_b2 = ax_b.twinx()
    if "policy_loss" in metrics:
        ax_b.plot(epochs, metrics["policy_loss"], linewidth=spec.line_width,
                  color=TRAINING_SECONDARY, label="Policy Loss")
    if "value_loss" in metrics:
        ax_b2.plot(epochs, metrics["value_loss"], linewidth=spec.line_width,
                   color=TRAINING_CRITIC, alpha=0.8, label="Value Loss")
        ax_b2.set_ylabel("Value Loss", color=TRAINING_CRITIC)
        ax_b2.tick_params(axis="y", labelcolor=TRAINING_CRITIC)
    ax_b.set_ylabel("Policy Loss", color=TRAINING_SECONDARY)
    ax_b.tick_params(axis="y", labelcolor=TRAINING_SECONDARY)
    ax_b.set_title(plot_profile.panel_titles.get("loss", "(b) Loss Curves"))
    if plot_profile.legend_mode != "figure":
        handles, labels = ax_b.get_legend_handles_labels()
        handles2, labels2 = ax_b2.get_legend_handles_labels()
        if handles or handles2:
            ax_b.legend(handles + handles2, labels + labels2, loc="upper right", fontsize=7)
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
    ax_c.set_title(plot_profile.panel_titles.get("critic", "(c) Critic Quality"))
    if plot_profile.legend_mode != "figure":
        ax_c.legend(loc="lower right", fontsize=7)
    ax_c.grid(True, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    ax_d = axes[1, 1]
    if "value_grad_norm" in metrics:
        ax_d.plot(epochs, metrics["value_grad_norm"], linewidth=spec.line_width,
                  color=TRAINING_CRITIC, alpha=0.8, label="Value grad norm")
    ax_d.set_ylabel("Gradient Norm")
    ax_d2 = ax_d.twinx()
    if "entropy" in metrics:
        ax_d2.plot(epochs, metrics["entropy"], linewidth=1.0,
                   color=TRAINING_ENTROPY, alpha=0.7, label="Entropy")
        ax_d2.set_ylabel("Entropy", color=TRAINING_ENTROPY)
        ax_d2.tick_params(axis="y", labelcolor=TRAINING_ENTROPY)
    ax_d.set_xlabel("Epoch")
    ax_d.set_title(plot_profile.panel_titles.get("gradients", "(d) Gradient Health"))
    if plot_profile.legend_mode != "figure":
        handles, labels = ax_d.get_legend_handles_labels()
        handles2, labels2 = ax_d2.get_legend_handles_labels()
        if handles or handles2:
            ax_d.legend(handles + handles2, labels + labels2, loc="upper right", fontsize=7)
    ax_d.grid(True, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    if plot_profile.legend_mode == "figure":
        legend_handles = []
        legend_labels = []
        for ax in (ax_a, ax_b, ax_b2, ax_c, ax_d, ax_d2):
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
        if metrics.get("final_eval_mean") is not None and metrics.get("final_eval_count") is not None:
            final_label = f"Final deterministic eval (n={int(metrics['final_eval_count'])})"
            final_handle = Line2D(
                [0],
                [0],
                marker="D",
                color="#d95f02",
                linestyle="None",
                markersize=8,
                label=final_label,
            )
            legend_handles.append(final_handle)
            legend_labels.append(final_label)
            ax_a.scatter(
                epochs[-1] + 0.15,
                float(metrics["final_eval_mean"]),
                marker="D",
                s=55,
                color="#d95f02",
                zorder=4,
                label=final_label,
            )
        if len(epochs) > 1:
            fig.legend(legend_handles, legend_labels, loc="upper center", 
                       ncol=min(len(legend_handles), 5), bbox_to_anchor=(0.5, 0.94),
                       frameon=True, fontsize=8)

    footer_parts = []
    if metrics.get("run_label"):
        footer_parts.append(f"{metrics['run_label']} run")
    if metrics.get("train_epochs") is not None:
        footer_parts.append(f"{int(metrics['train_epochs'])} training epochs")
    if metrics.get("final_eval_mean") is not None and metrics.get("final_eval_std") is not None:
        footer_parts.append(
            f"final deterministic eval = {float(metrics['final_eval_mean']):.1f}% +/- {float(metrics['final_eval_std']):.1f}%"
        )
    if footer_parts:
        fig.text(0.5, 0.015, " | ".join(footer_parts), ha="center", va="bottom", fontsize=8, color=contour)
    if len(epochs) <= 1:
        fig.text(0.5, 0.985, "Partial history", ha="center", va="top", fontsize=8, color=contour)

    fig.suptitle(plot_profile.figure_title or "REINFORCE Training Dashboard", y=0.98, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.85])

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))

    return fig


def plot_ablation_training_curve(
    metrics: dict,
    *,
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
) -> plt.Figure:
    """Two-panel ablation-specific training plot for one variant."""
    plot_profile = resolve_experiment_plot_profile(
        "ablation",
        "plot_ablation_training_curve",
        context=context,
        profile=profile,
    )
    spec, theme, contour = _get_chart_style(ChartType.TRAINING_DASHBOARD, theme)

    epochs = _as_1d_array(metrics.get("epoch", []), "metrics.epoch").astype(np.float64, copy=False)
    training_loss = _as_1d_array(metrics.get("training_loss", []), "metrics.training_loss").astype(np.float64, copy=False)
    performance = _as_1d_array(
        metrics.get("performance_index", metrics.get("base_regime_index_proxy", [])),
        "metrics.performance_index",
    ).astype(np.float64, copy=False)
    _validate_equal_lengths(epoch=epochs, training_loss=training_loss, performance=performance)
    if not np.all(np.isfinite(training_loss)) or not np.all(np.isfinite(performance)):
        raise ValueError("ablation training metrics must contain only finite values")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    ax_loss, ax_perf = axes

    ax_loss.plot(
        epochs,
        training_loss,
        marker="o",
        markersize=max(spec.marker_size - 1, 3),
        linewidth=spec.line_width,
        color=TRAINING_SECONDARY,
        alpha=spec.line_alpha,
    )
    ax_loss.set_xlabel(plot_profile.axis_labels.get("x", "Epoch"))
    ax_loss.set_ylabel(plot_profile.axis_labels.get("loss_y", "Policy Loss"))
    ax_loss.set_title(plot_profile.panel_titles.get("loss", "(a) Training Objective"))
    ax_loss.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    ax_perf.plot(
        epochs,
        performance,
        marker="s",
        markersize=max(spec.marker_size - 1, 3),
        linewidth=spec.line_width,
        color=TRAINING_PRIMARY,
        alpha=spec.line_alpha,
    )
    ax_perf.set_xlabel(plot_profile.axis_labels.get("x", "Epoch"))
    ax_perf.set_ylabel(plot_profile.axis_labels.get("performance_y", "Base-Regime PI Proxy (%)"))
    ax_perf.set_title(plot_profile.panel_titles.get("performance", "(b) Performance Proxy"))
    ax_perf.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    meta_lines: list[str] = []
    if metrics.get("variant_label"):
        meta_lines.append(str(metrics["variant_label"]))
    if metrics.get("preprocessing") is not None:
        meta_lines.append(f"preprocessing = {metrics['preprocessing']}")
    if metrics.get("init_type") is not None:
        meta_lines.append(f"init_type = {metrics['init_type']}")
    if metrics.get("train_epochs") is not None:
        meta_lines.append(f"epochs = {metrics['train_epochs']}")
    meta_lines.append(f"final loss = {float(training_loss[-1]):.4f}")
    meta_lines.append(f"final PI = {float(performance[-1]):.1f}%")

    fig.text(
        0.99,
        0.02,
        "\n".join(meta_lines),
        ha="right",
        va="bottom",
        fontsize=8,
        color=contour,
    )
    fig.suptitle(plot_profile.figure_title or "Ablation Variant Training Curve", fontsize=13, fontweight="bold")

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))

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
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    plot_profile = resolve_experiment_plot_profile(
        "stats",
        "plot_raincloud",
        context=context,
        profile=profile,
    )
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
                    -np.inf,
                    m,
                )
                body.set_facecolor(color)
                body.set_edgecolor(contour)
                body.set_alpha(spec.fill_alpha)

        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(data))
        offset = 0.15
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
    ax.set_ylabel(plot_profile.axis_labels.get("y", y_label))
    # Migration to figure-level title to avoid overlap with stats bracket
    fig.suptitle(plot_profile.figure_title or f"{group_a_label} vs {group_b_label}: Distribution Comparison",
                 y=0.98, fontsize=13, fontweight="bold")
    ax.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha, axis="y")

    if stats:
        y_max = max(group_a_data.max(), group_b_data.max())
        
        # Explicitly set Y-axis limit to provide clear headroom for the stats bracket
        y_headroom = y_max * 1.30
        ax.set_ylim(top=y_headroom)
        
        bracket_y = y_max * 1.12 # Floating above the distribution
        ax.plot([0, 0, 1, 1], [bracket_y * 0.98, bracket_y, bracket_y, bracket_y * 0.98],
                color=contour, linewidth=1.2)
        parts = []
        if "p_value" in stats:
            p = stats["p_value"]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            if p < 0.0001:
                parts.append(f"p < 0.0001 {stars}")
            else:
                parts.append(f"p = {p:.4f} {stars}")
        if "cohen_d" in stats:
            parts.append(f"d = {stats['cohen_d']:.2f}")
        if "improvement_pct" in stats:
            parts.append(f"Δ = {stats['improvement_pct']:.1f}%")
        ax.text(0.5, bracket_y * 1.05, "  |  ".join(parts),
                ha="center", fontsize=spec.annotation_fontsize,
                fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))

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
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
) -> plt.Figure:
    """Annotated diverging heatmap for improvement-ratio grids.

    Used by gen_sweep.py for GibbsQ/Neural improvement ratios.

    Parameters
    ----------
    grid : array (n_y, n_x)
        Improvement ratio values.  >1.0 means GibbsQ wins.
    x_labels, y_labels : lists of str
        Tick labels for each axis.
    center : float
        Colormap normalisation midpoint (1.0 = break-even).
    """
    plot_profile = resolve_experiment_plot_profile(
        "generalize",
        "plot_improvement_heatmap",
        context=context,
        profile=profile,
    )
    center = float(plot_profile.thresholds.get("center", center))
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
    cbar = plt.colorbar(
        im,
        ax=ax,
        label=plot_profile.axis_labels.get("colorbar", "Improvement Ratio (GibbsQ / Neural)"),
    )

    cbar.ax.set_title("GibbsQ\nSuperior", fontsize=spec.annotation_fontsize - 1, fontweight="bold", pad=12)
    cbar.ax.text(0.5, -0.015, "Neural\nSuperior", transform=cbar.ax.transAxes, 
                 ha="center", va="top", fontsize=spec.annotation_fontsize - 1, fontweight="bold")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            text_color = "white" if abs(val - center) > vmax * 0.45 else "black"
            label = f"{val:{spec.value_format}}x"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=spec.annotation_fontsize + 1,
                color=text_color,
                fontweight="bold",
            )

    ax.grid(False)
    ax.set_xticks(np.arange(grid.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(plot_profile.axis_labels.get("x", x_axis_name))
    ax.set_ylabel(plot_profile.axis_labels.get("y", y_axis_name))
    ax.set_title(plot_profile.figure_title or "Generalization Sweep: Improvement Ratio")

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))

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
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    if not variant_names:
        raise ValueError("variant_names must be non-empty")
    if len(variant_names) != len(mean_values) or len(variant_names) != len(se_values):
        raise ValueError("variant_names, mean_values, and se_values must have matching lengths")

    plot_profile = resolve_experiment_plot_profile(
        "ablation",
        "plot_ablation_bars",
        context=context,
        profile=profile,
    )
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
    max_label_y = 0.0
    for i, bar in enumerate(bars):
        h = bar.get_height()
        val_y = h + se_values[i] + 0.02 * best_val
        ax.text(
            bar.get_x() + bar.get_width() / 2, val_y,
            f"{h:{spec.value_format}}",
            ha="center", va="bottom", fontsize=spec.annotation_fontsize,
        )
        max_label_y = max(max_label_y, val_y + 0.05 * best_val)

        if i > 0 and best_val > 0:
            delta = ((h - best_val) / best_val) * 100
            sign = "+" if delta > 0 else ""
            delta_y = h + se_values[i] + 0.08 * best_val
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                delta_y,
                f"{sign}{delta:.1f}%",
                ha="center", va="bottom",
                fontsize=spec.annotation_fontsize - 1,
                color="#D55E00" if delta > 0 else "#009E73",
                fontweight="bold",
            )
            max_label_y = max(max_label_y, delta_y + 0.12 * best_val)

    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=20, ha="right")
    ax.set_ylabel(plot_profile.axis_labels.get("y", metric_name))
    ax.set_title(plot_profile.figure_title or "Ablation Study: Component Contributions")
    ax.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha, axis="y")

    # Explicitly set ylim to avoid clipping labels
    if max_label_y > 0:
        ax.set_ylim(top=max_label_y)

    fig.tight_layout()

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])

    return fig


def _wrap_variant_label(label: str) -> str:
    replacements = {
        "Ablated: No Log-Norm": "Ablated:\nNo Log-Norm",
        "Ablated: No Zero-Init": "Ablated:\nNo Zero-Init",
        "Uniform Routing (Baseline)": "Uniform Routing\n(Baseline)",
    }
    return replacements.get(label, label)


def _style_ablation_axis(
    ax: plt.Axes,
    labels: list[str],
    *,
    show_grid: bool,
    show_xticks: bool,
    plot_profile: ExperimentPlotProfile,
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.arange(len(labels)))
    if show_xticks:
        ax.set_xticklabels([_wrap_variant_label(label) for label in labels], rotation=0, ha="center")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    if show_grid:
        ax.grid(True, axis="y", linestyle=(0, (3, 3)), alpha=0.28, linewidth=0.8)
    else:
        ax.grid(False)
    ax.set_ylabel(plot_profile.axis_labels.get("y", r"$\mathbb{E}[Q_{total}]$"))


def plot_ablation_dual_panel(
    variant_names: List[str],
    mean_values: List[float],
    se_values: List[float],
    *,
    metric_name: str = r"$\mathbb{E}[Q_{total}]$",
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
) -> plt.Figure:
    """Premium dual-panel ablation chart for highly skewed comparisons."""
    if not variant_names:
        raise ValueError("variant_names must be non-empty")
    if len(variant_names) != len(mean_values) or len(variant_names) != len(se_values):
        raise ValueError("variant_names, mean_values, and se_values must have matching lengths")

    plot_profile = resolve_experiment_plot_profile(
        "ablation",
        "plot_ablation_bars",
        context=context,
        profile=profile,
    )
    spec, theme, contour = _get_chart_style(ChartType.BAR_COMPARISON, theme)

    means = np.asarray(mean_values, dtype=np.float64)
    ses = np.asarray(se_values, dtype=np.float64)
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(ses)):
        raise ValueError("ablation dual-panel plot requires finite values")

    x = np.arange(len(variant_names))
    colors = [spec.palette[i % len(spec.palette)] for i in range(len(variant_names))]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12.5, 9.1),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.3, 2.2], "hspace": 0.05},
    )

    bar_kwargs = {
        "yerr": ses,
        "capsize": spec.error_bar_capsize,
        "color": colors,
        "alpha": 0.88,
        "edgecolor": contour,
        "linewidth": 1.2,
        "zorder": 3,
    }
    bars_top = ax_top.bar(x, means, **bar_kwargs)
    bars_bottom = ax_bottom.bar(x, means, **bar_kwargs)

    top_max = float(np.max(means + ses))
    lower_max = float(np.max(means[:3] + ses[:3])) if len(means) > 1 else top_max
    lower_min = float(np.min(np.maximum(means[:3] - ses[:3], 0.0))) if len(means) > 1 else 0.0
    lower_span = max(lower_max - lower_min, 1.0)

    ax_top.set_ylim(0.0, top_max * 1.35)
    ax_bottom.set_ylim(max(0.0, lower_min - 0.45 * lower_span), lower_max + 0.85 * lower_span)
    bottom_ylim = ax_bottom.get_ylim()

    _style_ablation_axis(ax_top, variant_names, show_grid=True, show_xticks=False, plot_profile=plot_profile)
    _style_ablation_axis(ax_bottom, variant_names, show_grid=True, show_xticks=True, plot_profile=plot_profile)
    ax_bottom.set_xlabel("Variant")
    ax_top.set_title(plot_profile.figure_title or "Ablation Study: Component Contributions", pad=14, fontsize=16)

    ax_top.text(
        0.015,
        0.92,
        "Full scale",
        transform=ax_top.transAxes,
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": contour, "alpha": 0.92},
    )
    ax_bottom.text(
        0.015,
        0.92,
        "Zoom on learned variants",
        transform=ax_bottom.transAxes,
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": contour, "alpha": 0.92},
    )

    top_value_pad = top_max * 0.02
    bottom_value_pad = max(lower_span * 0.35, 0.25)
    for idx, bar in enumerate(bars_top):
        height = float(bar.get_height())
        y = height + ses[idx] + top_value_pad
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{height:{spec.value_format}}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            clip_on=False,
            zorder=5,
        )

    full_model = float(means[0])
    for idx, bar in enumerate(bars_bottom[: min(3, len(bars_bottom))]):
        height = float(bar.get_height())
        value_y = height + ses[idx] + lower_span * 0.06
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2,
            value_y,
            f"{height:{spec.value_format}}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            clip_on=False,
            zorder=5,
        )
        if idx > 0 and full_model > 0:
            delta = ((height - full_model) / full_model) * 100.0
            delta_color = "#D55E00" if delta > 0 else "#009E73"
            sign = "+" if delta > 0 else ""
            ax_bottom.text(
                bar.get_x() + bar.get_width() / 2,
                value_y + lower_span * 0.12,
                f"{sign}{delta:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=delta_color,
                clip_on=False,
                zorder=5,
            )

    if len(bars_bottom) >= 4:
        bars_bottom[3].set_visible(False)
        ax_bottom.annotate(
            "Off-scale\nsee top panel",
            xy=(x[3], bottom_ylim[1] - 0.03 * lower_span),
            xytext=(x[3], bottom_ylim[1] - 0.28 * lower_span),
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#999999",
                "alpha": 0.96,
            },
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#999999",
                "linewidth": 1.0,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=6,
        )

    if len(means) >= 4 and full_model > 0:
        baseline_delta = ((means[3] - full_model) / full_model) * 100.0
        label_y = means[3] + ses[3] + top_value_pad
        ax_top.annotate(
            f"Baseline: +{baseline_delta:.1f}% vs Full Model",
            xy=(x[3], label_y + top_max * 0.04),
            xytext=(x[3], label_y + top_max * 0.10),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#D55E00",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#fff7f0",
                "edgecolor": "#D55E00",
                "linewidth": 1.0,
                "alpha": 0.96,
            },
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 1.2,
                "color": "#D55E00",
                "shrinkA": 4,
                "shrinkB": 6,
            },
            zorder=6,
        )

    kwargs = dict(transform=ax_top.transAxes, color=contour, clip_on=False, linewidth=1.1)
    ax_top.plot((-0.015, +0.015), (-0.015, +0.015), **kwargs)
    ax_top.plot((0.985, 1.015), (-0.015, +0.015), **kwargs)
    kwargs["transform"] = ax_bottom.transAxes
    ax_bottom.plot((-0.015, +0.015), (1 - 0.015, 1 + 0.015), **kwargs)
    ax_bottom.plot((0.985, 1.015), (1 - 0.015, 1 + 0.015), **kwargs)

    legend_handles = [
        Line2D([0], [0], color=colors[0], lw=8, label="Full model"),
        Line2D([0], [0], color=colors[1], lw=8, label="Ablation: no log-normalization"),
        Line2D([0], [0], color=colors[2], lw=8, label="Ablation: no zero-initialization"),
        Line2D([0], [0], color=colors[3], lw=8, label="Uniform routing baseline"),
    ]
    ax_top.legend(handles=legend_handles, loc="upper center", frameon=True, ncol=2, fontsize=10)

    fig.align_ylabels([ax_top, ax_bottom])

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"])

    return fig

def _deprecated_plot_critical_load_legacy(
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
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))

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
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
) -> plt.Figure:
    """Dual-line log-scale divergence plot for critical load comparison."""
    rho_values = _as_1d_array(rho_values, "rho_values").astype(np.float64, copy=False)
    neural_eq = _as_1d_array(neural_eq, "neural_eq").astype(np.float64, copy=False)
    gibbs_eq = _as_1d_array(gibbs_eq, "gibbs_eq").astype(np.float64, copy=False)
    _validate_equal_lengths(rho_values=rho_values, neural_eq=neural_eq, gibbs_eq=gibbs_eq)
    if not np.all(np.isfinite(rho_values)):
        raise ValueError("rho_values must contain only finite values")
    if not np.all(np.isfinite(neural_eq)) or not np.all(np.isfinite(gibbs_eq)):
        raise ValueError("neural_eq and gibbs_eq must contain only finite values")
    if np.any(neural_eq <= 0.0) or np.any(gibbs_eq <= 0.0):
        raise ValueError("neural_eq and gibbs_eq must be strictly positive for log-scale plots")

    plot_profile = resolve_experiment_plot_profile(
        "critical",
        "plot_critical_load",
        context=context,
        profile=profile,
    )
    critical_rho = float(plot_profile.thresholds.get("critical_rho", critical_rho))
    spec, theme, contour = _get_chart_style(ChartType.LINE_SERIES, theme)

    fig, ax = plt.subplots(figsize=spec.figsize, layout="constrained")

    ax.plot(
        rho_values,
        gibbs_eq,
        marker="o",
        markersize=spec.marker_size,
        linewidth=spec.line_width,
        color=PAIR_GIBBSQ,
        label="GibbsQ",
        alpha=spec.line_alpha,
    )
    ax.plot(
        rho_values,
        neural_eq,
        marker="s",
        markersize=spec.marker_size,
        linewidth=spec.line_width,
        color=PAIR_NEURAL,
        label="N-GibbsQ",
        alpha=spec.line_alpha,
    )

    ax.set_yscale("log")
    ax.set_xlabel(plot_profile.axis_labels.get("x", r"Load Factor $\rho$"))
    ax.set_ylabel(plot_profile.axis_labels.get("y", r"$\mathbb{E}[|Q|_1]$ (log scale)"))
    ax.set_title(plot_profile.figure_title or r"Critical Load: E[Q] vs $\rho$ Near Stability Boundary")

    import matplotlib.ticker as ticker
    target_ticks = [15, 20, 30, 40, 50, 60, 80]
    ax.set_yticks(target_ticks)
    ax.set_yticklabels([str(t) for t in target_ticks])
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    min_y = min(np.nanmin(neural_eq), np.nanmin(gibbs_eq))
    max_y = max(np.nanmax(neural_eq), np.nanmax(gibbs_eq))
    ax.set_ylim(min_y * 0.8, max_y * 1.25)

    if len(rho_values) == 1:
        v = float(rho_values[0])
        ax.set_xlim(v - 0.01, v + 0.01)

    if plot_profile.semantic_flags.get("show_near_critical_band", True):
        ax.axvspan(
            critical_rho,
            ax.get_xlim()[1],
            alpha=0.08,
            color="#D55E00",
            label=r"Near-critical ($\rho \geq %s$)" % critical_rho,
        )

    ax.legend(loc="upper left")
    ax.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))

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
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
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
    plot_profile = resolve_experiment_plot_profile(
        "policy",
        "plot_tier_comparison_bars",
        context=context,
        profile=profile,
    )
    spec, theme, contour = _get_chart_style(ChartType.BAR_COMPARISON, theme)
    palette = spec.palette
    
    tier_colors = {
        1: "#999999",
        2: "#DDAA33",
        3: "#CC6677",
        4: "#7A7A7A",
        5: "#117733",
    }
    colors = [tier_colors.get(t, palette[0]) for t in tiers]

    fig, ax = plt.subplots(figsize=spec.figsize)
    x = np.arange(len(labels))

    ax.bar(x, q_values, yerr=q_errors, color=colors, alpha=spec.fill_alpha,
           capsize=spec.error_bar_capsize, edgecolor=palette[0])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(plot_profile.axis_labels.get("y", 'Expected Total Queue Length E[Q_total]'))
    ax.set_title(plot_profile.figure_title or 'Corrected Policy Comparison')
    ax.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha, axis='y')
    ax.text(
        0.01,
        0.99,
        "Lower is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=contour,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor=contour),
    )

    from matplotlib.patches import Patch
    tier_labels = {
        1: "Tier 1: Blind",
        2: "Tier 2: Structural",
        3: "Tier 3: UAS",
        4: "Tier 4: Blind/Fixed-weight",
        5: "Tier 5: Proposed",
    }
    legend_elements = [Patch(facecolor=c, label=tier_labels.get(t, f'Tier {t}'), alpha=spec.fill_alpha)
                       for t, c in sorted(tier_colors.items()) if t in tiers]
    # Explicitly set ylim to avoid legend overlap with tall bars (Uniform)
    max_y = max([float(v) + float(e) for v, e in zip(q_values, q_errors)]) if q_values else 1.0
    ax.set_ylim(0, max_y * 1.35)  # 35% headroom for top-right legend

    ax.legend(handles=legend_elements, loc='upper right')

    fig.tight_layout()

    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))
        
        
    return fig

def plot_policy_dual_panel(
    labels: List[str],
    q_values: List[float],
    q_errors: List[float],
    tiers: List[int],
    *,
    metric_name: str = r"$\mathbb{E}[Q_{total}]$",
    save_path: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    formats: Optional[List[str]] = None,
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
) -> plt.Figure:
    """Premium dual-panel policy comparison chart for highly skewed distributions."""
    if not labels or len(labels) != len(q_values) or len(labels) != len(q_errors):
        raise ValueError("labels, q_values, and q_errors must have matching non-empty lengths")

    plot_profile = resolve_experiment_plot_profile(
        "policy",
        "plot_tier_comparison_bars",
        context=context,
        profile=profile,
    )
    spec, theme, contour = _get_chart_style(ChartType.BAR_COMPARISON, theme)
    palette = spec.palette
    
    tier_colors = {
        1: "#999999",
        2: "#DDAA33",
        3: "#CC6677",
        4: "#7A7A7A",
        5: "#117733",
    }
    colors = [tier_colors.get(t, palette[0]) for t in tiers]
    
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12.5, 9.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.3, 2.2], "hspace": 0.05},
    )
    
    x = np.arange(len(labels))
    bar_kwargs = {
        "yerr": q_errors,
        "capsize": spec.error_bar_capsize,
        "color": colors,
        "alpha": spec.fill_alpha,
        "edgecolor": contour,
        "linewidth": 1.2,
        "zorder": 3,
    }
    bars_top = ax_top.bar(x, q_values, **bar_kwargs)
    bars_bottom = ax_bottom.bar(x, q_values, **bar_kwargs)
    
    top_max = float(np.max(np.asarray(q_values) + np.asarray(q_errors)))
    
    median_val = np.median(q_values)
    zoom_vals = [v + e for v, e in zip(q_values, q_errors) if v < 10 * median_val]
    zoom_max = float(np.max(zoom_vals)) if zoom_vals else top_max
    
    ax_top.set_ylim(0, top_max * 1.35)
    ax_bottom.set_ylim(0, zoom_max * 1.55)
    bottom_ylim = ax_bottom.get_ylim()
    
    for ax in (ax_top, ax_bottom):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle=(0, (3, 3)), alpha=0.28, linewidth=0.8)
        ax.set_ylabel(plot_profile.axis_labels.get("y", r"Expected Total Queue Length $\mathbb{E}[Q_{total}]$"))
    
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels, rotation=45, ha="right")
    ax_top.tick_params(axis="x", length=0)
    
    ax_top.set_title(plot_profile.figure_title or "Corrected Policy Comparison", pad=14, fontsize=16)
    
    ax_top.text(
        0.015, 0.92, "Full scale", transform=ax_top.transAxes, fontsize=10, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": contour, "alpha": 0.92},
    )
    ax_bottom.text(
        0.015, 0.92, "Zoom on competitive policies", transform=ax_bottom.transAxes, fontsize=10, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": contour, "alpha": 0.92},
    )
    
    ax_top.text(
        0.985, 0.92, "Lower is better", transform=ax_top.transAxes, ha="right", fontsize=9, color=contour,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor=contour),
    )
    
    top_value_pad = top_max * 0.03
    for idx, bar in enumerate(bars_top):
        h = float(bar.get_height())
        if h > bottom_ylim[1]:
            ax_top.text(
                bar.get_x() + bar.get_width() / 2, h + q_errors[idx] + top_value_pad,
                f"{h:{spec.value_format}}", ha="center", va="bottom", fontsize=10, fontweight="semibold", zorder=5,
            )
        
    for idx, (bar, h) in enumerate(zip(bars_bottom, q_values)):
        if h > bottom_ylim[1]:
            bar.set_visible(False)
            ax_bottom.annotate(
                "Off-scale\nsee top panel",
                xy=(x[idx], bottom_ylim[1] - 0.03 * zoom_max),
                xytext=(x[idx], bottom_ylim[1] - 0.28 * zoom_max),
                ha="center", va="center", fontsize=9, color="#666666",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#999999", alpha=0.96),
                arrowprops=dict(arrowstyle="-|>", color="#999999", linewidth=1.0, shrinkA=4, shrinkB=4),
                zorder=6,
            )
        else:
            value_y = h + q_errors[idx] + zoom_max * 0.15
            ax_bottom.text(
                bar.get_x() + bar.get_width() / 2, value_y,
                f"{h:{spec.value_format}}", ha="center", va="bottom", fontsize=10, fontweight="semibold", zorder=5,
            )
            
    kwargs = dict(transform=ax_top.transAxes, color=contour, clip_on=False, linewidth=1.1)
    ax_top.plot((-0.015, +0.015), (-0.015, +0.015), **kwargs)
    ax_top.plot((0.985, 1.015), (-0.015, +0.015), **kwargs)
    kwargs["transform"] = ax_bottom.transAxes
    ax_bottom.plot((-0.015, +0.015), (1 - 0.015, 1 + 0.015), **kwargs)
    ax_bottom.plot((0.985, 1.015), (1 - 0.015, 1 + 0.015), **kwargs)
    
    from matplotlib.patches import Patch
    tier_labels_str = {
        1: "Tier 1: Blind", 2: "Tier 2: Structural", 3: "Tier 3: UAS",
        4: "Tier 4: Blind/Fixed-weight", 5: "Tier 5: Proposed",
    }
    legend_elements = [Patch(facecolor=c, label=tier_labels_str.get(t, f'Tier {t}'), alpha=spec.fill_alpha)
                       for t, c in sorted(tier_colors.items()) if t in tiers]
    ax_top.legend(handles=legend_elements, loc="upper center", frameon=True, ncol=len(legend_elements)//2 or 1, fontsize=10)
    
    fig.align_ylabels([ax_top, ax_bottom])
    
    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))
        
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
    profile: str | ExperimentPlotProfile | None = None,
    context: ExperimentPlotContext | None = None,
) -> plt.Figure:
    """Proposed grid analysis: Performance Envelope and Generalization Efficiency.
    
    Dual-panel plot. Left: Log-scale E[Q] comparison vs rho. 
    Right: Performance Index (%) relative to JSQ and Uniform vs rho.
    """
    plot_profile = resolve_experiment_plot_profile(
        "policy",
        "plot_platinum_grid",
        context=context,
        profile=profile,
    )
    spec, theme, contour = _get_chart_style(ChartType.LINE_SERIES, theme)

    rho_values = _as_1d_array(rho_values, "rho_values").astype(np.float64, copy=False)
    uniform_eq = _as_1d_array(uniform_eq, "uniform_eq").astype(np.float64, copy=False)
    neural_eq = _as_1d_array(neural_eq, "neural_eq").astype(np.float64, copy=False)
    jsq_eq = _as_1d_array(jsq_eq, "jsq_eq").astype(np.float64, copy=False)
    performance_index = _as_1d_array(performance_index, "performance_index").astype(np.float64, copy=False)
    _validate_equal_lengths(
        rho_values=rho_values,
        uniform_eq=uniform_eq,
        neural_eq=neural_eq,
        jsq_eq=jsq_eq,
        performance_index=performance_index,
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(rho_values, uniform_eq, linestyle='--', marker='x', color='#95a5a6',
             linewidth=spec.line_width, label='Uniform')
    ax1.plot(rho_values, neural_eq, linestyle='-', marker='o',
             linewidth=spec.line_width + 0.5, color=PAIR_NEURAL, label='N-GibbsQ (Proposed)')
    ax1.plot(rho_values, jsq_eq, linestyle='-.', marker='s',
             linewidth=spec.line_width, color=PAIR_GIBBSQ, label='JSQ (Optimal)')
    ax1.set_yscale('log')
    ax1.set_xlabel(plot_profile.axis_labels.get("envelope_x", r'Load Factor $\rho$'))
    ax1.set_ylabel(plot_profile.axis_labels.get("envelope_y", r'$\mathbb{E}[|Q|_1]$ (Log Scale)'))
    ax1.set_title(plot_profile.panel_titles.get("envelope", 'Performance Envelope'))
    ax1.legend()
    ax1.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)
    
    ax2.plot(rho_values, performance_index, linestyle='-', marker='D',
             linewidth=spec.line_width + 0.5, color=spec.palette[0])
    ax2.axhline(100, color=PAIR_GIBBSQ, linestyle='--', alpha=0.5, label='JSQ Parity')
    ax2.axhline(0, color='#95a5a6', linestyle='--', alpha=0.5, label='Uniform Parity')
    ax2.set_ylim(-10, 110)
    ax2.set_xlabel(plot_profile.axis_labels.get("efficiency_x", r'Load Factor $\rho$'))
    ax2.set_ylabel(plot_profile.axis_labels.get("efficiency_y", 'Performance Index (%)'))
    ax2.set_title(plot_profile.panel_titles.get("efficiency", 'Generalization Efficiency'))
    ax2.legend()
    ax2.grid(spec.grid_visible, linestyle=spec.grid_style, alpha=spec.grid_alpha)
    
    fig.suptitle(plot_profile.figure_title or "Proposed Grid Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    
    if save_path:
        save_chart(fig, Path(save_path), formats or list(plot_profile.preferred_formats))
        
    return fig

