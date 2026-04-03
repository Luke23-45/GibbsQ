"""Experiment-aware plotting profiles.

This layer keeps the public plotting API stable while letting experiment
drivers declare semantic intent such as thresholds, annotation language,
panel titles, and legend strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

__all__ = [
    "ExperimentPlotContext",
    "ExperimentPlotProfile",
    "resolve_experiment_plot_profile",
]


@dataclass(frozen=True)
class ExperimentPlotContext:
    """Semantic metadata supplied by experiment runners."""

    experiment_id: str | None = None
    chart_name: str | None = None
    semantic_overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentPlotProfile:
    """Semantic plotting behavior for one experiment/chart combination."""

    experiment_id: str = "generic"
    chart_name: str | None = None
    figure_title: str | None = None
    panel_titles: Mapping[str, str] = field(default_factory=dict)
    axis_labels: Mapping[str, str] = field(default_factory=dict)
    legend_mode: str = "axes"
    annotations_mode: str = "default"
    preferred_formats: tuple[str, ...] = ("png", "pdf")
    thresholds: Mapping[str, float] = field(default_factory=dict)
    semantic_flags: Mapping[str, Any] = field(default_factory=dict)

    def merged(self, overrides: Mapping[str, Any]) -> "ExperimentPlotProfile":
        """Create a copy with context overrides applied."""
        if not overrides:
            return self
        return replace(
            self,
            figure_title=overrides.get("figure_title", self.figure_title),
            legend_mode=overrides.get("legend_mode", self.legend_mode),
            annotations_mode=overrides.get("annotations_mode", self.annotations_mode),
            preferred_formats=tuple(overrides.get("preferred_formats", self.preferred_formats)),
            panel_titles={**self.panel_titles, **dict(overrides.get("panel_titles", {}))},
            axis_labels={**self.axis_labels, **dict(overrides.get("axis_labels", {}))},
            thresholds={**self.thresholds, **dict(overrides.get("thresholds", {}))},
            semantic_flags={**self.semantic_flags, **dict(overrides.get("semantic_flags", {}))},
        )


_GENERIC_PROFILE = ExperimentPlotProfile()

_PROFILE_REGISTRY: dict[tuple[str, str], ExperimentPlotProfile] = {
    ("generic", "plot_trajectory"): ExperimentPlotProfile(
        chart_name="plot_trajectory",
        figure_title="Total Queue Length Trajectory",
        axis_labels={"x": "Time (t)", "y": r"$|Q(t)|_1$"},
        preferred_formats=("png",),
    ),
    ("generic", "plot_policy_comparison"): ExperimentPlotProfile(
        chart_name="plot_policy_comparison",
        preferred_formats=("png",),
    ),
    ("generic", "plot_convergence"): ExperimentPlotProfile(
        chart_name="plot_convergence",
        figure_title="Running Average of Total Queue Length",
        axis_labels={"x": "Time (t)", "y": r"$\frac{1}{t}\int_0^t |Q(s)|_1 ds$"},
        preferred_formats=("png",),
    ),
    ("verification", "plot_drift_landscape"): ExperimentPlotProfile(
        experiment_id="verification",
        chart_name="plot_drift_landscape",
        figure_title="Drift Landscape",
        axis_labels={"x": r"$Q_1$", "y": r"$Q_2$", "colorbar": r"Generator Drift ${\cal L}V(Q)$"},
        annotations_mode="theoretical_boundary",
        thresholds={"drift_center": 0.0},
        semantic_flags={"highlight_zero_contour": True},
    ),
    ("verification", "plot_drift_vs_norm"): ExperimentPlotProfile(
        experiment_id="verification",
        chart_name="plot_drift_vs_norm",
        figure_title="Drift Verification",
        axis_labels={"x": r"State Norm $|Q|_1$", "y": r"Generator Drift ${\cal L}V(Q)$"},
        annotations_mode="theoretical_boundary",
        thresholds={"drift_center": 0.0},
    ),
    ("stress", "plot_stress_dashboard"): ExperimentPlotProfile(
        experiment_id="stress",
        chart_name="plot_stress_dashboard",
        figure_title="Stress Test Dashboard",
        panel_titles={
            "scaling": "(a) Scaling Test",
            "critical": "(b) Critical Load",
            "heterogeneity": "(c) Heterogeneity",
        },
        axis_labels={
            "scaling_x": "Number of Servers (N)",
            "scaling_y": r"$\mathbb{E}[|Q|_1]$",
            "critical_x": r"Load Factor $\rho$",
            "critical_y": r"$\mathbb{E}[|Q|_1]$ (log)",
            "heterogeneity_y": r"$\mathbb{E}[|Q|_1]$",
        },
        legend_mode="figure",
        annotations_mode="stress_semantic",
        thresholds={"stationary_cutoff": 1.0},
        semantic_flags={
            "show_stationarity_legend": True,
            "annotate_heterogeneity_gini": True,
            "show_near_critical_band": True,
        },
    ),
    ("training", "plot_training_dashboard"): ExperimentPlotProfile(
        experiment_id="training",
        chart_name="plot_training_dashboard",
        figure_title="REINFORCE Training Dashboard",
        panel_titles={
            "performance": "(a) Base-Regime Diagnostic",
            "loss": "(b) Loss Curves",
            "critic": "(c) Critic Quality",
            "gradients": "(d) Gradient Health",
        },
        legend_mode="figure",
        annotations_mode="training_semantic",
        semantic_flags={"merge_twin_axis_legends": True},
    ),
    ("generalize", "plot_improvement_heatmap"): ExperimentPlotProfile(
        experiment_id="generalize",
        chart_name="plot_improvement_heatmap",
        figure_title="Generalization Sweep: Improvement Ratio",
        axis_labels={
            "x": r"Load Factor $\rho$",
            "y": "Service Rate Scale (x base distribution)",
            "colorbar": "Improvement Ratio (GibbsQ / Neural)",
        },
        annotations_mode="improvement_semantic",
        thresholds={"center": 1.0, "win_threshold": 1.0},
        semantic_flags={"semantic_cell_labels": True},
    ),
    ("critical", "plot_critical_load"): ExperimentPlotProfile(
        experiment_id="critical",
        chart_name="plot_critical_load",
        figure_title=r"Critical Load: E[Q] vs $\rho$ Near Stability Boundary",
        axis_labels={
            "x": r"Load Factor $\rho$",
            "y": r"$\mathbb{E}[|Q|_1]$ (log scale)",
        },
        legend_mode="axes",
        annotations_mode="critical_semantic",
        thresholds={"critical_rho": 0.95},
        semantic_flags={"show_near_critical_band": True, "strict_log_validation": True},
    ),
    ("sweep", "plot_alpha_sweep"): ExperimentPlotProfile(
        experiment_id="sweep",
        chart_name="plot_alpha_sweep",
        figure_title=r"System Performance vs Inverse Routing Temperature ($\alpha$)",
        axis_labels={
            "x": r"Inverse Temperature $\alpha$",
            "y": r"Expected Total Queue Length $\mathbb{E}[|Q|_1]$",
            "legend_title": "Load Factor",
        },
        annotations_mode="sweep_semantic",
    ),
    ("reinforce_check", "plot_gradient_scatter"): ExperimentPlotProfile(
        experiment_id="reinforce_check",
        chart_name="plot_gradient_scatter",
        figure_title="Gradient Estimator Agreement",
        axis_labels={
            "x": "Finite-Difference Gradient (ground truth)",
            "y": "REINFORCE Gradient (estimate)",
            "colorbar": "Z-score magnitude",
        },
        annotations_mode="gradient_check",
    ),
    ("stats", "plot_raincloud"): ExperimentPlotProfile(
        experiment_id="stats",
        chart_name="plot_raincloud",
        figure_title="Queue-Length Distribution Comparison",
        axis_labels={"y": r"$\mathbb{E}[|Q|_1]$"},
        annotations_mode="distribution_comparison",
    ),
    ("policy", "plot_tier_comparison_bars"): ExperimentPlotProfile(
        experiment_id="policy",
        chart_name="plot_tier_comparison_bars",
        figure_title="Corrected Policy Comparison",
        axis_labels={"y": "Expected Total Queue Length E[Q_total]"},
        legend_mode="axes",
        annotations_mode="tier_comparison",
    ),
    ("policy", "plot_platinum_grid"): ExperimentPlotProfile(
        experiment_id="policy",
        chart_name="plot_platinum_grid",
        figure_title="Proposed Grid Analysis",
        panel_titles={
            "envelope": "Performance Envelope",
            "efficiency": "Generalization Efficiency",
        },
        axis_labels={
            "envelope_x": r"Load Factor $\rho$",
            "envelope_y": r"$\mathbb{E}[|Q|_1]$ (Log Scale)",
            "efficiency_x": r"Load Factor $\rho$",
            "efficiency_y": "Performance Index (%)",
        },
        annotations_mode="policy_generalization",
    ),
    ("ablation", "plot_ablation_bars"): ExperimentPlotProfile(
        experiment_id="ablation",
        chart_name="plot_ablation_bars",
        figure_title="Ablation Study: Component Contributions",
        axis_labels={"y": r"$\mathbb{E}[Q_{total}]$"},
        annotations_mode="ablation_delta",
        semantic_flags={"reference_first_variant": True},
    ),
    ("ablation", "plot_ablation_training_curve"): ExperimentPlotProfile(
        experiment_id="ablation",
        chart_name="plot_ablation_training_curve",
        figure_title="Ablation Variant Training Curve",
        panel_titles={
            "loss": "(a) Training Objective",
            "performance": "(b) Performance Proxy",
        },
        axis_labels={
            "x": "Epoch",
            "loss_y": "Policy Loss",
            "performance_y": "Base-Regime PI Proxy (%)",
        },
        annotations_mode="ablation_training",
    ),
}


def resolve_experiment_plot_profile(
    experiment_id: str | None,
    chart_name: str,
    context: ExperimentPlotContext | None = None,
    profile: str | ExperimentPlotProfile | None = None,
) -> ExperimentPlotProfile:
    """Resolve an experiment-aware plotting profile.

    Resolution order:
    1. explicit profile instance / profile key
    2. context experiment id
    3. provided experiment id
    4. generic fallback
    """
    if isinstance(profile, ExperimentPlotProfile):
        base = profile
    else:
        selected_experiment = (
            (context.experiment_id if context and context.experiment_id else None)
            or (profile if isinstance(profile, str) else None)
            or experiment_id
            or "generic"
        )
        base = _PROFILE_REGISTRY.get((selected_experiment, chart_name), _GENERIC_PROFILE)

    if context is not None and context.chart_name is not None and context.chart_name != chart_name:
        raise ValueError(
            f"context.chart_name={context.chart_name!r} does not match chart_name={chart_name!r}"
        )

    if context is None:
        return base
    return base.merged(dict(context.semantic_overrides))
