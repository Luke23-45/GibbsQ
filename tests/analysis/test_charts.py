import logging
import inspect
import re
from types import SimpleNamespace
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gibbsq.analysis import plotting
from gibbsq.analysis.plot_profiles import ExperimentPlotContext, resolve_experiment_plot_profile
from gibbsq.analysis.theme import PublicationTheme, get_dark_params, get_publication_params
from gibbsq.core.drift import DriftResult
from gibbsq.engines.numpy_engine import SimResult
from experiments.evaluation.n_gibbsq_evals.ablation_ssa import AblationReinforceTrainer
from experiments.evaluation.n_gibbsq_evals.critical_load import CriticalLoadTest
from gibbsq.utils.chart_exporter import ChartConfig, save_chart, save_data
from gibbsq.utils.run_artifacts import figures_dir


@pytest.fixture
def sample_drift_result():
    states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    exact_drifts = np.array([1.0, 0.5, -1.0, -2.0], dtype=np.float64)
    return DriftResult(
        states=states,
        exact_drifts=exact_drifts,
        upper_bounds=exact_drifts + 0.5,
        simplified_bounds=exact_drifts + 1.0,
        violations=0,
        norms=states.sum(axis=1).astype(np.float64),
    )


@pytest.fixture
def sample_sim_result():
    return SimResult(
        times=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        states=np.array([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=np.int64),
        arrival_count=4,
        departure_count=1,
        final_time=3.0,
        num_servers=2,
    )


def test_publication_theme_has_white_background():
    params = get_publication_params()
    assert params["figure.facecolor"] == "white"
    assert params["axes.facecolor"] == "white"


def test_dark_theme_has_non_white_background():
    params = get_dark_params()
    assert params["figure.facecolor"] != "white"


def test_apply_theme_uses_installed_sans_serif_font(caplog):
    caplog.set_level(logging.WARNING, logger="matplotlib.font_manager")

    plotting._apply_theme()

    fig, ax = plt.subplots()
    ax.set_title("Font smoke test")
    fig.canvas.draw()
    plt.close(fig)

    assert "Generic family 'sans-serif' not found" not in caplog.text
    assert "DejaVu Sans" in plt.rcParams["font.sans-serif"]


def test_publication_theme_palette_is_copy():
    theme = PublicationTheme()
    colors = theme.get_color_palette()
    assert len(colors) >= 4
    colors[0] = "#000000"
    assert theme.get_color_palette()[0] != "#000000"


def test_save_chart_writes_requested_formats(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    paths = save_chart(
        fig,
        tmp_path / "chart",
        formats=["png", "pdf"],
        config=ChartConfig(dpi=300),
        close_fig=True,
    )

    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.exists() for path in paths)


def test_save_data_writes_json_and_npz(tmp_path):
    json_path = save_data({"metric": [1, 2, 3]}, tmp_path / "metrics", format="json")
    npz_path = save_data({"values": np.array([1.0, 2.0])}, tmp_path / "arrays", format="npz")
    assert json_path.exists()
    assert npz_path.exists()


def test_plot_trajectory_saves_png(sample_sim_result, tmp_path):
    fig = plotting.plot_trajectory(
        sample_sim_result,
        save_path=tmp_path / "trajectory",
        theme="publication",
    )
    assert (tmp_path / "trajectory.png").exists()
    assert fig.axes[0].get_title() == "Total Queue Length Trajectory"


def test_plot_drift_landscape_and_drift_vs_norm_render(sample_drift_result, tmp_path):
    landscape = plotting.plot_drift_landscape(
        sample_drift_result,
        alpha=1.0,
        save_path=tmp_path / "drift_landscape",
        theme="publication",
    )
    drift = plotting.plot_drift_vs_norm(
        sample_drift_result,
        eps=0.5,
        R=2.0,
        save_path=tmp_path / "drift_vs_norm",
        theme="publication",
    )
    assert (tmp_path / "drift_landscape.png").exists()
    assert (tmp_path / "drift_vs_norm.png").exists()
    assert landscape.axes[0].get_title().startswith("Drift Landscape")
    assert drift.get_facecolor()[:3] == (1.0, 1.0, 1.0)


def test_plot_policy_comparison_alpha_sweep_and_convergence_render(sample_sim_result, tmp_path):
    comparison = plotting.plot_policy_comparison(
        {"JSQ": [3.0, 4.0], "GibbsQ": [2.0, 2.5]},
        "Mean Queue",
        save_path=tmp_path / "policy_compare",
        theme="publication",
    )
    sweep = plotting.plot_alpha_sweep(
        np.array([0.1, 1.0, 10.0]),
        np.array([[5.0, 4.0, 3.0], [6.0, 5.0, 4.0]]),
        ["rho=0.7", "rho=0.9"],
        save_path=tmp_path / "alpha_sweep",
        theme="publication",
    )
    convergence = plotting.plot_convergence(
        sample_sim_result,
        save_path=tmp_path / "convergence",
        theme="publication",
    )
    assert (tmp_path / "policy_compare.png").exists()
    assert (tmp_path / "alpha_sweep.png").exists()
    assert (tmp_path / "convergence.png").exists()
    assert len(comparison.axes[0].patches) == 2
    assert sweep.axes[0].get_xscale() == "log"
    assert convergence.axes[0].get_title() == "Running Average of Total Queue Length"


def test_plot_training_dashboard_and_critical_load_render(tmp_path):
    metrics = {
        "epoch": [0, 1, 2],
        "performance_index": [10.0, 20.0, 30.0],
        "performance_index_ema": [10.0, 15.0, 22.0],
        "policy_loss": [-1.0, -0.8, -0.6],
        "value_loss": [0.5, 0.4, 0.3],
        "ev_ema": [0.1, 0.2, 0.3],
        "corr_ema": [0.2, 0.3, 0.4],
        "policy_grad_norm": [1.0, 0.8, 0.6],
        "value_grad_norm": [0.9, 0.7, 0.5],
        "entropy": [1.0, 0.9, 0.8],
    }
    training = plotting.plot_training_dashboard(
        metrics,
        jsq_baseline=100.0,
        random_baseline=0.0,
        save_path=tmp_path / "training",
        theme="publication",
    )
    critical = plotting.plot_critical_load(
        np.array([0.8, 0.9, 0.95], dtype=np.float64),
        np.array([5.0, 10.0, 20.0], dtype=np.float64),
        np.array([6.0, 12.0, 25.0], dtype=np.float64),
        save_path=tmp_path / "critical",
        theme="publication",
    )
    assert (tmp_path / "training.png").exists()
    assert (tmp_path / "critical.png").exists()
    assert len(training.axes) >= 4
    assert critical.axes[0].get_yscale() == "log"


def test_plot_ablation_training_curve_render_and_context(tmp_path):
    fig = plotting.plot_ablation_training_curve(
        {
            "epoch": [0, 1, 2],
            "training_loss": [0.7, 0.5, 0.4],
            "performance_index": [-120.0, -60.0, -10.0],
            "variant_label": "Full Model",
            "preprocessing": "log1p",
            "init_type": "zero_final",
            "train_epochs": 3,
        },
        save_path=tmp_path / "ablation_training",
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="ablation",
            chart_name="plot_ablation_training_curve",
        ),
    )
    assert (tmp_path / "ablation_training.png").exists()
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "(a) Training Objective"
    figure_text = [text.get_text() for text in fig.texts]
    assert any("Full Model" in text for text in figure_text)


def test_plot_training_dashboard_includes_secondary_axis_legend_entries():
    metrics = {
        "epoch": [0, 1, 2],
        "policy_loss": [-1.0, -0.8, -0.6],
        "value_loss": [0.5, 0.4, 0.3],
        "policy_grad_norm": [1.0, 0.8, 0.6],
        "entropy": [1.0, 0.9, 0.8],
    }
    fig = plotting.plot_training_dashboard(metrics, theme="publication")
    loss_legend = fig.axes[1].get_legend()
    grad_legend = fig.axes[3].get_legend()
    if loss_legend is None or grad_legend is None:
        assert fig.legends
        merged_labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert "Policy Loss" in merged_labels
        assert "Value Loss" in merged_labels
        assert "Entropy" in merged_labels
        return
    loss_labels = [text.get_text() for text in loss_legend.get_texts()]
    grad_labels = [text.get_text() for text in grad_legend.get_texts()]
    assert "Policy Loss" in loss_labels
    assert "Value Loss" in loss_labels
    assert "Entropy" in grad_labels


def test_plot_training_dashboard_handles_sparse_history_cleanly():
    metrics = {
        "epoch": [4],
        "base_regime_index": [-204.3],
        "base_regime_index_ema": [-67.4],
        "policy_loss": [0.0066],
        "value_loss": [9384.5],
        "ev_ema": [0.215],
        "corr_ema": [-0.0049],
        "policy_grad_norm": [0.096],
        "value_grad_norm": [54028.7],
        "entropy": [0.563],
    }
    fig = plotting.plot_training_dashboard(
        metrics,
        jsq_baseline=100.0,
        random_baseline=0.0,
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="training",
            chart_name="plot_training_dashboard",
        ),
    )
    assert not fig.legends
    assert fig.axes[0].get_xlim()[0] < 4.0 < fig.axes[0].get_xlim()[1]
    figure_text = [text.get_text() for text in fig.texts]
    assert any("Partial history" in text for text in figure_text)


def test_plot_training_dashboard_surfaces_final_eval_summary():
    metrics = {
        "epoch": [0, 1, 2, 3, 4],
        "base_regime_index": [-587.7, -1385.2, -1875.8, -582.4, -204.3],
        "base_regime_index_ema": [-587.7, -850.9, -1189.1, -988.9, -730.0],
        "policy_loss": [0.0343, 0.0392, 0.0447, 0.0278, 0.0066],
        "value_loss": [17635.3, 259089.2, 394622.5, 9052.8, 9384.6],
        "ev_ema": [0.6357, 0.5653, 0.5148, 0.5855, 0.6077],
        "corr_ema": [-0.0126, -0.0291, -0.0404, -0.0334, -0.0273],
        "policy_grad_norm": [0.0527, 0.0973, 0.0952, 0.1144, 0.0970],
        "value_grad_norm": [10186.9, 212826.7, 325833.1, 15128.0, 54028.7],
        "entropy": [0.5743, 0.5602, 0.5671, 0.5532, 0.5637],
        "final_eval_mean": 93.87,
        "final_eval_std": 17.32,
        "final_eval_count": 3,
        "train_epochs": 5,
        "run_label": "debug",
    }
    fig = plotting.plot_training_dashboard(
        metrics,
        jsq_baseline=100.0,
        random_baseline=0.0,
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="training",
            chart_name="plot_training_dashboard",
        ),
    )
    perf_labels = fig.axes[0].get_legend_handles_labels()[1]
    assert any("Final deterministic eval" in label for label in perf_labels)
    figure_text = [text.get_text() for text in fig.texts]
    assert any("debug run" in text for text in figure_text)
    assert any("final deterministic eval = 93.9%" in text for text in figure_text)


def test_plot_profile_resolution_prefers_explicit_context_and_validates_chart_name():
    profile = resolve_experiment_plot_profile("critical", "plot_critical_load")
    assert profile.experiment_id == "critical"
    assert profile.thresholds["critical_rho"] == pytest.approx(0.95)

    generic = resolve_experiment_plot_profile("unknown", "plot_critical_load")
    assert generic.experiment_id == "generic"

    with pytest.raises(ValueError, match="does not match"):
        resolve_experiment_plot_profile(
            "critical",
            "plot_critical_load",
            context=ExperimentPlotContext(
                experiment_id="critical",
                chart_name="plot_training_dashboard",
            ),
        )


def test_plot_gradient_scatter_render_and_constant_data_no_warnings(tmp_path):
    fig = plotting.plot_gradient_scatter(
        np.array([1.0, -0.5, 0.25]),
        np.array([0.9, -0.45, 0.3]),
        z_scores=np.array([0.5, 1.2, 0.3]),
        summary_stats={"cosine_similarity": 0.98, "relative_error": 0.07, "passed": True},
        save_path=tmp_path / "gradient_scatter",
        theme="publication",
    )
    assert (tmp_path / "gradient_scatter.png").exists()
    assert fig.axes[0].get_aspect() == 1.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig2 = plotting.plot_gradient_scatter(
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            theme="publication",
        )
    assert not caught
    assert fig2.axes[0].get_xlim()[0] < fig2.axes[0].get_xlim()[1]


def test_plot_stress_dashboard_raincloud_heatmap_ablation_tier_and_platinum_render(tmp_path):
    stress = plotting.plot_stress_dashboard(
        scaling_data={"n_values": [2, 4, 8], "mean_q": [1.0, 2.0, 4.0], "gini": [0.1, 0.2, 0.3]},
        critical_data={"rho_values": [0.8, 0.9, 0.95], "mean_q": [2.0, 5.0, 10.0], "stationary": [True, True, False]},
        hetero_data={"scenario_names": ["100x Gap"], "mean_q": [3.0], "gini": [0.42]},
        save_path=tmp_path / "stress_dashboard",
        theme="publication",
    )
    raincloud = plotting.plot_raincloud(
        np.array([1.0, 2.0, 2.5, 3.0]),
        np.array([1.5, 2.2, 2.7, 3.5]),
        stats={"p_value": 0.02, "cohen_d": 0.6, "improvement_pct": 12.5},
        save_path=tmp_path / "raincloud",
        theme="publication",
    )
    heatmap = plotting.plot_improvement_heatmap(
        np.array([[0.8, 1.0], [1.2, 1.4]]),
        ["0.8", "0.9"],
        ["1.0x", "1.2x"],
        save_path=tmp_path / "improvement_heatmap",
        theme="publication",
    )
    ablation = plotting.plot_ablation_bars(
        ["Full", "No LN", "Baseline"],
        [1.0, 1.2, 1.5],
        [0.05, 0.04, 0.06],
        save_path=tmp_path / "ablation_bars",
        theme="publication",
    )
    tier = plotting.plot_tier_comparison_bars(
        ["A", "B", "C"],
        [10.0, 8.0, 6.0],
        [0.5, 0.4, 0.3],
        [1, 4, 5],
        save_path=tmp_path / "tier_comparison",
        theme="publication",
    )
    platinum = plotting.plot_platinum_grid(
        np.array([0.7, 0.8, 0.9]),
        np.array([12.0, 15.0, 20.0]),
        np.array([8.0, 11.0, 18.0]),
        np.array([7.0, 10.0, 16.0]),
        np.array([80.0, 85.0, 90.0]),
        save_path=tmp_path / "platinum_grid",
        theme="publication",
    )

    assert (tmp_path / "stress_dashboard.png").exists()
    assert (tmp_path / "raincloud.png").exists()
    assert (tmp_path / "improvement_heatmap.png").exists()
    assert (tmp_path / "ablation_bars.png").exists()
    assert (tmp_path / "tier_comparison.png").exists()
    assert (tmp_path / "platinum_grid.png").exists()
    assert len(stress.axes) == 3
    assert raincloud.axes[0].get_title().endswith("Distribution Comparison")
    assert len(heatmap.axes) == 2
    assert len(ablation.axes[0].patches) == 3
    assert tier.axes[0].get_title() == "Corrected Policy Comparison"
    assert len(platinum.axes) == 2


def test_experiment_aware_stress_heatmap_and_critical_semantics():
    stress = plotting.plot_stress_dashboard(
        scaling_data={"n_values": [2, 4, 8], "mean_q": [1.0, 2.0, 4.0], "gini": [0.1, 0.2, 0.3]},
        critical_data={"rho_values": [0.8, 0.9, 0.98], "mean_q": [2.0, 5.0, 10.0], "stationary": [True, True, False]},
        hetero_data={"scenario_names": ["100x Gap"], "mean_q": [3.0], "gini": [0.42]},
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="stress",
            chart_name="plot_stress_dashboard",
            semantic_overrides={"thresholds": {"critical_rho": 0.9}},
        ),
    )
    legend = stress.legends[0]
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert "Critical-load trend" in legend_labels

    heatmap = plotting.plot_improvement_heatmap(
        np.array([[0.8, 1.0], [1.2, 1.4]]),
        ["0.8", "0.9"],
        ["1.0x", "1.2x"],
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="generalize",
            chart_name="plot_improvement_heatmap",
        ),
    )
    cell_text = heatmap.axes[0].texts[0].get_text()
    assert "GibbsQ wins" in cell_text
    high_ratio_text = heatmap.axes[0].texts[2].get_text()
    assert "Neural wins" in high_ratio_text

    critical = plotting.plot_critical_load(
        np.array([0.8, 0.9, 0.95], dtype=np.float64),
        np.array([5.0, 10.0, 20.0], dtype=np.float64),
        np.array([6.0, 12.0, 25.0], dtype=np.float64),
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="critical",
            chart_name="plot_critical_load",
            semantic_overrides={"thresholds": {"critical_rho": 0.9}},
        ),
    )
    legend = critical.axes[0].get_legend()
    critical_labels = [text.get_text() for text in legend.get_texts()]
    assert any("Near-critical" in label for label in critical_labels)


def test_experiment_aware_remaining_experiment_profiles():
    sweep = plotting.plot_alpha_sweep(
        np.array([0.1, 1.0, 10.0]),
        np.array([[5.0, 4.0, 3.0], [6.0, 5.0, 4.0]]),
        ["rho=0.7", "rho=0.9"],
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="sweep",
            chart_name="plot_alpha_sweep",
        ),
    )
    assert "Inverse Routing Temperature" in sweep.axes[0].get_title()

    gradient = plotting.plot_gradient_scatter(
        np.array([1.0, -0.5, 0.25]),
        np.array([0.9, -0.45, 0.3]),
        z_scores=np.array([0.5, 1.2, 0.3]),
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="reinforce_check",
            chart_name="plot_gradient_scatter",
        ),
    )
    assert gradient.axes[0].get_title() == "Gradient Estimator Agreement"

    stats = plotting.plot_raincloud(
        np.array([1.0, 2.0, 2.5, 3.0]),
        np.array([1.5, 2.2, 2.7, 3.5]),
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="stats",
            chart_name="plot_raincloud",
            semantic_overrides={"figure_title": "GibbsQ vs N-GibbsQ: Distribution Comparison"},
        ),
    )
    assert stats.axes[0].get_title() == "GibbsQ vs N-GibbsQ: Distribution Comparison"

    tier = plotting.plot_tier_comparison_bars(
        ["A", "B", "C"],
        [10.0, 8.0, 6.0],
        [0.5, 0.4, 0.3],
        [1, 4, 5],
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="policy",
            chart_name="plot_tier_comparison_bars",
        ),
    )
    assert tier.axes[0].get_title() == "Corrected Policy Comparison"

    platinum = plotting.plot_platinum_grid(
        np.array([0.7, 0.8, 0.9]),
        np.array([12.0, 15.0, 20.0]),
        np.array([8.0, 11.0, 18.0]),
        np.array([7.0, 10.0, 16.0]),
        np.array([80.0, 85.0, 90.0]),
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="policy",
            chart_name="plot_platinum_grid",
        ),
    )
    axis_titles = {ax.get_title() for ax in platinum.axes}
    assert "Performance Envelope" in axis_titles
    assert "Generalization Efficiency" in axis_titles

    ablation_curve = plotting.plot_ablation_training_curve(
        {
            "epoch": [0, 1, 2],
            "training_loss": [0.7, 0.5, 0.4],
            "performance_index": [-120.0, -60.0, -10.0],
        },
        theme="publication",
        context=ExperimentPlotContext(
            experiment_id="ablation",
            chart_name="plot_ablation_training_curve",
        ),
    )
    assert ablation_curve._suptitle.get_text() == "Ablation Variant Training Curve"


def test_plot_improvement_heatmap_handles_break_even_grid():
    fig = plotting.plot_improvement_heatmap(
        np.ones((2, 2), dtype=np.float64),
        ["0.8", "0.9"],
        ["1.0x", "1.2x"],
        theme="publication",
    )
    assert len(fig.axes) == 2
    assert fig.axes[0].images[0].norm.vmin < fig.axes[0].images[0].norm.vmax


def test_plot_functions_raise_for_invalid_lengths_and_shapes(sample_drift_result):
    with pytest.raises(ValueError, match="mean_q_matrix must have shape"):
        plotting.plot_alpha_sweep(
            np.array([0.1, 1.0]),
            np.array([[1.0, 2.0, 3.0]]),
            ["rho=0.7"],
            theme="publication",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        plotting.plot_critical_load(
            np.array([0.8, 0.9]),
            np.array([0.0, 1.0]),
            np.array([1.0, 2.0]),
            theme="publication",
        )

    with pytest.raises(ValueError, match="matching lengths"):
        plotting.plot_tier_comparison_bars(
            ["A", "B"],
            [1.0],
            [0.1, 0.2],
            [1, 2],
            theme="publication",
        )

    with pytest.raises(ValueError, match="matching lengths"):
        plotting.plot_ablation_bars(
            ["Full", "Baseline"],
            [1.0],
            [0.1, 0.2],
            theme="publication",
        )

    with pytest.raises(ValueError, match="Input lengths must match"):
        plotting.plot_ablation_training_curve(
            {
                "epoch": [0, 1],
                "training_loss": [0.7],
                "performance_index": [-120.0, -60.0],
            },
            theme="publication",
        )

    with pytest.raises(ValueError, match="Input lengths must match"):
        plotting.plot_platinum_grid(
            np.array([0.7, 0.8]),
            np.array([10.0, 11.0]),
            np.array([8.0, 9.0]),
            np.array([7.0]),
            np.array([90.0, 91.0]),
            theme="publication",
        )

    sparse_drift = DriftResult(
        states=np.array([[0, 0], [1, 1]], dtype=np.int64),
        exact_drifts=np.array([1.0, -1.0], dtype=np.float64),
        upper_bounds=np.array([1.5, -0.5], dtype=np.float64),
        simplified_bounds=np.array([2.0, 0.0], dtype=np.float64),
        violations=0,
        norms=np.array([0.0, 2.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="full dense 2D grid"):
        plotting.plot_drift_landscape(sparse_drift, alpha=1.0, theme="publication")


def test_plotting_smoke_script_equivalent(tmp_path, sample_drift_result, sample_sim_result):
    charts = [
        plotting.plot_trajectory(sample_sim_result, save_path=tmp_path / "trajectory_smoke", theme="publication"),
        plotting.plot_drift_landscape(sample_drift_result, alpha=1.0, save_path=tmp_path / "drift_landscape_smoke", theme="publication"),
        plotting.plot_drift_vs_norm(sample_drift_result, eps=0.5, R=2.0, save_path=tmp_path / "drift_vs_norm_smoke", theme="publication"),
        plotting.plot_policy_comparison({"JSQ": [3.0, 4.0], "GibbsQ": [2.0, 2.5]}, "Mean Queue", save_path=tmp_path / "policy_compare_smoke", theme="publication"),
        plotting.plot_alpha_sweep(np.array([0.1, 1.0, 10.0]), np.array([[5.0, 4.0, 3.0], [6.0, 5.0, 4.0]]), ["rho=0.7", "rho=0.9"], save_path=tmp_path / "alpha_sweep_smoke", theme="publication"),
        plotting.plot_convergence(sample_sim_result, save_path=tmp_path / "convergence_smoke", theme="publication"),
        plotting.plot_gradient_scatter(np.array([1.0, -0.5, 0.25]), np.array([0.9, -0.45, 0.3]), save_path=tmp_path / "gradient_smoke", theme="publication"),
        plotting.plot_stress_dashboard(
            {"n_values": [2, 4, 8], "mean_q": [1.0, 2.0, 4.0], "gini": [0.1, 0.2, 0.3]},
            {"rho_values": [0.8, 0.9, 0.95], "mean_q": [2.0, 5.0, 10.0], "stationary": [True, True, False]},
            {"scenario_names": ["100x Gap"], "mean_q": [3.0], "gini": [0.42]},
            save_path=tmp_path / "stress_smoke",
            theme="publication",
        ),
        plotting.plot_training_dashboard(
            {
                "epoch": [0, 1, 2],
                "performance_index": [10.0, 20.0, 30.0],
                "performance_index_ema": [10.0, 15.0, 22.0],
                "policy_loss": [-1.0, -0.8, -0.6],
                "value_loss": [0.5, 0.4, 0.3],
                "ev_ema": [0.1, 0.2, 0.3],
                "corr_ema": [0.2, 0.3, 0.4],
                "policy_grad_norm": [1.0, 0.8, 0.6],
                "value_grad_norm": [0.9, 0.7, 0.5],
                "entropy": [1.0, 0.9, 0.8],
            },
            save_path=tmp_path / "training_smoke",
            theme="publication",
        ),
        plotting.plot_raincloud(np.array([1.0, 2.0, 2.5, 3.0]), np.array([1.5, 2.2, 2.7, 3.5]), save_path=tmp_path / "raincloud_smoke", theme="publication"),
        plotting.plot_improvement_heatmap(np.array([[1.0, 1.0], [1.0, 1.0]]), ["0.8", "0.9"], ["1.0x", "1.2x"], save_path=tmp_path / "heatmap_smoke", theme="publication"),
        plotting.plot_ablation_bars(["Full", "No LN", "Baseline"], [1.0, 1.2, 1.5], [0.05, 0.04, 0.06], save_path=tmp_path / "ablation_smoke", theme="publication"),
        plotting.plot_critical_load(np.array([0.8, 0.9, 0.95]), np.array([5.0, 10.0, 20.0]), np.array([6.0, 12.0, 25.0]), save_path=tmp_path / "critical_smoke", theme="publication"),
        plotting.plot_tier_comparison_bars(["A", "B", "C"], [10.0, 8.0, 6.0], [0.5, 0.4, 0.3], [1, 4, 5], save_path=tmp_path / "tier_smoke", theme="publication"),
        plotting.plot_platinum_grid(np.array([0.7, 0.8, 0.9]), np.array([12.0, 15.0, 20.0]), np.array([8.0, 11.0, 18.0]), np.array([7.0, 10.0, 16.0]), np.array([80.0, 85.0, 90.0]), save_path=tmp_path / "platinum_smoke", theme="publication"),
    ]

    assert all(chart is not None for chart in charts)
    assert len(list(tmp_path.glob("*.png"))) == len(charts)


def test_all_public_plot_helpers_accept_profile_and_context():
    public_plot_helpers = [name for name in plotting.__all__ if name.startswith("plot_")]
    missing = []
    for name in public_plot_helpers:
        signature = inspect.signature(getattr(plotting, name))
        if "profile" not in signature.parameters or "context" not in signature.parameters:
            missing.append(name)
    assert not missing, f"Missing profile/context parameters: {missing}"


def test_experiment_context_callers_only_target_contract_compliant_helpers():
    plotting_signature_ok = {
        name
        for name in plotting.__all__
        if name.startswith("plot_")
        and "profile" in inspect.signature(getattr(plotting, name)).parameters
        and "context" in inspect.signature(getattr(plotting, name)).parameters
    }
    experiment_paths = [
        Path("experiments/verification/drift_verification.py"),
        Path("experiments/sweeps/stability_sweep.py"),
        Path("experiments/testing/stress_test.py"),
        Path("experiments/training/train_reinforce.py"),
        Path("experiments/testing/reinforce_gradient_check.py"),
        Path("experiments/evaluation/baselines_comparison.py"),
        Path("experiments/evaluation/n_gibbsq_evals/critical_load.py"),
        Path("experiments/evaluation/n_gibbsq_evals/stats_bench.py"),
        Path("experiments/evaluation/n_gibbsq_evals/gen_sweep.py"),
        Path("experiments/evaluation/n_gibbsq_evals/ablation_ssa.py"),
    ]

    unresolved = []
    for rel_path in experiment_paths:
        text = rel_path.read_text(encoding="utf-8")
        for match in re.finditer(r"(plot_[a-z_]+)\s*\(", text):
            helper_name = match.group(1)
            snippet = text[match.start(): match.start() + 500]
            if "context=ExperimentPlotContext(" in snippet and helper_name not in plotting_signature_ok:
                unresolved.append((str(rel_path), helper_name))

    assert not unresolved, f"Experiment callers target non-compliant plot helpers: {unresolved}"


def test_critical_load_progress_plot_persists_partial_results(tmp_path):
    test = object.__new__(CriticalLoadTest)
    test.run_dir = tmp_path
    test.run_logger = None
    test.cfg = SimpleNamespace(
        generalization=SimpleNamespace(rho_boundary_threshold=0.95),
    )
    figures_dir(tmp_path).mkdir(parents=True, exist_ok=True)

    test._plot_progress([0.95], [18.4], [20.7])

    assert (figures_dir(tmp_path) / "critical_load_curve.png").exists()
    assert (figures_dir(tmp_path) / "critical_load_curve.pdf").exists()

    # Empty partial state should remain a no-op.
    before = sorted(path.name for path in tmp_path.iterdir())
    test._plot_progress([], [], [])
    after = sorted(path.name for path in tmp_path.iterdir())
    assert after == before


def test_critical_load_progress_raises_when_curve_files_are_missing(tmp_path, monkeypatch):
    test = object.__new__(CriticalLoadTest)
    test.run_dir = tmp_path
    test.run_logger = None
    test.cfg = SimpleNamespace(
        generalization=SimpleNamespace(rho_boundary_threshold=0.95),
    )
    figures_dir(tmp_path).mkdir(parents=True, exist_ok=True)

    def _fake_plot(*args, **kwargs):
        fig, _ = plt.subplots()
        return fig

    monkeypatch.setattr("gibbsq.analysis.plotting.plot_critical_load", _fake_plot)

    with pytest.raises(RuntimeError, match="did not produce the required figure artifact"):
        test._plot_progress([0.95], [18.4], [20.7])


def test_ablation_save_assets_uses_dedicated_training_plot(tmp_path, monkeypatch):
    trainer = object.__new__(AblationReinforceTrainer)
    trainer.run_dir = tmp_path
    trainer.cfg = SimpleNamespace(
        neural=SimpleNamespace(preprocessing="log1p", init_type="zero_final"),
    )

    saved_paths = []
    plot_calls = []

    def _fake_serialize(path, _obj):
        saved_paths.append(path)

    def _fake_plot(**kwargs):
        plot_calls.append(kwargs)
        fig, _ = plt.subplots()
        return fig

    monkeypatch.setattr("equinox.tree_serialise_leaves", _fake_serialize)
    monkeypatch.setattr("gibbsq.analysis.plotting.plot_ablation_training_curve", _fake_plot)

    trainer._save_assets(
        object(),
        object(),
        [0.7, 0.5, 0.4],
        [-120.0, -60.0, -10.0],
    )

    assert any(path.name == "n_gibbsq_reinforce_weights.eqx" for path in saved_paths)
    assert any(path.name == "value_network_weights.eqx" for path in saved_paths)
    assert plot_calls
    assert plot_calls[0]["save_path"].name == "ablation_training_curve"
    assert plot_calls[0]["context"].chart_name == "plot_ablation_training_curve"
    assert plot_calls[0]["metrics"]["training_loss"] == [0.7, 0.5, 0.4]
