import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gibbsq.analysis import plotting
from gibbsq.analysis.theme import PublicationTheme, get_dark_params, get_publication_params
from gibbsq.core.drift import DriftResult
from gibbsq.engines.numpy_engine import SimResult
from gibbsq.utils.chart_exporter import ChartConfig, save_chart, save_data


@pytest.fixture
def sample_drift_result():
    states = np.array([[0, 0], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
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


def test_plot_drift_vs_norm_uses_publication_facecolor(sample_drift_result, tmp_path):
    fig = plotting.plot_drift_vs_norm(
        sample_drift_result,
        eps=0.5,
        R=2.0,
        save_path=tmp_path / "drift_vs_norm",
        theme="publication",
    )
    assert (tmp_path / "drift_vs_norm.png").exists()
    assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)


def test_plot_policy_comparison_and_alpha_sweep_render(tmp_path):
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
    assert (tmp_path / "policy_compare.png").exists()
    assert (tmp_path / "alpha_sweep.png").exists()
    assert len(comparison.axes[0].patches) == 2
    assert sweep.axes[0].get_xscale() == "log"


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
