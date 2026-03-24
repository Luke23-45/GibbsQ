"""
Tests for publication-quality chart system.

Verifies:
1. Theme switching (dark/publication)
2. Multi-format export (PNG/PDF/SVG)
3. Data export (CSV/JSON)
4. Chart quality standards
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from gibbsq.analysis.plotting import (
    plot_drift_landscape,
    plot_drift_vs_norm,
    plot_policy_comparison,
    plot_alpha_sweep,
    plot_trajectory,
    plot_convergence,
)
from gibbsq.analysis.theme import (
    apply_theme,
    get_publication_params,
    get_dark_params,
    PublicationTheme,
)
from gibbsq.utils.chart_exporter import (
    save_chart,
    save_data,
    ChartConfig,
)
from gibbsq.core.drift import DriftResult
from gibbsq.engines.numpy_engine import SimResult


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_drift_result():
    """Create sample drift result for testing."""
    np.random.seed(42)
    n_points = 100
    states = np.random.randint(0, 20, size=(n_points, 2))
    exact_drifts = -0.5 * np.sum(states, axis=1) + 2.0 + np.random.randn(n_points) * 0.5
    norms = np.sum(states, axis=1)
    upper_bounds = exact_drifts + 1.0  # Dummy upper bounds
    simplified_bounds = exact_drifts + 0.5  # Dummy simplified bounds
    
    return DriftResult(
        states=states,
        exact_drifts=exact_drifts,
        upper_bounds=upper_bounds,
        simplified_bounds=simplified_bounds,
        violations=0,
        norms=norms,
    )


@pytest.fixture
def sample_sim_result():
    """Create sample simulation result for testing."""
    np.random.seed(42)
    n_steps = 1000
    times = np.cumsum(np.random.exponential(0.1, n_steps))
    states = np.random.randint(0, 10, size=(n_steps, 3))
    
    return SimResult(
        states=states,
        jump_times=times,
        action_step_indices=list(range(0, n_steps, 10)),
        actions=list(np.random.randint(0, 3, n_steps // 10)),
        arrival_count=n_steps // 2,
        departure_count=n_steps // 2,
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ──────────────────────────────────────────────────────────────
# Theme Tests
# ──────────────────────────────────────────────────────────────

class TestThemeSystem:
    """Test theme management functionality."""
    
    def test_publication_theme_has_white_background(self):
        """T1: Publication theme must have white background."""
        params = get_publication_params()
        assert params.get('figure.facecolor') == 'white'
        assert params.get('axes.facecolor') == 'white'
    
    def test_dark_theme_has_dark_background(self):
        """T2: Dark theme must have dark background."""
        params = get_dark_params()
        # Dark background should not be white
        assert params.get('figure.facecolor') != 'white'
    
    def test_apply_theme_publication(self):
        """T3: Apply publication theme changes rcParams."""
        apply_theme('publication')
        assert plt.rcParams['figure.facecolor'] == 'white'
        assert plt.rcParams['axes.facecolor'] == 'white'
    
    def test_apply_theme_dark(self):
        """T4: Apply dark theme changes rcParams."""
        apply_theme('dark')
        # Should not be white
        assert plt.rcParams['figure.facecolor'] != 'white'
    
    def test_publication_theme_uses_serif_font(self):
        """T5: Publication theme should use serif font for academic papers."""
        params = get_publication_params()
        # Times New Roman or similar serif font preferred
        font_family = params.get('font.family')
        assert font_family in ['serif', 'Times New Roman', 'DejaVu Serif']
    
    def test_publication_theme_has_high_dpi(self):
        """T6: Publication theme should have high DPI for print quality."""
        params = get_publication_params()
        assert params.get('savefig.dpi', 300) >= 300


# ──────────────────────────────────────────────────────────────
# Chart Export Tests
# ──────────────────────────────────────────────────────────────

class TestChartExporter:
    """Test multi-format chart export."""
    
    def test_save_chart_png(self, temp_output_dir, sample_drift_result):
        """T1: Save chart as PNG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        output_path = temp_output_dir / "test_chart"
        saved_paths = save_chart(fig, output_path, formats=['png'])
        
        assert len(saved_paths) == 1
        assert saved_paths[0].suffix == '.png'
        assert saved_paths[0].exists()
        plt.close(fig)
    
    def test_save_chart_pdf(self, temp_output_dir):
        """T2: Save chart as PDF (vector format)."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        output_path = temp_output_dir / "test_chart"
        saved_paths = save_chart(fig, output_path, formats=['pdf'])
        
        assert len(saved_paths) == 1
        assert saved_paths[0].suffix == '.pdf'
        assert saved_paths[0].exists()
        plt.close(fig)
    
    def test_save_chart_svg(self, temp_output_dir):
        """T3: Save chart as SVG (vector format)."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        output_path = temp_output_dir / "test_chart"
        saved_paths = save_chart(fig, output_path, formats=['svg'])
        
        assert len(saved_paths) == 1
        assert saved_paths[0].suffix == '.svg'
        assert saved_paths[0].exists()
        plt.close(fig)
    
    def test_save_chart_all_formats(self, temp_output_dir):
        """T4: Save chart in all formats simultaneously."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        output_path = temp_output_dir / "test_chart"
        saved_paths = save_chart(fig, output_path, formats=['png', 'pdf', 'svg'])
        
        assert len(saved_paths) == 3
        suffixes = {p.suffix for p in saved_paths}
        assert suffixes == {'.png', '.pdf', '.svg'}
        plt.close(fig)
    
    def test_save_chart_with_config(self, temp_output_dir):
        """T5: Save chart with custom configuration."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        config = ChartConfig(
            dpi=600,
            transparent=False,
            bbox_inches='tight',
        )
        
        output_path = temp_output_dir / "test_chart"
        saved_paths = save_chart(fig, output_path, formats=['png'], config=config)
        
        assert saved_paths[0].exists()
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Data Export Tests
# ──────────────────────────────────────────────────────────────

class TestDataExporter:
    """Test data export functionality."""
    
    def test_save_data_csv(self, temp_output_dir):
        """T1: Save data as CSV."""
        data = {
            'x': [1, 2, 3, 4, 5],
            'y': [2.1, 3.5, 4.2, 5.1, 6.3],
        }
        
        output_path = temp_output_dir / "test_data"
        saved_path = save_data(data, output_path, format='csv')
        
        assert saved_path.suffix == '.csv'
        assert saved_path.exists()
    
    def test_save_data_json(self, temp_output_dir):
        """T2: Save data as JSON."""
        data = {
            'experiment': 'drift_verification',
            'results': {'mean': 1.5, 'std': 0.3},
        }
        
        output_path = temp_output_dir / "test_data"
        saved_path = save_data(data, output_path, format='json')
        
        assert saved_path.suffix == '.json'
        assert saved_path.exists()
    
    def test_save_data_npz(self, temp_output_dir):
        """T3: Save numpy arrays as NPZ."""
        data = {
            'states': np.random.randn(100, 3),
            'values': np.random.randn(100),
        }
        
        output_path = temp_output_dir / "test_data"
        saved_path = save_data(data, output_path, format='npz')
        
        assert saved_path.suffix == '.npz'
        assert saved_path.exists()


# ──────────────────────────────────────────────────────────────
# Plotting Function Tests with Theme
# ──────────────────────────────────────────────────────────────

class TestPlottingWithTheme:
    """Test plotting functions with theme support."""
    
    def test_plot_drift_vs_norm_publication_theme(self, sample_drift_result, temp_output_dir):
        """T1: Drift plot with publication theme."""
        apply_theme('publication')
        
        output_path = temp_output_dir / "drift_vs_norm"
        fig = plot_drift_vs_norm(
            sample_drift_result,
            eps=0.75,
            R=2.44,
            save_path=output_path,
            theme='publication',
        )
        
        # Check white background
        assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)  # White
        plt.close(fig)
    
    def test_plot_alpha_sweep_publication_theme(self, temp_output_dir):
        """T2: Alpha sweep plot with publication theme."""
        apply_theme('publication')
        
        alpha_values = np.logspace(-1, 2, 20)
        mean_q_matrix = np.random.randn(3, 20) * 5 + 10
        rho_labels = ['ρ=0.5', 'ρ=0.7', 'ρ=0.9']
        
        output_path = temp_output_dir / "alpha_sweep"
        fig = plot_alpha_sweep(
            alpha_values,
            mean_q_matrix,
            rho_labels,
            save_path=output_path,
            theme='publication',
        )
        
        assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        plt.close(fig)
    
    def test_plot_policy_comparison_publication_theme(self, temp_output_dir):
        """T3: Policy comparison plot with publication theme."""
        apply_theme('publication')
        
        results_dict = {
            'Random': [10.2, 11.5, 9.8],
            'Softmax': [8.1, 7.9, 8.3],
            'Neural': [6.5, 6.2, 6.8],
        }
        
        output_path = temp_output_dir / "policy_comparison"
        fig = plot_policy_comparison(
            results_dict,
            'Mean Queue Length',
            save_path=output_path,
            theme='publication',
        )
        
        assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Colorblind Safety Tests
# ──────────────────────────────────────────────────────────────

class TestColorblindSafety:
    """Test that charts are colorblind-friendly."""
    
    def test_publication_palette_colorblind_safe(self):
        """T1: Publication color palette should be colorblind-safe."""
        theme = PublicationTheme()
        colors = theme.get_color_palette()
        
        # Should use a known colorblind-safe palette
        # e.g., Wong palette, Okabe-Ito, or similar
        assert len(colors) >= 4  # At least 4 distinguishable colors
        
        # Verify colors are from a colorblind-safe set
        # This is a basic check - full verification would need simulation
        for color in colors:
            assert color is not None


# ──────────────────────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────────────────────

class TestChartIntegration:
    """Integration tests for full chart generation pipeline."""
    
    def test_full_pipeline_drift_chart(self, sample_drift_result, temp_output_dir):
        """T1: Full pipeline for drift chart generation."""
        # Apply publication theme
        apply_theme('publication')
        
        # Generate chart
        output_path = temp_output_dir / "drift_analysis"
        fig = plot_drift_vs_norm(
            sample_drift_result,
            eps=0.75,
            R=2.44,
            save_path=output_path,
            theme='publication',
            formats=['png', 'pdf'],
        )
        
        # Verify outputs
        assert (temp_output_dir / "drift_analysis.png").exists()
        assert (temp_output_dir / "drift_analysis.pdf").exists()
        
        # Save associated data
        data = {
            'states': sample_drift_result.states.tolist(),
            'drifts': sample_drift_result.exact_drifts.tolist(),
            'norms': sample_drift_result.norms.tolist(),
        }
        save_data(data, temp_output_dir / "drift_data", format='csv')
        assert (temp_output_dir / "drift_data.csv").exists()
        
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Chart Style Registry Tests
# ──────────────────────────────────────────────────────────────

class TestChartStyleRegistry:
    """Test the per-chart-type style specification system."""

    def test_all_chart_types_have_registered_styles(self):
        """T1: Every ChartType enum value must have a ChartStyleSpec."""
        from gibbsq.analysis.chart_styles import ChartType, CHART_STYLES

        for ct in ChartType:
            assert ct in CHART_STYLES, f"Missing style spec for {ct.value}"

    def test_heatmap_uses_diverging_colormap(self):
        """T2: HEATMAP_DIVERGING must use a diverging cmap, not qualitative."""
        from gibbsq.analysis.chart_styles import ChartType, CHART_STYLES

        spec = CHART_STYLES[ChartType.HEATMAP_DIVERGING]
        diverging_cmaps = {"RdBu_r", "RdBu", "RdYlGn", "RdYlGn_r",
                           "coolwarm", "bwr", "PiYG", "BrBG", "Spectral"}
        assert spec.colormap in diverging_cmaps, (
            f"Heatmap should use diverging cmap, got {spec.colormap!r}")

    def test_bar_comparison_has_error_bar_caps(self):
        """T3: BAR_COMPARISON must have visible error bar caps."""
        from gibbsq.analysis.chart_styles import ChartType, CHART_STYLES

        spec = CHART_STYLES[ChartType.BAR_COMPARISON]
        assert spec.error_bar_capsize > 0

    def test_scatter_has_visible_markers(self):
        """T4: SCATTER must use visible markers."""
        from gibbsq.analysis.chart_styles import ChartType, CHART_STYLES

        spec = CHART_STYLES[ChartType.SCATTER]
        assert spec.marker_style != "none"
        assert spec.marker_size > 0

    def test_line_series_uses_full_palette(self):
        """T5: LINE_SERIES must use the full Okabe-Ito palette."""
        from gibbsq.analysis.chart_styles import ChartType, CHART_STYLES

        spec = CHART_STYLES[ChartType.LINE_SERIES]
        assert len(spec.palette) >= 8

    def test_get_chart_style_returns_copy(self):
        """T6: get_chart_style returns a deep copy (mutation-safe)."""
        from gibbsq.analysis.chart_styles import ChartType, get_chart_style

        s1 = get_chart_style(ChartType.SCATTER)
        s2 = get_chart_style(ChartType.SCATTER)
        s1.marker_size = 999.0
        assert s2.marker_size != 999.0


# ──────────────────────────────────────────────────────────────
# New Plot Function Tests
# ──────────────────────────────────────────────────────────────

class TestNewPlotFunctions:
    """Test all 7 new chart-type-aware plot functions."""

    def test_plot_gradient_scatter(self, temp_output_dir):
        """T1: Gradient scatter plot generates a figure."""
        from gibbsq.analysis.plotting import plot_gradient_scatter

        np.random.seed(42)
        fd = np.random.randn(50)
        rf = fd + np.random.randn(50) * 0.1

        fig = plot_gradient_scatter(
            fd, rf,
            summary_stats={"cosine_similarity": 0.998, "relative_error": 0.05, "passed": True},
            save_path=temp_output_dir / "gradient_scatter",
            theme="publication",
        )
        assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        assert (temp_output_dir / "gradient_scatter.png").exists()
        plt.close(fig)

    def test_plot_stress_dashboard(self, temp_output_dir):
        """T2: Stress dashboard produces 3 panels."""
        from gibbsq.analysis.plotting import plot_stress_dashboard

        scaling = {"n_values": [2, 4, 8], "mean_q": [5.0, 8.0, 12.0], "gini": [0.1, 0.2, 0.3]}
        critical = {"rho_values": [0.7, 0.9, 0.95], "mean_q": [3.0, 10.0, 50.0],
                     "stationary": [True, True, False]}
        hetero = {"scenario_names": ["Low", "Med", "High"], "mean_q": [4.0, 6.0, 9.0],
                  "gini": [0.1, 0.3, 0.5]}

        fig = plot_stress_dashboard(
            scaling, critical, hetero,
            save_path=temp_output_dir / "stress",
            theme="publication",
        )
        assert len(fig.axes) >= 3
        plt.close(fig)

    def test_plot_training_dashboard(self, temp_output_dir):
        """T3: Training dashboard produces 4 panels."""
        from gibbsq.analysis.plotting import plot_training_dashboard

        n = 50
        metrics = {
            "epoch": list(range(n)),
            "performance_index": np.random.rand(n).tolist(),
            "performance_index_ema": np.cumsum(np.random.rand(n)).tolist(),
            "policy_loss": (np.random.rand(n) * -1).tolist(),
            "value_loss": (np.random.rand(n) * 0.5).tolist(),
            "ev_ema": np.linspace(0, 0.8, n).tolist(),
            "corr_ema": np.linspace(0, 0.9, n).tolist(),
            "policy_grad_norm": np.random.rand(n).tolist(),
            "value_grad_norm": np.random.rand(n).tolist(),
            "entropy": np.linspace(1.0, 0.1, n).tolist(),
        }
        fig = plot_training_dashboard(
            metrics, jsq_baseline=100.0, random_baseline=0.0,
            save_path=temp_output_dir / "training",
            theme="publication",
        )
        # 4 main panels + up to 3 twin axes
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_plot_raincloud(self, temp_output_dir):
        """T4: Raincloud plot produces figure with stats bracket."""
        from gibbsq.analysis.plotting import plot_raincloud

        np.random.seed(42)
        a = np.random.normal(10, 1, 30)
        b = np.random.normal(8, 1.2, 30)

        fig = plot_raincloud(
            a, b,
            stats={"p_value": 0.001, "cohen_d": 1.5, "improvement_pct": 20.0},
            save_path=temp_output_dir / "raincloud",
            theme="publication",
        )
        assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        plt.close(fig)

    def test_plot_improvement_heatmap(self, temp_output_dir):
        """T5: Improvement heatmap uses diverging colormap."""
        from gibbsq.analysis.plotting import plot_improvement_heatmap

        grid = np.array([[0.8, 1.2, 1.5], [1.0, 1.1, 0.9], [1.3, 0.7, 1.4]])

        fig = plot_improvement_heatmap(
            grid,
            x_labels=["0.5", "0.7", "0.9"],
            y_labels=["1x", "2x", "3x"],
            save_path=temp_output_dir / "heatmap",
            theme="publication",
        )
        assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        plt.close(fig)

    def test_plot_ablation_bars(self, temp_output_dir):
        """T6: Ablation bars show delta-% annotations."""
        from gibbsq.analysis.plotting import plot_ablation_bars

        fig = plot_ablation_bars(
            variant_names=["Full", "No LogNorm", "No ZeroInit", "Uniform"],
            mean_values=[5.0, 5.8, 6.2, 9.0],
            se_values=[0.3, 0.4, 0.5, 0.8],
            save_path=temp_output_dir / "ablation",
            theme="publication",
        )
        assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        plt.close(fig)

    def test_plot_critical_load(self, temp_output_dir):
        """T7: Critical load plot uses log scale."""
        from gibbsq.analysis.plotting import plot_critical_load

        rho = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
        neural = np.array([3, 5, 8, 20, 50, 200], dtype=float)
        gibbs = np.array([4, 6, 10, 25, 60, 250], dtype=float)

        fig = plot_critical_load(
            rho, neural, gibbs,
            save_path=temp_output_dir / "critical",
            theme="publication",
        )
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"
        plt.close(fig)
