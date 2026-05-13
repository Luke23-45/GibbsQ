#!/usr/bin/env python3
"""
Generate thesis-quality figures from z2 experiment CSV data.

This module reads the CSV files produced by the z2 experiments and
generates publication-ready PDF and PNG figures for the thesis.

Usage:
    python -m studies.analysis.generate_figures
    python -m studies.analysis.generate_figures --data-dir outputs/data
    python -m studies.analysis.generate_figures --figure-dir outputs/figures

Each figure function reads CSV data, performs no computation beyond
layout, and writes both PDF and PNG outputs.

Design:
    - Figures are fully deterministic given the CSV inputs.
    - All figure functions use a shared style configuration for
      visual consistency.
    - Each figure function can be called independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "outputs/data"
DEFAULT_FIGURE_DIR = "outputs/figures"

# Lazy import matplotlib to allow --help without display
_plt = None
_mpl = None


def _ensure_matplotlib():
    """Lazy-load matplotlib with Agg backend."""
    global _plt, _mpl
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
        _mpl = matplotlib


def _apply_thesis_style():
    """Apply a consistent thesis-quality plot style."""
    _ensure_matplotlib()
    _plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


# ──────────────────────────────────────────────────────────────────────
# CSV discovery utilities
# ──────────────────────────────────────────────────────────────────────

def _find_latest_csv(data_dir: Path, prefix: str) -> Path | None:
    """Find the most recent CSV file matching a given prefix.

    Parameters
    ----------
    data_dir : Path
        Directory to search.
    prefix : str
        Filename prefix (e.g., 'exhaustive_drift_summary').

    Returns
    -------
    Path or None
        Path to the most recent matching CSV, or None if not found.
    """
    candidates = sorted(
        data_dir.glob(f"{prefix}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and return rows as dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _save_figure(fig, figure_dir: Path, name: str) -> list[Path]:
    """Save a figure as both PDF and PNG.

    Returns list of output paths.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = figure_dir / f"{name}.{ext}"
        fig.savefig(path, format=ext)
        paths.append(path)
        log.info("  Saved: %s", path)
    _plt.close(fig)
    return paths


# ──────────────────────────────────────────────────────────────────────
# Figure 1: Boundary Equilibrium Verification (H2)
# ──────────────────────────────────────────────────────────────────────

def fig_boundary_equilibrium(data_dir: Path, figure_dir: Path) -> list[Path]:
    """Generate the boundary-equilibrium verification figure.

    Produces a bar chart showing max_discrepancy per system, with a
    horizontal tolerance line.  Systems that PASS are green; FAIL are red.
    """
    _ensure_matplotlib()
    csv_path = _find_latest_csv(data_dir, "boundary_equilibrium_verification")
    if csv_path is None:
        log.warning("No boundary_equilibrium_verification CSV found — skipping")
        return []

    rows = _read_csv(csv_path)
    systems = [r["system_id"] for r in rows]
    discrepancies = [float(r["max_discrepancy"]) for r in rows]
    statuses = [r["status"] for r in rows]
    colors = ["#2ca02c" if s == "PASS" else "#d62728" for s in statuses]

    fig, ax = _plt.subplots(figsize=(10, 5))
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
    return _save_figure(fig, figure_dir, "h2_boundary_equilibrium_verification")


# ──────────────────────────────────────────────────────────────────────
# Figure 2: Reflected-ODE Convergence Summary (H1/H2)
# ──────────────────────────────────────────────────────────────────────

def fig_reflected_ode_convergence(data_dir: Path, figure_dir: Path) -> list[Path]:
    """Generate the reflected-ODE convergence summary figure.

    Produces a table-like figure showing convergence metrics per system.
    """
    _ensure_matplotlib()
    csv_path = _find_latest_csv(data_dir, "reflected_ode_convergence_summary")
    if csv_path is None:
        log.warning("No reflected_ode_convergence_summary CSV found — skipping")
        return []

    rows = _read_csv(csv_path)
    systems = [r["system_id"] for r in rows]
    diameters = [float(r["terminal_diameter"]) for r in rows]
    residuals = [float(r["max_residual_norm"]) for r in rows]
    conv_rates = [float(r["convergence_rate"]) for r in rows]
    statuses = [r["status"] for r in rows]

    fig, axes = _plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Terminal diameter
    colors = ["#2ca02c" if s == "PASS" else "#d62728" for s in statuses]
    x = np.arange(len(systems))
    axes[0].bar(x, diameters, color=colors, edgecolor="#333", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(systems, rotation=25, ha="right")
    axes[0].set_ylabel("Terminal pairwise diameter")
    axes[0].set_title("Attractor Uniqueness")
    axes[0].set_yscale("symlog", linthresh=1e-16)

    # Panel 2: Max residual norm
    axes[1].bar(x, residuals, color=colors, edgecolor="#333", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(systems, rotation=25, ha="right")
    axes[1].set_ylabel(r"$\max \|\dot{q}(T)\|_\infty$")
    axes[1].set_title("Residual at Terminal Time")
    axes[1].set_yscale("symlog", linthresh=1e-16)

    # Panel 3: Convergence rate
    axes[2].bar(x, conv_rates, color=colors, edgecolor="#333", alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(systems, rotation=25, ha="right")
    axes[2].set_ylabel("Convergence rate")
    axes[2].set_title("Fraction Converged")

    fig.suptitle("Reflected-ODE Multi-Start Convergence (H1/H2)", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_figure(fig, figure_dir, "h1h2_reflected_ode_convergence")


# ──────────────────────────────────────────────────────────────────────
# Figure 3: Exhaustive Drift Audit Summary (H4)
# ──────────────────────────────────────────────────────────────────────

def fig_exhaustive_drift_audit(data_dir: Path, figure_dir: Path) -> list[Path]:
    """Generate the exhaustive drift audit summary figure.

    Produces a bar chart showing the maximum residual per system,
    annotated with violation count and ε value.
    """
    _ensure_matplotlib()
    csv_path = _find_latest_csv(data_dir, "exhaustive_drift_summary")
    if csv_path is None:
        log.warning("No exhaustive_drift_summary CSV found — skipping")
        return []

    rows = _read_csv(csv_path)
    systems = [r["system_id"] for r in rows]
    max_residuals = [float(r["max_residual"]) for r in rows]
    violations = [int(r["violations"]) for r in rows]
    epsilons = [float(r["epsilon"]) for r in rows]
    statuses = [r["status"] for r in rows]
    colors = ["#2ca02c" if s == "PASS" else "#d62728" for s in statuses]

    fig, ax = _plt.subplots(figsize=(10, 5))
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
    return _save_figure(fig, figure_dir, "h4_exhaustive_drift_audit")


# ──────────────────────────────────────────────────────────────────────
# Figure 4: Theorem-Constant Sweep Heatmap (H4)
# ──────────────────────────────────────────────────────────────────────

def fig_theorem_constant_sweep(data_dir: Path, figure_dir: Path) -> list[Path]:
    """Generate the theorem-constant sweep heatmap.

    Shows ε values across the (beta, gamma) parameter grid, with
    certified points in green and non-certified in red.
    """
    _ensure_matplotlib()
    csv_path = _find_latest_csv(data_dir, "theorem_constant_sweep")
    if csv_path is None:
        log.warning("No theorem_constant_sweep CSV found — skipping")
        return []

    rows = _read_csv(csv_path)

    # Extract unique β and γ values
    betas = sorted(set(float(r["beta"]) for r in rows))
    gammas = sorted(set(float(r["gamma"]) for r in rows))

    if len(betas) < 2 or len(gammas) < 2:
        log.warning("Insufficient sweep range for heatmap — skipping")
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

    fig, ax = _plt.subplots(figsize=(10, 7))
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
    return _save_figure(fig, figure_dir, "h4_theorem_constant_sweep_heatmap")


# ──────────────────────────────────────────────────────────────────────
# Figure 5: Boundary Mismatch Demo (H3)
# ──────────────────────────────────────────────────────────────────────

def fig_boundary_mismatch(data_dir: Path, figure_dir: Path) -> list[Path]:
    """Generate the boundary-mismatch demonstration figure.

    Shows the decomposition of (L H)(Q) into interior, boundary,
    and remainder terms for a representative system.
    """
    _ensure_matplotlib()
    csv_path = _find_latest_csv(data_dir, "ctmc_boundary_mismatch_summary")
    if csv_path is None:
        log.warning("No ctmc_boundary_mismatch_summary CSV found — skipping")
        return []

    rows = _read_csv(csv_path)
    systems = [r["system_id"] for r in rows]
    positive_counts = [int(r["positive_boundary_count"]) for r in rows]
    max_gaps = [float(r["max_gap"]) for r in rows]
    boundary_counts = [int(r["boundary_states"]) for r in rows]

    fig, axes = _plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(systems))
    colors_gap = ["#d62728" if g > 0.01 else "#2ca02c" for g in max_gaps]

    # Panel 1: Positive boundary term count
    axes[0].bar(x, positive_counts, color="#e07020", edgecolor="#333", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(systems, rotation=25, ha="right")
    axes[0].set_ylabel("States with positive boundary term")
    axes[0].set_title("Boundary Obstruction Count")

    # Panel 2: Max gap
    axes[1].bar(x, max_gaps, color=colors_gap, edgecolor="#333", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(systems, rotation=25, ha="right")
    axes[1].set_ylabel("max(CTMC drift − reflected-ODE drift)")
    axes[1].set_title("Generator–ODE Drift Gap")

    fig.suptitle(
        "CTMC Generator Boundary Mismatch (H3)\n"
        "Demonstrates why the old H-based proof route fails",
        fontsize=13, y=1.04,
    )
    fig.tight_layout()
    return _save_figure(fig, figure_dir, "h3_boundary_mismatch_demo")


# ──────────────────────────────────────────────────────────────────────
# Master figure generation
# ──────────────────────────────────────────────────────────────────────

ALL_FIGURES = [
    ("H2: Boundary Equilibrium", fig_boundary_equilibrium),
    ("H1/H2: Reflected-ODE Convergence", fig_reflected_ode_convergence),
    ("H4: Exhaustive Drift Audit", fig_exhaustive_drift_audit),
    ("H4: Theorem-Constant Sweep", fig_theorem_constant_sweep),
    ("H3: Boundary Mismatch", fig_boundary_mismatch),
]


def generate_all_figures(
    data_dir: str | Path,
    figure_dir: str | Path,
) -> list[Path]:
    """Generate all thesis figures.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing experiment CSV files.
    figure_dir : str or Path
        Output directory for generated figures.

    Returns
    -------
    list of Path
        All generated figure paths.
    """
    _apply_thesis_style()
    data_dir = Path(data_dir)
    figure_dir = Path(figure_dir)

    all_paths: list[Path] = []
    for label, func in ALL_FIGURES:
        log.info("Generating figure: %s", label)
        try:
            paths = func(data_dir, figure_dir)
            all_paths.extend(paths)
        except Exception as exc:
            log.error("  Figure generation failed for %s: %s", label, exc)

    log.info("Generated %d figure files total", len(all_paths))
    return all_paths


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][figures] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis-quality figures from z2 experiment CSV data. "
            "Reads from outputs/data/ and writes PDF+PNG to outputs/figures/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    paths = generate_all_figures(args.data_dir, args.figure_dir)
    log.info("Done: %d figures generated", len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
