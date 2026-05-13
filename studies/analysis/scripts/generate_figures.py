"""
Generate all publication figures for the GibbsQ thesis.

Usage::

    python -m analysis.scripts.generate_figures [--data-dir PATH] [--figure-dir PATH]

Produces PDF and PNG versions of all figures in a modular directory
structure organised by hypothesis.

Output structure::

    {figure-dir}/
    ├── h1h2_reflected_ode_convergence.{pdf,png}
    ├── h2_boundary_equilibrium_verification.{pdf,png}
    ├── h3_boundary_mismatch_demo.{pdf,png}
    ├── h4_exhaustive_drift_audit.{pdf,png}
    └── h4_theorem_constant_sweep_heatmap.{pdf,png}
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from analysis.common.io import resolve_data_root

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Generate all figures into modular subdirectories."""
    data_dir = Path(args.data_dir) if args.data_dir else resolve_data_root()
    figure_dir = Path(args.figure_dir)

    logger.info("Data dir:   %s", data_dir)
    logger.info("Figure dir: %s", figure_dir)

    figure_dir.mkdir(parents=True, exist_ok=True)

    # Import study-specific figure generators
    from analysis.studies.convergence.figures import (
        fig_reflected_ode_convergence,
    )
    from analysis.studies.equilibrium.figures import (
        fig_boundary_equilibrium,
    )
    from analysis.studies.drift.figures import (
        fig_exhaustive_drift_audit,
        fig_theorem_constant_sweep,
    )
    from analysis.studies.mismatch.figures import (
        fig_boundary_mismatch,
    )

    all_figures = [
        ("H2: Boundary Equilibrium", fig_boundary_equilibrium),
        ("H1/H2: Reflected-ODE Convergence", fig_reflected_ode_convergence),
        ("H4: Exhaustive Drift Audit", fig_exhaustive_drift_audit),
        ("H4: Theorem-Constant Sweep", fig_theorem_constant_sweep),
        ("H3: Boundary Mismatch", fig_boundary_mismatch),
    ]

    from analysis.common.style import apply_thesis_style
    with apply_thesis_style():
        for label, func in all_figures:
            logger.info("Generating figure: %s", label)
            try:
                func(data_dir, figure_dir)
            except Exception as exc:
                logger.error("  Figure generation failed for %s: %s", label, exc)

    logger.info("All figures generated successfully.")


def cli() -> None:
    """Parse CLI arguments and run."""
    parser = argparse.ArgumentParser(
        description="Generate all publication figures for the GibbsQ thesis.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to outputs/data/ directory.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="outputs/figures",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    main(args)


if __name__ == "__main__":
    cli()
