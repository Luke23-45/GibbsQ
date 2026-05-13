"""
Generate all LaTeX tables for the GibbsQ thesis.

Usage::

    python -m analysis.scripts.generate_tables [--data-dir PATH] [--table-dir PATH]

Produces .tex and .md files in a modular directory structure.

Output structure::

    {table-dir}/
    ├── h2_boundary_equilibrium.{tex,md}
    ├── h1h2_reflected_ode_convergence.{tex,md}
    ├── h4_exhaustive_drift_audit.{tex,md}
    └── h3_boundary_mismatch.{tex,md}
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from analysis.common.io import resolve_data_root

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Generate all tables into modular subdirectories."""
    data_dir = Path(args.data_dir) if args.data_dir else resolve_data_root()
    table_dir = Path(args.table_dir)

    logger.info("Data dir:  %s", data_dir)
    logger.info("Table dir: %s", table_dir)

    table_dir.mkdir(parents=True, exist_ok=True)

    # Import study-specific table generators
    from analysis.studies.convergence.table import table_reflected_ode
    from analysis.studies.equilibrium.table import table_boundary_equilibrium
    from analysis.studies.drift.table import table_drift_audit
    from analysis.studies.mismatch.table import table_boundary_mismatch

    all_tables = [
        ("H2: Boundary Equilibrium", table_boundary_equilibrium),
        ("H1/H2: Reflected-ODE Convergence", table_reflected_ode),
        ("H4: Exhaustive Drift Audit", table_drift_audit),
        ("H3: Boundary Mismatch", table_boundary_mismatch),
    ]

    for label, func in all_tables:
        logger.info("Generating table: %s", label)
        try:
            func(data_dir, table_dir)
        except Exception as exc:
            logger.error("  Table generation failed for %s: %s", label, exc)

    logger.info("All tables generated successfully.")


def cli() -> None:
    """Parse CLI arguments and run."""
    parser = argparse.ArgumentParser(
        description="Generate all LaTeX tables for the GibbsQ thesis.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to outputs/data/ directory.",
    )
    parser.add_argument(
        "--table-dir",
        type=str,
        default="outputs/tables",
        help="Output directory for tables.",
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
