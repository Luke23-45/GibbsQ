"""
Master script: generate ALL analysis outputs (tables + figures + stats).

Usage::

    python -m analysis.scripts.generate_all [--data-dir PATH] [--output-dir PATH]

This is the single entry point for reproducing all thesis artifacts, utilizing
a modular directory structure.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common.io import resolve_data_root

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Generate everything."""
    data_root = Path(args.data_dir) if args.data_dir else resolve_data_root()
    base_output = Path(args.output_dir)

    base_output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("GibbsQ Analysis Framework — Full Generation")
    logger.info("Data root:  %s", data_root)
    logger.info("Output dir: %s", base_output)
    logger.info("=" * 60)

    # Tables
    logger.info("\n── Phase 1: Tables ──")
    from analysis.scripts.generate_tables import generate_all_tables
    generate_all_tables(str(data_root), str(base_output / "tables"))

    # Figures
    logger.info("\n── Phase 2: Figures ──")
    from analysis.scripts.generate_figures import generate_all_figures
    generate_all_figures(str(data_root), str(base_output / "figures"))

    # Statistics
    logger.info("\n── Phase 3: Statistics ──")
    from analysis.scripts.generate_stats import generate_statistical_summary
    generate_statistical_summary(data_root, base_output / "reports")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("All outputs generated in %.1f seconds.", elapsed)
    logger.info("Root Output Dir: %s", base_output)
    logger.info("=" * 60)


def cli() -> None:
    """Parse CLI arguments and run."""
    parser = argparse.ArgumentParser(
        description="Generate ALL analysis outputs for the GibbsQ thesis.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to outputs/data/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base directory for all outputs.",
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
