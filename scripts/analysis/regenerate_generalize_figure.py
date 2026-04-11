#!/usr/bin/env python3
"""Regenerate the premium generalization heatmap from frozen run artifacts."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gibbsq.analysis.generalize_regeneration import regenerate_generalize_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Generalization run directory to regenerate from")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where regenerated figures should be written",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="publication",
        help="Plot theme to use for regeneration",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    outputs = regenerate_generalize_figure(
        args.run_dir,
        output_dir=args.output_dir,
        theme=args.theme,
    )

    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
