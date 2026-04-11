#!/usr/bin/env python3
"""Regenerate the premium ablation figure from frozen run artifacts."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gibbsq.analysis.ablation_regeneration import regenerate_ablation_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Ablation run directory to regenerate from")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where regenerated figures/data should be written",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="publication",
        help="Plot theme to use for regeneration",
    )
    parser.add_argument(
        "--strict-summary",
        action="store_true",
        help="Fail instead of reconstructing metrics when the saved summary JSONL is invalid",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    outputs = regenerate_ablation_figure(
        args.run_dir,
        output_dir=args.output_dir,
        theme=args.theme,
        allow_fallback_reconstruction=not args.strict_summary,
    )

    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
