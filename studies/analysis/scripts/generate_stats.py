"""
Generate statistical summary reports for the GibbsQ thesis.

Usage::

    python -m analysis.scripts.generate_stats [--data-dir PATH] [--report-dir PATH]

Produces Markdown reports aggregating results across all hypothesis tests.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from analysis.common.io import (
    find_latest_csv,
    read_csv_rows,
    load_metadata_sidecar,
    resolve_data_root,
    write_file,
)

logger = logging.getLogger(__name__)


def generate_statistical_summary(
    data_dir: Path,
    report_dir: Path,
) -> Path:
    """Generate a comprehensive statistical summary report.

    Parameters
    ----------
    data_dir : Path
        Directory containing experiment CSV files.
    report_dir : Path
        Output directory for the report.

    Returns
    -------
    Path
        Path to the generated report.
    """
    data_dir = Path(data_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    lines: list[str] = [
        "# z2 Statistical Summary Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    # ── H2: Boundary Equilibrium ──
    lines.append("## H2: Boundary Equilibrium Verification")
    lines.append("")
    csv_path = find_latest_csv(data_dir, "boundary_equilibrium_verification")
    if csv_path:
        rows = read_csv_rows(csv_path)
        n_pass = sum(1 for r in rows if r["status"] == "PASS")
        max_disc = max(float(r["max_discrepancy"]) for r in rows)
        max_comp = max(float(r["max_complementarity_residual"]) for r in rows)
        lines.extend([
            f"- **Source**: `{csv_path.name}`",
            f"- **Systems tested**: {len(rows)}",
            f"- **Pass rate**: {n_pass}/{len(rows)}",
            f"- **Maximum discrepancy**: {max_disc:.2e}",
            f"- **Maximum complementarity residual**: {max_comp:.2e}",
            f"- **Conclusion**: {'All systems verified' if n_pass == len(rows) else 'FAILURES DETECTED'}",
            "",
        ])
    else:
        lines.extend(["- Data not found", ""])

    # ── H1/H2: Reflected ODE Convergence ──
    lines.append("## H1/H2: Reflected-ODE Multi-Start Convergence")
    lines.append("")
    csv_path = find_latest_csv(data_dir, "reflected_ode_convergence_summary")
    if csv_path:
        rows = read_csv_rows(csv_path)
        total_traj = sum(int(r["num_trajectories"]) for r in rows)
        total_conv = sum(int(r["num_converged"]) for r in rows)
        max_diameter = max(float(r["terminal_diameter"]) for r in rows)
        max_residual = max(float(r["max_residual_norm"]) for r in rows)
        lines.extend([
            f"- **Source**: `{csv_path.name}`",
            f"- **Systems tested**: {len(rows)}",
            f"- **Total trajectories**: {total_traj}",
            f"- **Total converged**: {total_conv} ({100*total_conv/total_traj:.1f}%)",
            f"- **Maximum terminal diameter**: {max_diameter:.2e}",
            f"- **Maximum residual norm**: {max_residual:.2e}",
            f"- **Conclusion**: {'Global convergence confirmed' if total_conv == total_traj else 'CONVERGENCE FAILURES'}",
            "",
        ])
    else:
        lines.extend(["- Data not found", ""])

    # ── H4: Drift Audit ──
    lines.append("## H4: Exhaustive CTMC Drift Audit")
    lines.append("")
    csv_path = find_latest_csv(data_dir, "exhaustive_drift_summary")
    if csv_path:
        rows = read_csv_rows(csv_path)
        total_states = sum(int(r["total_states"]) for r in rows)
        total_violations = sum(int(r["violations"]) for r in rows)
        max_residual = max(float(r["max_residual"]) for r in rows)
        eps_range = (
            min(float(r["epsilon"]) for r in rows),
            max(float(r["epsilon"]) for r in rows),
        )
        lines.extend([
            f"- **Source**: `{csv_path.name}`",
            f"- **Systems tested**: {len(rows)}",
            f"- **Total states enumerated**: {total_states:,}",
            f"- **Total violations**: {total_violations}",
            f"- **Maximum residual**: {max_residual:.6e}",
            f"- **ε range**: [{eps_range[0]:.6f}, {eps_range[1]:.6f}]",
            f"- **Conclusion**: {'Zero violations — drift bound holds' if total_violations == 0 else f'{total_violations} VIOLATIONS DETECTED'}",
            "",
        ])
    else:
        lines.extend(["- Data not found", ""])

    # ── H4: Theorem Sweep ──
    lines.append("## H4: Theorem-Constant Parameter Sweep")
    lines.append("")
    csv_path = find_latest_csv(data_dir, "theorem_constant_sweep")
    if csv_path:
        rows = read_csv_rows(csv_path)
        n_certified = sum(1 for r in rows if float(r.get("epsilon", 0)) > 1e-10)
        lines.extend([
            f"- **Source**: `{csv_path.name}`",
            f"- **Parameter points tested**: {len(rows)}",
            f"- **Certified (ε > 0)**: {n_certified}/{len(rows)}",
            f"- **Conclusion**: {'All candidates certified' if n_certified == len(rows) else f'{len(rows) - n_certified} candidates NOT certified'}",
            "",
        ])
    else:
        lines.extend(["- Data not found", ""])

    # ── H3: Boundary Mismatch ──
    lines.append("## H3: CTMC Generator Boundary Mismatch")
    lines.append("")
    csv_path = find_latest_csv(data_dir, "ctmc_boundary_mismatch_summary")
    if csv_path:
        rows = read_csv_rows(csv_path)
        any_obstruction = any(r["obstruction_demonstrated"] == "True" for r in rows)
        max_gap = max(float(r["max_gap"]) for r in rows)
        total_boundary = sum(int(r["boundary_states"]) for r in rows)
        total_positive = sum(int(r["positive_boundary_count"]) for r in rows)
        lines.extend([
            f"- **Source**: `{csv_path.name}`",
            f"- **Systems tested**: {len(rows)}",
            f"- **Total boundary states**: {total_boundary}",
            f"- **States with positive boundary term**: {total_positive}",
            f"- **Maximum CTMC–ODE gap**: {max_gap:.4f}",
            f"- **Obstruction demonstrated**: {'Yes' if any_obstruction else 'No'}",
            f"- **Conclusion**: {'Boundary mismatch confirmed — old H route invalid' if any_obstruction else 'Boundary term non-positive in tested range'}",
            "",
        ])
    else:
        lines.extend(["- Data not found", ""])

    # ── Data Inventory ──
    lines.append("## Data Inventory")
    lines.append("")
    all_csvs = sorted(data_dir.glob("*.csv"))
    all_metas = sorted(data_dir.glob("*.meta.json"))
    lines.extend([
        f"- **CSV files**: {len(all_csvs)}",
        f"- **Metadata sidecars**: {len(all_metas)}",
        "",
        "| File | Size (KB) |",
        "|------|----------|",
    ])
    for csv_file in all_csvs:
        size_kb = csv_file.stat().st_size / 1024
        lines.append(f"| {csv_file.name} | {size_kb:.1f} |")

    lines.append("")
    report_path = report_dir / f"statistical_summary_{timestamp}.md"
    write_file(report_path, "\n".join(lines))
    return report_path


def main(args: argparse.Namespace) -> None:
    """Generate all statistical reports."""
    data_dir = Path(args.data_dir) if args.data_dir else resolve_data_root()
    report_dir = Path(args.report_dir)

    logger.info("Data dir:   %s", data_dir)
    logger.info("Report dir: %s", report_dir)

    report = generate_statistical_summary(data_dir, report_dir)
    logger.info("Report generated: %s", report)


def cli() -> None:
    """Parse CLI arguments and run."""
    parser = argparse.ArgumentParser(
        description="Generate statistical summary reports.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to outputs/data/ directory.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="outputs/reports",
        help="Output directory for reports.",
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
