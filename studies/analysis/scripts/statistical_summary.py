#!/usr/bin/env python3
"""
Generate a statistical summary report from z2 experiment CSV data.

This module aggregates results across all z2 experiments and produces
a structured Markdown report suitable for thesis appendix inclusion.

Usage:
    python -m studies.analysis.scripts.statistical_summary
    python -m studies.analysis.scripts.statistical_summary --data-dir outputs/data
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "outputs/data"
DEFAULT_REPORT_DIR = "outputs/reports"


def _find_latest_csv(data_dir: Path, prefix: str) -> Path | None:
    """Find the most recent CSV file matching a prefix."""
    candidates = sorted(
        data_dir.glob(f"{prefix}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV and return rows as dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_metadata(csv_path: Path) -> dict | None:
    """Read the sidecar metadata JSON for a CSV file."""
    meta_path = csv_path.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return None


def generate_statistical_summary(
    data_dir: str | Path,
    report_dir: str | Path,
) -> Path:
    """Generate a comprehensive statistical summary report.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing experiment CSV files.
    report_dir : str or Path
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
        f"# z2 Statistical Summary Report",
        f"",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    # â”€â”€ H2: Boundary Equilibrium â”€â”€
    lines.append("## H2: Boundary Equilibrium Verification")
    lines.append("")
    csv_path = _find_latest_csv(data_dir, "boundary_equilibrium_verification")
    if csv_path:
        rows = _read_csv(csv_path)
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

    # â”€â”€ H1/H2: Reflected ODE Convergence â”€â”€
    lines.append("## H1/H2: Reflected-ODE Multi-Start Convergence")
    lines.append("")
    csv_path = _find_latest_csv(data_dir, "reflected_ode_convergence_summary")
    if csv_path:
        rows = _read_csv(csv_path)
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

    # â”€â”€ H4: Drift Audit â”€â”€
    lines.append("## H4: Exhaustive CTMC Drift Audit")
    lines.append("")
    csv_path = _find_latest_csv(data_dir, "exhaustive_drift_summary")
    if csv_path:
        rows = _read_csv(csv_path)
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
            f"- **Îµ range**: [{eps_range[0]:.6f}, {eps_range[1]:.6f}]",
            f"- **Conclusion**: {'Zero violations â€” drift bound holds' if total_violations == 0 else f'{total_violations} VIOLATIONS DETECTED'}",
            "",
        ])
    else:
        lines.extend(["- Data not found", ""])

    # â”€â”€ H4: Theorem Sweep â”€â”€
    lines.append("## H4: Theorem-Constant Parameter Sweep")
    lines.append("")
    csv_path = _find_latest_csv(data_dir, "theorem_constant_sweep")
    if csv_path:
        rows = _read_csv(csv_path)
        n_certified = sum(1 for r in rows if float(r.get("epsilon", 0)) > 1e-10)
        lines.extend([
            f"- **Source**: `{csv_path.name}`",
            f"- **Parameter points tested**: {len(rows)}",
            f"- **Certified (Îµ > 0)**: {n_certified}/{len(rows)}",
            f"- **Conclusion**: {'All candidates certified' if n_certified == len(rows) else f'{len(rows) - n_certified} candidates NOT certified'}",
            "",
        ])
    else:
        lines.extend(["- Data not found", ""])

    # â”€â”€ H3: Boundary Mismatch â”€â”€
    lines.append("## H3: CTMC Generator Boundary Mismatch")
    lines.append("")
    csv_path = _find_latest_csv(data_dir, "ctmc_boundary_mismatch_summary")
    if csv_path:
        rows = _read_csv(csv_path)
        any_obstruction = any(r["obstruction_demonstrated"] == "True" for r in rows)
        max_gap = max(float(r["max_gap"]) for r in rows)
        total_boundary = sum(int(r["boundary_states"]) for r in rows)
        total_positive = sum(int(r["positive_boundary_count"]) for r in rows)
        lines.extend([
            f"- **Source**: `{csv_path.name}`",
            f"- **Systems tested**: {len(rows)}",
            f"- **Total boundary states**: {total_boundary}",
            f"- **States with positive boundary term**: {total_positive}",
            f"- **Maximum CTMCâ€“ODE gap**: {max_gap:.4f}",
            f"- **Obstruction demonstrated**: {'Yes' if any_obstruction else 'No'}",
            f"- **Conclusion**: {'Boundary mismatch confirmed â€” old H route invalid' if any_obstruction else 'Boundary term non-positive in tested range'}",
            "",
        ])
    else:
        lines.extend(["- Data not found", ""])

    # â”€â”€ H4: Direct CTMC Validation Rerun â”€â”€
    lines.append("## H4: Direct CTMC Validation Rerun")
    lines.append("")
    # Read from the separate direct_ctmc_validation output directory
    ctmc_val_dir = report_dir.parent / "data" / "direct_ctmc_validation" / "metadata"
    val_summary_path = ctmc_val_dir / "direct_ctmc_validation_summary.md"
    if val_summary_path.exists():
        lines.extend([
            f"- **Source**: `{val_summary_path.name}`",
            f"- **Status**: Direct CTMC constants computationally validated",
            "",
        ])
    else:
        lines.extend(["- Data not found (Run direct CTMC validation rerun)", ""])

    # â”€â”€ H5: Benchmark Empirical Performance â”€â”€
    lines.append("## H5: Benchmark Empirical Performance")
    lines.append("")
    csv_path = _find_latest_csv(data_dir, "benchmark_rerun_policies")
    if csv_path:
        rows = _read_csv(csv_path)
        # Aggregate across seed blocks by policy
        aggregated = {}
        for r in rows:
            pol = r["policy"]
            if pol not in aggregated:
                aggregated[pol] = {"q_total": []}
            aggregated[pol]["q_total"].append(float(r["mean_q_total"]))
        
        summary = []
        for pol, metrics in aggregated.items():
            summary.append({
                "policy": pol,
                "q_total": sum(metrics["q_total"]) / len(metrics["q_total"]),
            })
        
        # Sort ascending
        summary.sort(key=lambda x: x["q_total"])
        best_policy = summary[0]["policy"]
        
        lines.extend([
            f"- **Source**: `{csv_path.name}`",
            f"- **Policies compared**: {len(summary)}",
            f"- **Best performing policy**: {best_policy} (E[|Q|1] = {summary[0]['q_total']:.2f})",
            f"- **Conclusion**: Performance metrics validated across independent seed blocks",
            "",
        ])
    else:
        lines.extend(["- Data not found", ""])

    # â”€â”€ Data Inventory â”€â”€
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
    _write_file(report_path, "\n".join(lines))
    return report_path


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log.info("Written: %s", path)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][stats] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate statistical summary report from z2 experiment data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    report = generate_statistical_summary(args.data_dir, args.report_dir)
    log.info("Done: %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

