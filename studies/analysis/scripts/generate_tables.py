#!/usr/bin/env python3
"""
Generate thesis-quality tables from z2 experiment CSV data.

This module reads the CSV files produced by the z2 experiments and
generates LaTeX and Markdown tables suitable for direct inclusion
in the thesis manuscript.

Usage:
    python -m studies.analysis.scripts.generate_tables
    python -m studies.analysis.scripts.generate_tables --data-dir outputs/data
    python -m studies.analysis.scripts.generate_tables --table-dir outputs/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "outputs/data"
DEFAULT_TABLE_DIR = "outputs/tables"


def _find_latest_csv(data_dir: Path, prefix: str) -> Path | None:
    """Find the most recent CSV file matching a given prefix."""
    candidates = sorted(
        data_dir.glob(f"{prefix}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and return rows as dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_file(path: Path, content: str) -> None:
    """Write content to a file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log.info("  Written: %s", path)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Table 1: Boundary Equilibrium Results (H2)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def table_boundary_equilibrium(data_dir: Path, table_dir: Path) -> list[Path]:
    """Generate boundary-equilibrium verification table."""
    csv_path = _find_latest_csv(data_dir, "boundary_equilibrium_verification")
    if csv_path is None:
        log.warning("No boundary_equilibrium_verification CSV â€” skipping")
        return []

    rows = _read_csv(csv_path)
    paths: list[Path] = []

    # â”€â”€ LaTeX â”€â”€
    latex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Boundary equilibrium verification results (H2).}",
        r"  \label{tab:boundary_equilibrium}",
        r"  \begin{tabular}{lrrrrr}",
        r"    \toprule",
        r"    System & $N$ & $\rho$ & $\|q^*_{\text{exact}} - q^*_{\text{ODE}}\|_\infty$ & Active Set & Status \\",
        r"    \midrule",
    ]
    for r in rows:
        active_size = r["active_set_size"]
        disc = float(r["max_discrepancy"])
        sys_id = r['system_id'].replace('_', r'\_')
        latex_lines.append(
            f"    {sys_id} & "
            f"{r['N']} & "
            f"{float(r['rho']):.4f} & "
            f"{disc:.2e} & "
            f"{active_size}/{r['N']} & "
            f"{r['status']} \\\\"
        )
    latex_lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    tex_path = table_dir / "h2_boundary_equilibrium.tex"
    _write_file(tex_path, "\n".join(latex_lines))
    paths.append(tex_path)

    # â”€â”€ Markdown â”€â”€
    md_lines = [
        "# Boundary Equilibrium Verification (H2)",
        "",
        "| System | N | Ï | Max Discrepancy | Active Set | Status |",
        "|--------|---|---|----------------|-----------|--------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['system_id']} | {r['N']} | {float(r['rho']):.4f} | "
            f"{float(r['max_discrepancy']):.2e} | "
            f"{r['active_set_size']}/{r['N']} | {r['status']} |"
        )
    md_path = table_dir / "h2_boundary_equilibrium.md"
    _write_file(md_path, "\n".join(md_lines))
    paths.append(md_path)

    return paths


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Table 2: Reflected-ODE Convergence Summary (H1/H2)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def table_reflected_ode(data_dir: Path, table_dir: Path) -> list[Path]:
    """Generate reflected-ODE convergence summary table."""
    csv_path = _find_latest_csv(data_dir, "reflected_ode_convergence_summary")
    if csv_path is None:
        log.warning("No reflected_ode_convergence_summary CSV â€” skipping")
        return []

    rows = _read_csv(csv_path)
    paths: list[Path] = []

    # â”€â”€ LaTeX â”€â”€
    latex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Reflected-ODE multi-start convergence results (H1/H2).}",
        r"  \label{tab:reflected_ode_convergence}",
        r"  \begin{tabular}{lrrrrl}",
        r"    \toprule",
        r"    System & $N$ & Trajectories & Diameter & $\|\dot{q}(T)\|_\infty$ & Status \\",
        r"    \midrule",
    ]
    for r in rows:
        sys_id = r['system_id'].replace('_', r'\_')
        latex_lines.append(
            f"    {sys_id} & "
            f"{r['N']} & "
            f"{r['num_trajectories']} & "
            f"{float(r['terminal_diameter']):.2e} & "
            f"{float(r['max_residual_norm']):.2e} & "
            f"{r['status']} \\\\"
        )
    latex_lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    tex_path = table_dir / "h1h2_reflected_ode_convergence.tex"
    _write_file(tex_path, "\n".join(latex_lines))
    paths.append(tex_path)

    # â”€â”€ Markdown â”€â”€
    md_lines = [
        "# Reflected-ODE Multi-Start Convergence (H1/H2)",
        "",
        "| System | N | Trajectories | Diameter | Max Residual | Status |",
        "|--------|---|-------------|----------|-------------|--------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['system_id']} | {r['N']} | {r['num_trajectories']} | "
            f"{float(r['terminal_diameter']):.2e} | "
            f"{float(r['max_residual_norm']):.2e} | {r['status']} |"
        )
    md_path = table_dir / "h1h2_reflected_ode_convergence.md"
    _write_file(md_path, "\n".join(md_lines))
    paths.append(md_path)

    return paths


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Table 3: Exhaustive Drift Audit (H4)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def table_drift_audit(data_dir: Path, table_dir: Path) -> list[Path]:
    """Generate exhaustive drift audit table."""
    csv_path = _find_latest_csv(data_dir, "exhaustive_drift_summary")
    if csv_path is None:
        log.warning("No exhaustive_drift_summary CSV â€” skipping")
        return []

    rows = _read_csv(csv_path)
    paths: list[Path] = []

    # â”€â”€ LaTeX â”€â”€
    latex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Exhaustive small-grid CTMC drift audit (H4).}",
        r"  \label{tab:drift_audit}",
        r"  \begin{tabular}{lrrrrrrl}",
        r"    \toprule",
        r"    System & $N$ & $\rho$ & States & Violations & $\varepsilon$ & $R$ & Status \\",
        r"    \midrule",
    ]
    for r in rows:
        sys_id = r['system_id'].replace('_', r'\_')
        latex_lines.append(
            f"    {sys_id} & "
            f"{r['N']} & "
            f"{float(r['rho']):.4f} & "
            f"{r['total_states']} & "
            f"{r['violations']} & "
            f"{float(r['epsilon']):.4f} & "
            f"{float(r['R']):.2f} & "
            f"{r['status']} \\\\"
        )
    latex_lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    tex_path = table_dir / "h4_exhaustive_drift_audit.tex"
    _write_file(tex_path, "\n".join(latex_lines))
    paths.append(tex_path)

    # â”€â”€ Markdown â”€â”€
    md_lines = [
        "# Exhaustive Small-Grid CTMC Drift Audit (H4)",
        "",
        "| System | N | Ï | States | Violations | Îµ | R | Status |",
        "|--------|---|---|--------|-----------|---|---|--------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['system_id']} | {r['N']} | {float(r['rho']):.4f} | "
            f"{r['total_states']} | {r['violations']} | "
            f"{float(r['epsilon']):.4f} | {float(r['R']):.2f} | {r['status']} |"
        )
    md_path = table_dir / "h4_exhaustive_drift_audit.md"
    _write_file(md_path, "\n".join(md_lines))
    paths.append(md_path)

    return paths


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Table 4: Boundary Mismatch Summary (H3)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def table_boundary_mismatch(data_dir: Path, table_dir: Path) -> list[Path]:
    """Generate boundary-mismatch summary table."""
    csv_path = _find_latest_csv(data_dir, "ctmc_boundary_mismatch_summary")
    if csv_path is None:
        log.warning("No ctmc_boundary_mismatch_summary CSV â€” skipping")
        return []

    rows = _read_csv(csv_path)
    paths: list[Path] = []

    # â”€â”€ LaTeX â”€â”€
    latex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{CTMC generator boundary mismatch demonstration (H3).}",
        r"  \label{tab:boundary_mismatch}",
        r"  \begin{tabular}{lrrrrrl}",
        r"    \toprule",
        r"    System & $N$ & Boundary States & Positive $\partial$-term & Max Gap & Obstruction \\",
        r"    \midrule",
    ]
    for r in rows:
        sys_id = r['system_id'].replace('_', r'\_')
        latex_lines.append(
            f"    {sys_id} & "
            f"{r['N']} & "
            f"{r['boundary_states']} & "
            f"{r['positive_boundary_count']} & "
            f"{float(r['max_gap']):.4f} & "
            f"{'Yes' if r['obstruction_demonstrated'] == 'True' else 'No'} \\\\"
        )
    latex_lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    tex_path = table_dir / "h3_boundary_mismatch.tex"
    _write_file(tex_path, "\n".join(latex_lines))
    paths.append(tex_path)

    # â”€â”€ Markdown â”€â”€
    md_lines = [
        "# CTMC Generator Boundary Mismatch (H3)",
        "",
        "| System | N | Boundary States | Positive âˆ‚-term | Max Gap | Obstruction |",
        "|--------|---|----------------|----------------|---------|------------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['system_id']} | {r['N']} | {r['boundary_states']} | "
            f"{r['positive_boundary_count']} | "
            f"{float(r['max_gap']):.4f} | "
            f"{'Yes' if r['obstruction_demonstrated'] == 'True' else 'No'} |"
        )
    md_path = table_dir / "h3_boundary_mismatch.md"
    _write_file(md_path, "\n".join(md_lines))
    paths.append(md_path)

    return paths


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Table 5: Benchmark Empirical Performance (H5)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def table_benchmark_empirical(data_dir: Path, table_dir: Path) -> list[Path]:
    """Generate benchmark empirical performance table."""
    csv_path = _find_latest_csv(data_dir, "benchmark_rerun_policies")
    if csv_path is None:
        log.warning("No benchmark_rerun_policies CSV â€” skipping")
        return []

    rows = _read_csv(csv_path)
    paths: list[Path] = []
    
    # Aggregate across seed blocks by policy
    aggregated = {}
    for r in rows:
        pol = r["policy"]
        if pol not in aggregated:
            aggregated[pol] = {"q_total": [], "gini": [], "sojourn": []}
        aggregated[pol]["q_total"].append(float(r["mean_q_total"]))
        aggregated[pol]["gini"].append(float(r["mean_gini"]))
        aggregated[pol]["sojourn"].append(float(r["mean_sojourn"]))
        
    summary = []
    for pol, metrics in aggregated.items():
        summary.append({
            "policy": pol,
            "q_total": sum(metrics["q_total"]) / len(metrics["q_total"]),
            "gini": sum(metrics["gini"]) / len(metrics["gini"]),
            "sojourn": sum(metrics["sojourn"]) / len(metrics["sojourn"]),
        })

    # Sort by Q total ascending
    summary.sort(key=lambda x: x["q_total"])

    # â”€â”€ LaTeX â”€â”€
    latex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Benchmark empirical performance (H5). Average across seed blocks.}",
        r"  \label{tab:benchmark_empirical}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Policy & $\mathbb{E}[|Q|_1]$ & Gini Index & Mean Sojourn \\",
        r"    \midrule",
    ]
    for s in summary:
        policy = s['policy'].replace('_', r'\_')
        latex_lines.append(
            f"    {policy} & "
            f"{s['q_total']:.2f} & "
            f"{s['gini']:.4f} & "
            f"{s['sojourn']:.3f} \\\\"
        )
    latex_lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    tex_path = table_dir / "h5_benchmark_empirical.tex"
    _write_file(tex_path, "\n".join(latex_lines))
    paths.append(tex_path)

    # â”€â”€ Markdown â”€â”€
    md_lines = [
        "# Benchmark Empirical Performance (H5)",
        "",
        "| Policy | E[Q_total] | Gini Index | Mean Sojourn |",
        "|--------|------------|------------|--------------|",
    ]
    for s in summary:
        md_lines.append(
            f"| {s['policy']} | {s['q_total']:.2f} | {s['gini']:.4f} | {s['sojourn']:.3f} |"
        )
    md_path = table_dir / "h5_benchmark_empirical.md"
    _write_file(md_path, "\n".join(md_lines))
    paths.append(md_path)

    return paths


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Master table generation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ALL_TABLES = [
    ("H2: Boundary Equilibrium", table_boundary_equilibrium),
    ("H1/H2: Reflected-ODE Convergence", table_reflected_ode),
    ("H4: Exhaustive Drift Audit", table_drift_audit),
    ("H3: Boundary Mismatch", table_boundary_mismatch),
    ("H5: Benchmark Empirical Performance", table_benchmark_empirical),
]


def generate_all_tables(
    data_dir: str | Path,
    table_dir: str | Path,
) -> list[Path]:
    """Generate all thesis tables."""
    data_dir = Path(data_dir)
    table_dir = Path(table_dir)

    all_paths: list[Path] = []
    for label, func in ALL_TABLES:
        log.info("Generating table: %s", label)
        try:
            paths = func(data_dir, table_dir)
            all_paths.extend(paths)
        except Exception as exc:
            log.error("  Table generation failed for %s: %s", label, exc)

    log.info("Generated %d table files total", len(all_paths))
    return all_paths


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][tables] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX and Markdown tables from z2 experiment CSV data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    paths = generate_all_tables(args.data_dir, args.table_dir)
    log.info("Done: %d tables generated", len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

