"""
Table generation for the convergence study (H1/H2).

Produces LaTeX and Markdown tables for reflected-ODE convergence results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from analysis.common.io import find_latest_csv, read_csv_rows, write_file

logger = logging.getLogger(__name__)


def table_reflected_ode(data_dir: Path, table_dir: Path) -> List[Path]:
    """Generate reflected-ODE convergence summary table."""
    csv_path = find_latest_csv(data_dir, "reflected_ode_convergence_summary")
    if csv_path is None:
        logger.warning("No reflected_ode_convergence_summary CSV — skipping")
        return []

    rows = read_csv_rows(csv_path)
    paths: list[Path] = []

    # ── LaTeX ──
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
        latex_lines.append(
            f"    {r['system_id'].replace('_', r'\\_')} & "
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
    write_file(tex_path, "\n".join(latex_lines))
    paths.append(tex_path)

    # ── Markdown ──
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
    write_file(md_path, "\n".join(md_lines))
    paths.append(md_path)

    return paths
