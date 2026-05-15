"""
Table generation for the equilibrium study (H2).

Produces LaTeX and Markdown tables for boundary equilibrium results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from studies.analysis.common.io import find_latest_csv, read_csv_rows, write_file

logger = logging.getLogger(__name__)


def table_boundary_equilibrium(data_dir: Path, table_dir: Path) -> List[Path]:
    """Generate boundary-equilibrium verification table."""
    csv_path = find_latest_csv(data_dir, "boundary_equilibrium_verification")
    if csv_path is None:
        logger.warning("No boundary_equilibrium_verification CSV — skipping")
        return []

    rows = read_csv_rows(csv_path)
    paths: list[Path] = []

    # ── LaTeX ──
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
    write_file(tex_path, "\n".join(latex_lines))
    paths.append(tex_path)

    # ── Markdown ──
    md_lines = [
        "# Boundary Equilibrium Verification (H2)",
        "",
        "| System | N | ρ | Max Discrepancy | Active Set | Status |",
        "|--------|---|---|----------------|-----------|--------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['system_id']} | {r['N']} | {float(r['rho']):.4f} | "
            f"{float(r['max_discrepancy']):.2e} | "
            f"{r['active_set_size']}/{r['N']} | {r['status']} |"
        )
    md_path = table_dir / "h2_boundary_equilibrium.md"
    write_file(md_path, "\n".join(md_lines))
    paths.append(md_path)

    return paths
