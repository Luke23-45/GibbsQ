"""
Table generation for the mismatch study (H3).

Produces LaTeX and Markdown tables for boundary mismatch results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from studies.analysis.common.io import find_latest_csv, read_csv_rows, write_file

logger = logging.getLogger(__name__)


def table_boundary_mismatch(data_dir: Path, table_dir: Path) -> List[Path]:
    """Generate boundary-mismatch summary table."""
    csv_path = find_latest_csv(data_dir, "ctmc_boundary_mismatch_summary")
    if csv_path is None:
        logger.warning("No ctmc_boundary_mismatch_summary CSV — skipping")
        return []

    rows = read_csv_rows(csv_path)
    paths: list[Path] = []

    # ── LaTeX ──
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
    write_file(tex_path, "\n".join(latex_lines))
    paths.append(tex_path)

    # ── Markdown ──
    md_lines = [
        "# CTMC Generator Boundary Mismatch (H3)",
        "",
        "| System | N | Boundary States | Positive ∂-term | Max Gap | Obstruction |",
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
    write_file(md_path, "\n".join(md_lines))
    paths.append(md_path)

    return paths
