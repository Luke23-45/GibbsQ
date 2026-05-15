"""
Table generation for the drift study (H4).

Produces LaTeX and Markdown tables for exhaustive drift audit results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from analysis.common.io import find_latest_csv, read_csv_rows, write_file

logger = logging.getLogger(__name__)


def table_drift_audit(data_dir: Path, table_dir: Path) -> List[Path]:
    """Generate exhaustive drift audit table."""
    csv_path = find_latest_csv(data_dir, "exhaustive_drift_summary")
    if csv_path is None:
        logger.warning("No exhaustive_drift_summary CSV — skipping")
        return []

    rows = read_csv_rows(csv_path)
    paths: list[Path] = []

    # ── LaTeX ──
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
    write_file(tex_path, "\n".join(latex_lines))
    paths.append(tex_path)

    # ── Markdown ──
    md_lines = [
        "# Exhaustive Small-Grid CTMC Drift Audit (H4)",
        "",
        "| System | N | ρ | States | Violations | ε | R | Status |",
        "|--------|---|---|--------|-----------|---|---|--------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['system_id']} | {r['N']} | {float(r['rho']):.4f} | "
            f"{r['total_states']} | {r['violations']} | "
            f"{float(r['epsilon']):.4f} | {float(r['R']):.2f} | {r['status']} |"
        )
    md_path = table_dir / "h4_exhaustive_drift_audit.md"
    write_file(md_path, "\n".join(md_lines))
    paths.append(md_path)

    return paths
