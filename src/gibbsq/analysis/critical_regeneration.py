"""Regenerate premium critical load figures from saved run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gibbsq.analysis.plotting import plot_critical_load
from gibbsq.utils.run_artifacts import metrics_path


def regenerate_critical_figure(
    run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    theme: str = "publication",
) -> dict[str, Path | str]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Critical run directory does not exist: {run_dir}")

    summary_path = metrics_path(run_dir, "metrics.jsonl")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing critical metrics: {summary_path}")

    records = []
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError("No records found in critical metrics.")

    records.sort(key=lambda r: r["rho"])
    
    rho_values = np.array([r["rho"] for r in records])
    neural_eq = np.array([r["neural_eq"] for r in records])
    gibbs_eq = np.array([r["gibbs_eq"] for r in records])

    out_path = Path(output_dir) if output_dir else run_dir / "figures"
    out_path.mkdir(parents=True, exist_ok=True)

    fig_base = out_path / "critical_load_regenerated"

    plot_critical_load(
        rho_values=rho_values,
        neural_eq=neural_eq,
        gibbs_eq=gibbs_eq,
        save_path=fig_base,
        theme=theme,
        formats=["png", "pdf"],
    )

    return {
        "figure_png": str(fig_base.with_suffix(".png")),
        "figure_pdf": str(fig_base.with_suffix(".pdf")),
    }
