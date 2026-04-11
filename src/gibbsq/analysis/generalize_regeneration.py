"""Regenerate premium generalization heatmap figures from saved run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gibbsq.analysis.plotting import plot_improvement_heatmap
from gibbsq.utils.run_artifacts import metrics_path


def regenerate_generalize_figure(
    run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    theme: str = "publication",
) -> dict[str, Path | str]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Generalization run directory does not exist: {run_dir}")

    summary_path = metrics_path(run_dir, "metrics.jsonl")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing generalization metrics: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        # Expected shape is monolithic matrix JSON dumped fully on first line
        data = json.loads(f.readline().strip())

    grid = np.array(data["grid"])
    y_labels = [f"{v}x" for v in data["scale_vals"]]
    x_labels = [str(v) for v in data["rho_vals"]]

    out_path = Path(output_dir) if output_dir else run_dir / "figures"
    out_path.mkdir(parents=True, exist_ok=True)

    fig_base = out_path / "generalization_heatmap_regenerated"

    plot_improvement_heatmap(
        grid=grid,
        x_labels=x_labels,
        y_labels=y_labels,
        save_path=fig_base,
        theme=theme,
        formats=["png", "pdf"],
    )

    return {
        "figure_png": str(fig_base.with_suffix(".png")),
        "figure_pdf": str(fig_base.with_suffix(".pdf")),
    }
