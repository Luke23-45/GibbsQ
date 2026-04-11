"""Regenerate premium sweep figures from saved run artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gibbsq.analysis.plotting import plot_alpha_sweep
from gibbsq.utils.run_artifacts import metrics_path


def regenerate_sweep_figure(
    run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    theme: str = "publication",
) -> dict[str, Path | str]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Sweep run directory does not exist: {run_dir}")

    summary_path = metrics_path(run_dir, "metrics.jsonl")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing sweep metrics: {summary_path}")

    records = []
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError("No records found in sweep metrics.")

    rho_to_vals = defaultdict(list)
    for r in records:
        rho_to_vals[r["rho"]].append(r)

    rho_keys = sorted(rho_to_vals.keys())
    rho_labels = [f"ρ={rho:.2f}" for rho in rho_keys]

    alpha_values = sorted(list({r["alpha"] for r in records}))
    alpha_to_idx = {val: idx for idx, val in enumerate(alpha_values)}

    num_rhos = len(rho_keys)
    num_alphas = len(alpha_values)

    mean_q_matrix = np.full((num_rhos, num_alphas), np.nan, dtype=np.float64)
    stationary_matrix = np.ones((num_rhos, num_alphas), dtype=bool)

    for i, rho in enumerate(rho_keys):
        for r in rho_to_vals[rho]:
            j = alpha_to_idx[r["alpha"]]
            mean_q_matrix[i, j] = r["mean_q_total"]
            stationary_matrix[i, j] = r.get("stationarity_rate", 1.0) >= 0.90

    out_path = Path(output_dir) if output_dir else run_dir / "figures"
    out_path.mkdir(parents=True, exist_ok=True)

    fig_base = out_path / "alpha_sweep_regenerated"

    plot_alpha_sweep(
        np.array(alpha_values),
        mean_q_matrix,
        rho_labels,
        stationary_matrix=stationary_matrix,
        save_path=fig_base,
        theme=theme,
        formats=["png", "pdf"],
    )

    return {
        "figure_png": str(fig_base.with_suffix(".png")),
        "figure_pdf": str(fig_base.with_suffix(".pdf")),
    }
