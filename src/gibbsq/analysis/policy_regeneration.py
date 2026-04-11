"""Regenerate premium policy figures from saved run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from gibbsq.analysis.plotting import plot_policy_dual_panel
from gibbsq.utils.run_artifacts import metrics_path

def load_policy_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL in {path} at line {line_no}: {exc}") from exc
            rows.append(payload)
    return rows

def regenerate_policy_figure(
    run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    theme: str = "publication",
) -> dict[str, Path | str]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Policy run directory does not exist: {run_dir}")

    summary_path = metrics_path(run_dir, "corrected_comparison_metrics.jsonl")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing policy summary metrics: {summary_path}")

    records = load_policy_records(summary_path)
    if not records:
        raise ValueError("No records found in comparison metrics.")
    
    labels = [r["policy"] for r in records]
    q_values = [r["mean_q_total"] for r in records]
    q_errors = [r["se_q_total"] for r in records]
    tiers = [r["tier"] for r in records]
    
    out_path = Path(output_dir) if output_dir else run_dir / "figures"
    out_path.mkdir(parents=True, exist_ok=True)
    
    fig_base = out_path / "policy_comparison_regenerated"
    
    plot_policy_dual_panel(
        labels=labels,
        q_values=q_values,
        q_errors=q_errors,
        tiers=tiers,
        save_path=fig_base,
        theme=theme,
        formats=["png", "pdf"]
    )
    
    return {
        "figure_png": str(fig_base.with_suffix(".png")),
        "figure_pdf": str(fig_base.with_suffix(".pdf")),
    }
