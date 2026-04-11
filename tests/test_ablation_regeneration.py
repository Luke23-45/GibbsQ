import json
import uuid
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from gibbsq.analysis.ablation_regeneration import (
    AblationRecord,
    CANONICAL_VARIANTS,
    build_ablation_plot_payload,
    load_validated_ablation_records,
    regenerate_ablation_figure,
)
from gibbsq.analysis.plotting import plot_ablation_dual_panel


def _write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _sample_rows():
    return [
        {
            "variant": "Full Model",
            "preprocessing": "log1p",
            "init_type": "zero_final",
            "mean_q_total": 12.997737203495632,
            "se_q_total": 0.09170239423903785,
        },
        {
            "variant": "Ablated: No Log-Norm",
            "preprocessing": "none",
            "init_type": "zero_final",
            "mean_q_total": 13.027426654182273,
            "se_q_total": 0.09396965078099717,
        },
        {
            "variant": "Ablated: No Zero-Init",
            "preprocessing": "log1p",
            "init_type": "standard",
            "mean_q_total": 12.540652309612984,
            "se_q_total": 0.09167882318080789,
        },
        {
            "variant": "Uniform Routing (Baseline)",
            "preprocessing": "n/a",
            "init_type": "n/a",
            "mean_q_total": 802.6844569288389,
            "se_q_total": 6.734918674865772,
        },
    ]


def _sample_records():
    return [AblationRecord(**row) for row in _sample_rows()]


def _workspace_case_dir(name: str) -> Path:
    case_dir = Path.cwd() / ".test-scratch-ablation" / f"{name}-{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def test_load_validated_ablation_records_accepts_frozen_summary():
    summary_path = _workspace_case_dir("accepts-summary") / "metrics" / "ablation_ssa_metrics.jsonl"
    _write_summary(summary_path, _sample_rows())

    records = load_validated_ablation_records(summary_path)

    assert [record.variant for record in records] == CANONICAL_VARIANTS
    assert records[0].mean_q_total == pytest.approx(12.997737203495632)
    assert records[3].se_q_total == pytest.approx(6.734918674865772)


def test_load_validated_ablation_records_rejects_bad_order():
    rows = _sample_rows()
    rows[1], rows[2] = rows[2], rows[1]
    summary_path = _workspace_case_dir("rejects-order") / "metrics" / "ablation_ssa_metrics.jsonl"
    _write_summary(summary_path, rows)

    with pytest.raises(ValueError, match="Unexpected variant order"):
        load_validated_ablation_records(summary_path)


def test_regenerate_ablation_figure_writes_outputs_from_summary():
    run_dir = _workspace_case_dir("regenerate-summary") / "ablation_run"
    summary_path = run_dir / "metrics" / "ablation_ssa_metrics.jsonl"
    _write_summary(summary_path, _sample_rows())

    outputs = regenerate_ablation_figure(run_dir, allow_fallback_reconstruction=False)

    data_path = outputs["data_json"]
    metadata_path = outputs["metadata_json"]
    assert outputs["figure_png"].exists()
    assert outputs["figure_pdf"].exists()
    assert data_path.exists()
    assert metadata_path.exists()

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    assert payload["variants"] == CANONICAL_VARIANTS
    assert payload["mean_q_total"][0] == pytest.approx(12.997737203495632)
    assert payload["delta_pct_vs_full_model"][1] == pytest.approx(
        ((_sample_rows()[1]["mean_q_total"] - _sample_rows()[0]["mean_q_total"]) / _sample_rows()[0]["mean_q_total"]) * 100.0
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["data_source"] == "summary_jsonl"


def test_plot_ablation_dual_panel_builds_zoomed_premium_layout():
    payload = build_ablation_plot_payload(_sample_records())

    fig = plot_ablation_dual_panel(
        variant_names=payload["variants"],
        mean_values=payload["mean_q_total"],
        se_values=payload["se_q_total"],
    )

    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylim()[1] > 800.0
    assert fig.axes[1].get_ylim()[1] < 20.0
    assert "Component Contributions" in fig.axes[0].get_title()
    assert any("Zoom on learned variants" in text.get_text() for text in fig.axes[1].texts)
