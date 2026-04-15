import json
import uuid
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from experiments.evaluation.n_gibbsq_evals.ablation_ssa import variant_catalog
from gibbsq.analysis.ablation_regeneration import (
    AblationRecord,
    CANONICAL_VARIANTS,
    build_ablation_plot_payload,
    load_validated_ablation_records,
    regenerate_ablation_figure,
)


def _write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _sample_rows():
    return [
        {
            "variant": "No Log-Norm",
            "variant_kind": "neural",
            "panel": "architecture",
            "preprocessing": "none",
            "init_type": "standard",
            "bootstrap_mode": "expert",
            "teacher_policy": "uas",
            "mean_q_total": 11.22,
            "se_q_total": 0.08,
            "ci95_half_width": 0.16,
            "delta_vs_calibrated_uas_mean": 0.98,
            "delta_vs_best_neural_mean": 0.31,
        },
        {
            "variant": "Zero-Init Final",
            "variant_kind": "neural",
            "panel": "architecture",
            "preprocessing": "log1p",
            "init_type": "zero_final",
            "bootstrap_mode": "expert",
            "teacher_policy": "uas",
            "mean_q_total": 11.48,
            "se_q_total": 0.08,
            "ci95_half_width": 0.16,
            "delta_vs_calibrated_uas_mean": 1.24,
            "delta_vs_best_neural_mean": 0.57,
        },
        {
            "variant": "BC from UAS -> REINFORCE",
            "variant_kind": "neural",
            "panel": "teacher",
            "preprocessing": "log1p",
            "init_type": "standard",
            "bootstrap_mode": "expert",
            "teacher_policy": "uas",
            "mean_q_total": 10.91,
            "se_q_total": 0.07,
            "ci95_half_width": 0.14,
            "delta_vs_calibrated_uas_mean": 0.67,
            "delta_vs_best_neural_mean": 0.0,
        },
        {
            "variant": "BC from Calibrated UAS -> REINFORCE",
            "variant_kind": "neural",
            "panel": "teacher",
            "preprocessing": "log1p",
            "init_type": "standard",
            "bootstrap_mode": "expert",
            "teacher_policy": "calibrated_uas",
            "mean_q_total": 10.52,
            "se_q_total": 0.06,
            "ci95_half_width": 0.12,
            "delta_vs_calibrated_uas_mean": 0.28,
            "delta_vs_best_neural_mean": -0.32,
        },
        {
            "variant": "REINFORCE from Scratch",
            "variant_kind": "neural",
            "panel": "architecture",
            "preprocessing": "log1p",
            "init_type": "standard",
            "bootstrap_mode": "scratch",
            "teacher_policy": "n/a",
            "mean_q_total": 11.95,
            "se_q_total": 0.09,
            "ci95_half_width": 0.18,
            "delta_vs_calibrated_uas_mean": 1.71,
            "delta_vs_best_neural_mean": 1.04,
        },
        {
            "variant": "JSSQ",
            "variant_kind": "reference",
            "panel": "teacher",
            "preprocessing": "n/a",
            "init_type": "n/a",
            "bootstrap_mode": "expert",
            "teacher_policy": "n/a",
            "mean_q_total": 11.02,
            "se_q_total": 0.03,
            "ci95_half_width": 0.06,
            "delta_vs_calibrated_uas_mean": 0.78,
            "delta_vs_best_neural_mean": 0.18,
        },
        {
            "variant": "Calibrated UAS",
            "variant_kind": "reference",
            "panel": "teacher",
            "preprocessing": "n/a",
            "init_type": "n/a",
            "bootstrap_mode": "expert",
            "teacher_policy": "n/a",
            "mean_q_total": 10.24,
            "se_q_total": 0.02,
            "ci95_half_width": 0.04,
            "delta_vs_calibrated_uas_mean": 0.0,
            "delta_vs_best_neural_mean": -0.60,
        },
        {
            "variant": "UAS (alpha=10.0)",
            "variant_kind": "reference",
            "panel": "teacher",
            "preprocessing": "n/a",
            "init_type": "n/a",
            "bootstrap_mode": "expert",
            "teacher_policy": "n/a",
            "mean_q_total": 11.49,
            "se_q_total": 0.03,
            "ci95_half_width": 0.06,
            "delta_vs_calibrated_uas_mean": 1.25,
            "delta_vs_best_neural_mean": 0.65,
        },
    ]


def _sample_records():
    return [AblationRecord(**row) for row in _sample_rows()]


def _workspace_case_dir(name: str) -> Path:
    case_dir = Path.cwd() / ".test-scratch-ablation" / f"{name}-{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def test_variant_catalog_exposes_new_publication_ready_schema():
    catalog = variant_catalog()

    assert [row["name"] for row in catalog] == CANONICAL_VARIANTS
    assert any(row["name"] == "BC from Calibrated UAS -> REINFORCE" for row in catalog)
    assert any(row["name"] == "Calibrated UAS" and row["variant_kind"] == "reference" for row in catalog)


def test_load_validated_ablation_records_accepts_new_summary_schema():
    summary_path = _workspace_case_dir("accepts-summary") / "metrics" / "ablation_ssa_metrics.jsonl"
    _write_summary(summary_path, _sample_rows())

    records = load_validated_ablation_records(summary_path)

    assert [record.variant for record in records] == CANONICAL_VARIANTS
    assert records[0].variant == "No Log-Norm"
    assert records[3].teacher_policy == "calibrated_uas"
    assert records[6].delta_vs_calibrated_uas_mean == pytest.approx(0.0)


def test_load_validated_ablation_records_rejects_missing_variant():
    rows = _sample_rows()[:-1]
    summary_path = _workspace_case_dir("rejects-missing") / "metrics" / "ablation_ssa_metrics.jsonl"
    _write_summary(summary_path, rows)

    with pytest.raises(ValueError, match="Expected 8 ablation rows"):
        load_validated_ablation_records(summary_path)


def test_load_validated_ablation_records_rejects_legacy_duplicate_neural_base_schema():
    rows = _sample_rows()
    rows.insert(
        0,
        {
            "variant": "Neural-Base",
            "variant_kind": "neural",
            "panel": "architecture",
            "preprocessing": "log1p",
            "init_type": "standard",
            "bootstrap_mode": "expert",
            "teacher_policy": "uas",
            "mean_q_total": 10.84,
            "se_q_total": 0.07,
            "ci95_half_width": 0.14,
            "delta_vs_calibrated_uas_mean": 0.60,
            "delta_vs_best_neural_mean": -0.07,
        },
    )
    summary_path = _workspace_case_dir("rejects-legacy-schema") / "metrics" / "ablation_ssa_metrics.jsonl"
    _write_summary(summary_path, rows)

    with pytest.raises(ValueError, match="Expected 8 ablation rows"):
        load_validated_ablation_records(summary_path)


def test_load_validated_ablation_records_ignores_legacy_refined_delta_keys_without_v3_fields():
    rows = _sample_rows()
    rows[0].pop("delta_vs_calibrated_uas_mean", None)
    rows[0]["delta_vs_refined_uas_mean"] = 0.98
    summary_path = _workspace_case_dir("rejects-legacy-refined-delta") / "metrics" / "ablation_ssa_metrics.jsonl"
    _write_summary(summary_path, rows)

    records = load_validated_ablation_records(summary_path)

    assert records[0].delta_vs_calibrated_uas_mean is None


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
    assert payload["panel"][0] == "architecture"
    assert payload["delta_vs_calibrated_uas_mean"][6] == pytest.approx(0.0)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["data_source"] == "summary_jsonl"


def test_build_ablation_plot_payload_tracks_teacher_and_delta_fields():
    payload = build_ablation_plot_payload(_sample_records())

    assert payload["variant_kind"][0] == "neural"
    assert payload["teacher_policy"][3] == "calibrated_uas"
    assert payload["ci95_half_width"][6] == pytest.approx(0.04)
    assert payload["delta_vs_best_neural_mean"][4] == pytest.approx(1.04)
