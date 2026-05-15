#!/usr/bin/env python3
"""
Consolidated CTMC support capsule for the z2 stochastic thesis layer.

This script does not introduce a new theorem claim. It bundles the existing
CTMC-support evidence into one publication-facing artifact:
    1. the H3 boundary-obstruction demonstration,
    2. the H4 direct CTMC sampled audit, and
    3. the H4 exhaustive toy-grid audit.

What it computes:
    - one component-level CSV spanning obstruction, sampled-audit, and
      exhaustive-audit rows
    - one JSON summary of supportive vs non-supportive outcomes
    - one markdown summary suitable for thesis/manuscript support material

What it does not claim:
    - promotion of the direct CTMC route beyond the current z2 status files
    - replacement of the formal proofs or audits in docs/formal_math/z2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gibbsq.experiments.verification import ctmc_boundary_mismatch_demo as cbm  # noqa: E402
from gibbsq.experiments.verification import direct_ctmc_validation as dcv  # noqa: E402
from gibbsq.experiments.verification import exhaustive_drift_audit as eda  # noqa: E402
from gibbsq.qroute.utils.csv_writer import Column, ExperimentCSVWriter  # noqa: E402
from gibbsq.qroute.utils.run_artifacts import (  # noqa: E402
    attach_run_log_handler,
    create_run_capsule,
    metadata_path,
    metrics_dir,
    resolve_output_root,
    write_run_config,
)
from gibbsq.qroute.utils.progress import iter_progress  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "outputs/final"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _find_latest_csv_recursive(root: Path, prefix: str) -> Path | None:
    matches = sorted(
        root.rglob(f"{prefix}_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _find_latest_json_recursive(root: Path, name: str) -> Path | None:
    matches = sorted(
        root.rglob(name),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def run_ctmc_support_summary(
    output_dir: str | Path,
    *,
    config_name: str = dcv.DEFAULT_CONFIG_NAME,
    overrides: Sequence[str] = (),
    protocol: dcv.ValidationProtocol | None = None,
    boundary_max_norm: int = cbm.DEFAULT_MAX_NORM,
    exhaustive_max_norm: int = eda.DEFAULT_MAX_NORM,
    boundary_systems: list[cbm.SystemSpec] | None = None,
    exhaustive_systems: list[eda.ToySystem] | None = None,
    candidates: Sequence[dcv.CandidateSpec] | None = None,
) -> tuple[Path, Path]:
    """Run the consolidated CTMC support capsule."""
    protocol = protocol or dcv.ValidationProtocol()
    base_dir = resolve_output_root(output_dir)
    boundary_systems = list(boundary_systems or cbm.demo_systems())
    exhaustive_systems = list(exhaustive_systems or eda.toy_systems())
    candidates = list(candidates or dcv.default_candidate_catalog())

    run_dir, _ = create_run_capsule(base_dir, "ctmc_support_summary")
    attach_run_log_handler(run_dir)
    write_run_config(
        run_dir,
        {
            "experiment_name": "ctmc_support_summary",
            "output_dir": str(base_dir),
            "config_name": config_name,
            "boundary_max_norm": boundary_max_norm,
            "exhaustive_max_norm": exhaustive_max_norm,
            "candidate_names": [candidate.name for candidate in candidates],
        },
    )

    component_writer = ExperimentCSVWriter(
        experiment_name="ctmc_support_components",
        output_dir=metrics_dir(run_dir),
        columns=[
            Column("component_type", str, "boundary_obstruction / direct_audit / exhaustive_audit"),
            Column("item_id", str, "System or candidate identifier"),
            Column("hypothesis", str, "Primary hypothesis id"),
            Column("theorem_reference", str, "Formal-math source"),
            Column("status", str, "PASS / FAIL / SUPPORTIVE / MIXED"),
            Column("supportive", bool, "Whether the component supports its intended claim"),
            Column("benchmark_default", bool, "Whether this row is the benchmark default point"),
            Column("positive_boundary_count", int, "H3 obstruction count"),
            Column("obstruction_demonstrated", bool, "Whether boundary obstruction was exhibited"),
            Column("violations", int, "Count of violations for exhaustive audit"),
            Column("epsilon", float, "Direct-CTMC theorem constant epsilon"),
            Column("R", float, "Direct-CTMC theorem constant R"),
            Column("sampled_bound_pass", bool, "Whether sampled direct audit passed"),
            Column("max_residual", float, "Worst residual for this component"),
            Column("details_json", str, "Compact JSON with component-specific details"),
        ],
        metadata={
            "hypotheses": ["H3", "H4"],
            "description": "Consolidated stochastic support capsule for the z2 thesis program.",
        },
    )

    boundary_summary_path = _find_latest_csv_recursive(
        base_dir,
        "ctmc_boundary_mismatch_summary",
    )
    if boundary_summary_path is None:
        _, boundary_summary_path = cbm.run_boundary_mismatch_demo(
            boundary_systems,
            base_dir,
            max_norm=boundary_max_norm,
        )
    boundary_rows = _read_csv_rows(boundary_summary_path)

    exhaustive_summary_path = _find_latest_csv_recursive(
        base_dir,
        "exhaustive_drift_summary",
    )
    if exhaustive_summary_path is None:
        _, exhaustive_summary_path = eda.run_exhaustive_audit(
            exhaustive_systems,
            base_dir,
            max_norm=exhaustive_max_norm,
        )
    exhaustive_rows = _read_csv_rows(exhaustive_summary_path)

    audit_summary_path = _find_latest_json_recursive(
        base_dir,
        "direct_ctmc_audit_summary.json",
    )
    audit_jsonl_path = _find_latest_json_recursive(
        base_dir,
        "direct_ctmc_audit.jsonl",
    )
    if audit_summary_path is None:
        cfg, resolved_raw = dcv.load_policy_experiment_config(
            config_name=config_name,
            overrides=overrides,
            protocol=protocol,
            output_dir=str(base_dir),
        )
        audit_run_dir, _ = dcv.get_run_config(cfg, "direct_ctmc_validation", resolved_raw)
        audit_rows = dcv.run_audit(
            cfg=cfg,
            run_dir=audit_run_dir,
            candidates=candidates,
            protocol=protocol,
        )
        audit_summary_path = metadata_path(audit_run_dir, "direct_ctmc_audit_summary.json")
        audit_jsonl_path = audit_run_dir / "metrics" / "direct_ctmc_audit.jsonl"
    else:
        audit_run_dir = audit_summary_path.parent.parent
        audit_payload = json.loads(audit_summary_path.read_text(encoding="utf-8"))
        audit_rows = list(audit_payload["audit_rows"])

    supportive_counts = {"H3": 0, "H4": 0}
    total_counts = {"H3": 0, "H4": 0}

    for row in iter_progress(
        boundary_rows,
        total=len(boundary_rows),
        desc="support summary (H3)",
    ):
        supportive = row["obstruction_demonstrated"].lower() == "true"
        total_counts["H3"] += 1
        supportive_counts["H3"] += int(supportive)
        component_writer.write_row(
            {
                "component_type": "boundary_obstruction",
                "item_id": row["system_id"],
                "hypothesis": "H3",
                "theorem_reference": "z2/08_ctmc_generator_analysis.md",
                "status": "SUPPORTIVE" if supportive else "MIXED",
                "supportive": supportive,
                "benchmark_default": False,
                "positive_boundary_count": int(row["positive_boundary_count"]),
                "obstruction_demonstrated": supportive,
                "violations": 0,
                "epsilon": 0.0,
                "R": 0.0,
                "sampled_bound_pass": False,
                "max_residual": float(row["max_gap"]),
                "details_json": json.dumps(
                    {
                        "boundary_states": int(row["boundary_states"]),
                        "positive_boundary_fraction": float(row["positive_boundary_fraction"]),
                        "max_positive_boundary_term": float(row["max_positive_boundary_term"]),
                        "summary_csv": str(boundary_summary_path),
                    }
                ),
            }
        )

    for row in iter_progress(
        exhaustive_rows,
        total=len(exhaustive_rows),
        desc="support summary (exhaustive)",
    ):
        supportive = row["status"] == "PASS"
        total_counts["H4"] += 1
        supportive_counts["H4"] += int(supportive)
        component_writer.write_row(
            {
                "component_type": "exhaustive_audit",
                "item_id": row["system_id"],
                "hypothesis": "H4",
                "theorem_reference": "z2/10_direct_ctmc_quadratic_proof_attempt.md",
                "status": row["status"],
                "supportive": supportive,
                "benchmark_default": False,
                "positive_boundary_count": 0,
                "obstruction_demonstrated": False,
                "violations": int(row["violations"]),
                "epsilon": float(row["epsilon"]),
                "R": float(row["R"]),
                "sampled_bound_pass": False,
                "max_residual": float(row["max_residual"]),
                "details_json": json.dumps(
                    {
                        "max_norm": int(row["max_norm"]),
                        "total_states": int(row["total_states"]),
                        "summary_csv": str(exhaustive_summary_path),
                    }
                ),
            }
        )

    for row in iter_progress(
        audit_rows,
        total=len(audit_rows),
        desc="support summary (audit)",
    ):
        supportive = bool(row["passes_sampled_bound"]) and bool(row["has_positive_epsilon"])
        total_counts["H4"] += 1
        supportive_counts["H4"] += int(supportive)
        component_writer.write_row(
            {
                "component_type": "direct_audit",
                "item_id": str(row["name"]),
                "hypothesis": "H4",
                "theorem_reference": "z2/10_direct_ctmc_quadratic_proof_attempt.md",
                "status": "SUPPORTIVE" if supportive else "MIXED",
                "supportive": supportive,
                "benchmark_default": str(row["name"]) == "reflected_default",
                "positive_boundary_count": 0,
                "obstruction_demonstrated": False,
                "violations": int(row["positive_violation_count"]),
                "epsilon": float(row["epsilon"]),
                "R": float(row["R"]),
                "sampled_bound_pass": bool(row["passes_sampled_bound"]),
                "max_residual": float(row["max_sampled_residual"]),
                "details_json": json.dumps(
                    {
                        "family": row["family"],
                        "load_condition_ok": bool(row["load_condition_ok"]),
                        "has_positive_epsilon": bool(row["has_positive_epsilon"]),
                        "state_bank_size": int(row["state_bank_size"]),
                        "worst_state": row["worst_state"],
                        "audit_jsonl": str(audit_jsonl_path or (audit_run_dir / "metrics" / "direct_ctmc_audit.jsonl")),
                    }
                ),
            }
        )

    component_csv_path = component_writer.finalize()

    summary = {
        "description": "Consolidated stochastic support capsule for z2.",
        "claim_tier": "supporting experiment",
        "status_guardrail": (
            "Supportive experimental evidence for H3/H4. Does not by itself promote "
            "the direct CTMC theorem beyond the current z2 formal status files."
        ),
        "protocol": {
            "boundary_max_norm": boundary_max_norm,
            "exhaustive_max_norm": exhaustive_max_norm,
            **dcv.asdict(protocol),
        },
        "artifacts": {
            "component_csv": str(component_csv_path),
            "boundary_summary_csv": str(boundary_summary_path),
            "exhaustive_summary_csv": str(exhaustive_summary_path),
            "direct_audit_dir": str(audit_run_dir),
        },
        "counts": {
            "H3_supportive": supportive_counts["H3"],
            "H3_total": total_counts["H3"],
            "H4_supportive": supportive_counts["H4"],
            "H4_total": total_counts["H4"],
        },
        "benchmark_default": next(
            (
                {
                    "epsilon": float(row["epsilon"]),
                    "R": float(row["R"]),
                    "sampled_bound_pass": bool(row["passes_sampled_bound"]),
                    "max_sampled_residual": float(row["max_sampled_residual"]),
                }
                for row in audit_rows
                if row["name"] == "reflected_default"
            ),
            None,
        ),
    }

    summary_json_path = metadata_path(run_dir, "ctmc_support_summary.json")
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# CTMC Support Summary",
        "",
        "This report aggregates the stochastic support layer for the z2 thesis",
        "program. It is a supporting experiment only; it does not upgrade the",
        "formal theorem status on its own.",
        "",
        "## H3 Obstruction Support",
        f"- Supportive systems: {supportive_counts['H3']} / {total_counts['H3']}",
        "",
        "## H4 Direct-CTMC Support",
        f"- Supportive components: {supportive_counts['H4']} / {total_counts['H4']}",
    ]
    if summary["benchmark_default"] is not None:
        benchmark_default = summary["benchmark_default"]
        report_lines.extend(
            [
                "",
                "## Benchmark Default Direct Audit",
                f"- epsilon: {benchmark_default['epsilon']:.12f}",
                f"- R: {benchmark_default['R']:.12f}",
                f"- sampled_bound_pass: {_bool_str(bool(benchmark_default['sampled_bound_pass']))}",
                f"- max_sampled_residual: {benchmark_default['max_sampled_residual']:.12e}",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Guardrail",
            summary["status_guardrail"],
            "",
            "## Artifacts",
            f"- Component CSV: {component_csv_path}",
            f"- JSON summary: {summary_json_path}",
            f"- Boundary summary CSV: {boundary_summary_path}",
            f"- Exhaustive summary CSV: {exhaustive_summary_path}",
            f"- Direct audit directory: {audit_run_dir}",
        ]
    )
    report_path = metadata_path(run_dir, "ctmc_support_summary.md")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return component_csv_path, summary_json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidated CTMC support capsule for the z2 stochastic thesis layer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-name", default=dcv.DEFAULT_CONFIG_NAME)
    parser.add_argument("--override", action="append", default=[], help="OmegaConf override")
    parser.add_argument("--boundary-max-norm", type=int, default=cbm.DEFAULT_MAX_NORM)
    parser.add_argument("--exhaustive-max-norm", type=int, default=eda.DEFAULT_MAX_NORM)
    parser.add_argument("--num-replications", type=int, default=dcv.ANCHOR_REPLICATIONS)
    parser.add_argument("--sim-time", type=float, default=dcv.ANCHOR_SIM_TIME)
    parser.add_argument("--sample-interval", type=float, default=dcv.ANCHOR_SAMPLE_INTERVAL)
    parser.add_argument("--burn-in-fraction", type=float, default=dcv.ANCHOR_BURN_IN_FRACTION)
    parser.add_argument("--base-seed", type=int, default=dcv.ANCHOR_BASE_SEED)
    parser.add_argument("--alpha-uas", type=float, default=dcv.DEFAULT_UAS_ALPHA)
    parser.add_argument("--alpha-reflected", type=float, default=dcv.DEFAULT_REFLECTED_ALPHA)
    parser.add_argument("--audit-trajectory-reps", type=int, default=dcv.DEFAULT_AUDIT_TRAJECTORY_REPS)
    parser.add_argument("--audit-trajectory-sim-time", type=float, default=dcv.DEFAULT_AUDIT_TRAJECTORY_SIM_TIME)
    parser.add_argument(
        "--audit-trajectory-sample-interval",
        type=float,
        default=dcv.DEFAULT_AUDIT_TRAJECTORY_SAMPLE_INTERVAL,
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Extra beta,gamma,c candidate triplet for the direct audit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    protocol = dcv.ValidationProtocol(
        num_replications=args.num_replications,
        sim_time=args.sim_time,
        sample_interval=args.sample_interval,
        burn_in_fraction=args.burn_in_fraction,
        base_seed=args.base_seed,
        alpha_uas=args.alpha_uas,
        alpha_reflected=args.alpha_reflected,
        audit_trajectory_reps=args.audit_trajectory_reps,
        audit_trajectory_sim_time=args.audit_trajectory_sim_time,
        audit_trajectory_sample_interval=args.audit_trajectory_sample_interval,
    )
    candidates = dcv.extend_candidate_catalog(args.candidate)
    component_csv_path, summary_json_path = run_ctmc_support_summary(
        args.output_dir,
        config_name=args.config_name,
        overrides=args.override,
        protocol=protocol,
        boundary_max_norm=args.boundary_max_norm,
        exhaustive_max_norm=args.exhaustive_max_norm,
        candidates=candidates,
    )
    log.info("Component CSV: %s", component_csv_path)
    log.info("Summary JSON: %s", summary_json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
