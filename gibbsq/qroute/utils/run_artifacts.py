"""Helpers for standardized per-run artifact locations."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import yaml

__all__ = [
    "resolve_output_root",
    "create_run_capsule",
    "attach_run_log_handler",
    "write_run_config",
    "logs_dir",
    "figures_dir",
    "metrics_dir",
    "artifacts_dir",
    "metadata_dir",
    "metadata_path",
    "config_path",
    "metrics_path",
    "figure_path",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def resolve_output_root(output_dir: str | Path, *, project_root: Path | None = None) -> Path:
    """Resolve an output root against the repository root.

    Relative output paths must be anchored to the project root rather than the
    process cwd, because Hydra-enabled entry points may change the working
    directory during execution.
    """
    path = Path(output_dir)
    if path.is_absolute():
        return path.resolve()
    base = project_root or _PROJECT_ROOT
    return (base / path).resolve()


def create_run_capsule(
    output_root: str | Path,
    experiment_type: str,
    *,
    run_prefix: str = "final",
    timestamp: datetime | None = None,
) -> tuple[Path, str]:
    """Create a legacy-style per-run capsule directory.

    The resulting layout is:
    ``<output_root>/<experiment_type>/<run_prefix>_<timestamp>/{artifacts,figures,logs,metadata,metrics}``
    """
    resolved_root = resolve_output_root(output_root)
    stamp = (timestamp or datetime.now()).strftime(_RUN_TIMESTAMP_FORMAT)
    run_id = f"{run_prefix}_{stamp}"
    run_dir = resolved_root / experiment_type / run_id
    logs_dir(run_dir).mkdir(parents=True, exist_ok=True)
    figures_dir(run_dir).mkdir(parents=True, exist_ok=True)
    metrics_dir(run_dir).mkdir(parents=True, exist_ok=True)
    artifacts_dir(run_dir).mkdir(parents=True, exist_ok=True)
    metadata_dir(run_dir).mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def attach_run_log_handler(run_dir: Path) -> Path:
    """Attach a ``run.log`` file handler to the root logger once per path."""
    log_path = logs_dir(run_dir) / "run.log"
    root_logger = logging.getLogger()
    existing = {
        handler.baseFilename
        for handler in root_logger.handlers
        if isinstance(handler, logging.FileHandler)
    }
    resolved = str(log_path.resolve())
    if resolved not in existing:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        )
        root_logger.addHandler(handler)
    return log_path


def write_run_config(run_dir: Path, payload: dict) -> Path:
    """Write a small YAML config manifest into the run capsule metadata."""
    path = config_path(run_dir)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def logs_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "logs"


def figures_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "figures"


def metrics_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "metrics"


def artifacts_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "artifacts"


def metadata_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "metadata"


def metadata_path(run_dir: Path, name: str) -> Path:
    return metadata_dir(run_dir) / name


def config_path(run_dir: Path) -> Path:
    return metadata_path(run_dir, "config.yaml")


def metrics_path(run_dir: Path, name: str = "metrics.jsonl") -> Path:
    return metrics_dir(run_dir) / name


def figure_path(run_dir: Path, stem: str) -> Path:
    return figures_dir(run_dir) / stem
