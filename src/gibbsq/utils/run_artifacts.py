"""Helpers for standardized per-run artifact locations."""

from __future__ import annotations

from pathlib import Path

__all__ = [
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
