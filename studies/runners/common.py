from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from gibbsq.qroute.utils.run_artifacts import resolve_output_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = "outputs/final"


def resolve_config_output_dir(config_name: str) -> Path:
    """Resolve the output root declared by a config profile."""
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw_cfg = OmegaConf.load(config_path)
    output_dir = OmegaConf.select(raw_cfg, "output_dir")
    if not output_dir:
        raise ValueError(f"Config {config_name} does not define output_dir")
    return resolve_output_root(str(output_dir), project_root=PROJECT_ROOT)


def resolve_runner_output_dir(config_name: str, output_dir: str | None) -> Path:
    """Resolve the runner output root, preferring the config profile."""
    if output_dir:
        return resolve_output_root(output_dir, project_root=PROJECT_ROOT)
    return resolve_config_output_dir(config_name)


def build_module_command(
    *,
    module: str,
    config_name: str,
    output_dir: Path | None,
    hydra: bool,
    include_config_name: bool = True,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build a child command line for an experiment module."""
    cmd = [sys.executable, "-m", module]
    if include_config_name:
        cmd.extend(["--config-name", config_name])
    if hydra:
        cmd.append(f"++active_profile={config_name}")
        if output_dir is not None:
            cmd.append(f"++output_dir={output_dir}")
    elif output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    cmd.extend(extra_args)
    return cmd


def launch_module(
    *,
    module: str,
    config_name: str,
    output_dir: Path | None,
    hydra: bool,
    include_config_name: bool = True,
    extra_args: Sequence[str] = (),
    experiment_type: str | None = None,
) -> list[Path]:
    """Run a module and return the new capsule directory when detectable."""
    tracked_root = None if output_dir is None or experiment_type is None else output_dir / experiment_type
    before = set(tracked_root.iterdir()) if tracked_root and tracked_root.exists() else set()
    cmd = build_module_command(
        module=module,
        config_name=config_name,
        output_dir=output_dir,
        hydra=hydra,
        include_config_name=include_config_name,
        extra_args=extra_args,
    )
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if tracked_root is None:
        return [output_dir] if output_dir is not None else []
    after = set(tracked_root.iterdir()) if tracked_root.exists() else set()
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if created:
        return [created[-1]]
    if tracked_root.exists():
        latest = sorted(tracked_root.iterdir(), key=lambda path: path.stat().st_mtime)
        if latest:
            return [latest[-1]]
    return [tracked_root]
