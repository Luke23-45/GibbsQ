"""
Configuration loader for the GibbsQ analysis framework.

Loads ``configs/base.yaml`` via OmegaConf and auto-resolves the
``paths.project_root`` and ``paths.data_root`` values based on
the location of this file on disk.

Usage::

    from studies.analysis.common.config import get_config

    cfg = get_config()
    data_root = cfg.paths.data_root

Override at runtime via OmegaConf merge::

    from studies.analysis.common.config import get_config
    cfg = get_config(overrides={"paths.data_root": "/custom/path"})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Optional[DictConfig] = None
_CONFIGS_DIR: Path = Path(__file__).resolve().parent.parent / "configs"
_DEFAULT_CONFIG: Path = _CONFIGS_DIR / "base.yaml"


def _detect_project_root() -> Path:
    """Walk up from this file to find the GibbsQ project root.

    Layout: analysis/common/config.py → common/ → analysis/ → studies/ → GibbsQ/
    """
    return Path(__file__).resolve().parent.parent.parent.parent


def _detect_data_root() -> Path:
    """Resolve the outputs/data/ directory."""
    return _detect_project_root() / "outputs" / "data"


def get_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    use_cache: bool = True,
) -> DictConfig:
    """Load and return the analysis configuration.

    On first call, loads ``configs/base.yaml``, sets auto-detected
    ``paths.project_root`` and ``paths.data_root``, resolves all
    OmegaConf interpolations, and caches the result.

    Parameters
    ----------
    config_path : Path, optional
        Path to a YAML config file.  Defaults to ``configs/base.yaml``.
    overrides : dict, optional
        Key-value pairs to merge on top of the loaded config.
        Dot-notation keys are supported (e.g. ``{"paths.data_root": "..."}``).
    use_cache : bool
        If True (default), return the cached config on subsequent calls.
        Set to False to force a fresh load (useful in tests).

    Returns
    -------
    DictConfig
        Fully resolved OmegaConf configuration.
    """
    global _CONFIG_CACHE

    if use_cache and _CONFIG_CACHE is not None and overrides is None:
        return _CONFIG_CACHE

    cfg_path = config_path or _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Analysis config not found: {cfg_path}. "
            f"Expected at: {_DEFAULT_CONFIG}"
        )

    cfg = OmegaConf.load(cfg_path)
    assert isinstance(cfg, DictConfig)

    if cfg.paths.project_root is None:
        project_root = _detect_project_root()
        OmegaConf.update(cfg, "paths.project_root", str(project_root))
        logger.debug("Auto-detected project_root: %s", project_root)

    if cfg.paths.data_root is None:
        data_root = _detect_data_root()
        OmegaConf.update(cfg, "paths.data_root", str(data_root))
        logger.debug("Auto-detected data_root: %s", data_root)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(
            [f"{k}={v}" for k, v in overrides.items()]
        )
        cfg = OmegaConf.merge(cfg, override_cfg)

    if overrides is None:
        _CONFIG_CACHE = cfg

    logger.info(
        "Config loaded: data_root=%s",
        OmegaConf.select(cfg, "paths.data_root"),
    )
    return cfg


def clear_config_cache() -> None:
    """Clear the cached configuration (useful for testing)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def get_data_root(cfg: Optional[DictConfig] = None) -> Path:
    """Return the resolved data root as a Path object.

    Parameters
    ----------
    cfg : DictConfig, optional
        Pre-loaded config.  Loads default if None.

    Returns
    -------
    Path
        Absolute path to ``outputs/data/``.
    """
    if cfg is None:
        cfg = get_config()
    return Path(cfg.paths.data_root)


def get_study_data_dir(
    study: str,
    cfg: Optional[DictConfig] = None,
) -> Path:
    """Return the absolute data directory for a specific study.

    Parameters
    ----------
    study : str
        One of: ``"convergence"``, ``"equilibrium"``,
        ``"drift_audit"``, ``"boundary_mismatch"``, ``"benchmarks"``.
    cfg : DictConfig, optional
        Pre-loaded config.

    Returns
    -------
    Path
        Absolute path to the study's data directory.

    Raises
    ------
    KeyError
        If the study name is not found in ``paths``.
    """
    if cfg is None:
        cfg = get_config()

    data_root = Path(cfg.paths.data_root)
    relative_dir = OmegaConf.select(cfg, f"paths.{study}")

    if relative_dir is None:
        available = [
            k for k in cfg.paths
            if k not in ("project_root", "data_root", "output_root")
        ]
        raise KeyError(
            f"Unknown study '{study}'. "
            f"Available studies in config: {available}"
        )

    return data_root / relative_dir


def get_output_root(cfg: Optional[DictConfig] = None) -> Path:
    """Return the resolved output root as a Path object.

    Parameters
    ----------
    cfg : DictConfig, optional
        Pre-loaded config.

    Returns
    -------
    Path
        Absolute path to the analysis output directory.
    """
    if cfg is None:
        cfg = get_config()
    project_root = Path(cfg.paths.project_root)
    return project_root / cfg.paths.output_root
