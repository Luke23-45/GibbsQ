"""
Data I/O utilities for the GibbsQ analysis framework.

Handles all file loading and path discovery for experiment outputs
produced by the z2 runners (verification + benchmarks).

Key responsibilities
--------------------
1. **CSV discovery** – Find the latest CSV file matching a prefix.
2. **CSV parsing** – Read CSVs into list-of-dict or pandas DataFrames.
3. **JSON/YAML loading** – Parse metadata sidecars and configs.
4. **Figure saving** – Save matplotlib figures as PDF + PNG.

Design notes
------------
* All public functions accept ``pathlib.Path`` objects and perform
  existence checks with informative error messages.
* CSV discovery uses timestamp-based sorting for finding the latest run.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import yaml

logger = logging.getLogger(__name__)


def find_latest_csv(data_dir: Path, prefix: str) -> Optional[Path]:
    """Find the most recent CSV file matching a given prefix.

    Parameters
    ----------
    data_dir : Path
        Directory to search.
    prefix : str
        Filename prefix (e.g., 'exhaustive_drift_summary').

    Returns
    -------
    Path or None
        Path to the most recent matching CSV, or None if not found.
    """
    candidates = sorted(
        data_dir.glob(f"{prefix}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file and return rows as dicts.

    Parameters
    ----------
    path : Path
        Path to a ``.csv`` file.

    Returns
    -------
    list of dict
        Each row as a dict with column headers as keys.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_json(path: Path) -> Any:
    """Load a JSON file with informative error on failure.

    Parameters
    ----------
    path : Path
        Path to a ``.json`` file.

    Returns
    -------
    Any
        Parsed JSON content (dict, list, etc.).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file.

    Parameters
    ----------
    path : Path
        Path to a ``.yaml`` or ``.yml`` file.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_metadata_sidecar(csv_path: Path) -> Optional[Dict[str, Any]]:
    """Read the sidecar metadata JSON for a CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to a CSV file.  Expects a ``.meta.json`` sidecar next to it.

    Returns
    -------
    dict or None
        Parsed metadata, or None if not found.
    """
    meta_path = csv_path.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return None


def write_file(path: Path, content: str) -> None:
    """Write content to a file, creating parent directories.

    Parameters
    ----------
    path : Path
        Output file path.
    content : str
        Content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Written: %s", path)


def resolve_data_root(
    analysis_dir: Optional[Path] = None,
) -> Path:
    """Resolve the ``outputs/data/`` root directory.

    First attempts to load the path from ``configs/base.yaml`` via
    the config module.  Falls back to auto-detection by walking up
    from this file if the config module is not available.

    Parameters
    ----------
    analysis_dir : Path, optional
        Path to the ``analysis/`` directory.  Defaults to auto-detect.

    Returns
    -------
    Path
        Absolute path to ``outputs/data/``.

    Raises
    ------
    FileNotFoundError
        If the root cannot be found.
    """
    # Use the centralized config
    try:
        from analysis.common.config import get_data_root
        data_root = get_data_root()
        if data_root.exists():
            logger.debug("Data root from config: %s", data_root)
            return data_root
    except Exception:
        pass

    # Walk up from this file (legacy fallback)
    if analysis_dir is None:
        analysis_dir = Path(__file__).resolve().parent.parent

    # analysis/ → studies/ → GibbsQ/ → outputs/data/
    data_root = analysis_dir.parent.parent / "outputs" / "data"
    if data_root.exists():
        return data_root

    raise FileNotFoundError(
        f"Cannot locate data root. Searched:\n"
        f"  - Config-based resolution\n"
        f"  - {data_root}\n"
    )
