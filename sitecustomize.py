"""Workspace-local Python startup customization.

Ensures this repository's ``src`` tree is importable before any sibling
workspace package with the same top-level name.
"""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

if _SRC.is_dir():
    src_str = str(_SRC)
    try:
        sys.path.remove(src_str)
    except ValueError:
        pass
    sys.path.insert(0, src_str)
