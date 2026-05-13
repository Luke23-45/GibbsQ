"""Workspace-local Python startup customization.

Ensures this repository's root is importable so that ``gibbsq``
(containing ``qroute`` and ``experiments``) can be imported directly.
"""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent

root_str = str(_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
