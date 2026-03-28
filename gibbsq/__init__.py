"""Compatibility shim for the workspace src-layout package.

This makes ``import gibbsq`` resolve to this repository even when only the
workspace root, not ``src``, is on ``sys.path``.
"""

from __future__ import annotations

from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent
_SRC_PKG_DIR = _PKG_DIR.parent / "src" / "gibbsq"

if not _SRC_PKG_DIR.is_dir():
    raise ImportError(f"Expected src package directory at {_SRC_PKG_DIR}")

__path__ = [str(_PKG_DIR), str(_SRC_PKG_DIR)]
__file__ = str(_SRC_PKG_DIR / "__init__.py")

_globals = globals()
exec(compile((_SRC_PKG_DIR / "__init__.py").read_text(encoding="utf-8"), __file__, "exec"), _globals, _globals)
