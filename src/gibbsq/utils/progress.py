"""Shared terminal progress helpers for experiments and pipelines."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - exercised via monkeypatch in tests
    _tqdm = None

PROGRESS_ENV_VAR = "GIBBSQ_PROGRESS"
VALID_PROGRESS_MODES = {"auto", "on", "off"}

def normalize_progress_mode(mode: str | None) -> str:
    """Normalize progress mode values and reject unsupported ones."""
    normalized = (mode or "auto").strip().lower()
    if normalized not in VALID_PROGRESS_MODES:
        raise ValueError(
            f"progress mode must be one of {sorted(VALID_PROGRESS_MODES)}, "
            f"got {mode!r}"
        )
    return normalized

def get_progress_mode(mode: str | None = None) -> str:
    """Resolve progress mode from an explicit override or environment."""
    if mode is not None:
        return normalize_progress_mode(mode)
    return normalize_progress_mode(os.environ.get(PROGRESS_ENV_VAR))

def configure_progress_mode(mode: str | None, env: dict[str, str] | None = None) -> str:
    """Persist a progress mode into an environment mapping."""
    resolved = get_progress_mode(mode)
    target_env = os.environ if env is None else env
    target_env[PROGRESS_ENV_VAR] = resolved
    return resolved

def progress_enabled(
    mode: str | None = None,
    *,
    stream: Any | None = None,
    tqdm_module: Any | None = None,
) -> bool:
    """Return True when live progress bars should be rendered."""
    resolved = get_progress_mode(mode)
    if resolved == "off":
        return False

    tqdm_impl = _tqdm if tqdm_module is None else tqdm_module
    if tqdm_impl is None:
        return False

    if resolved == "on":
        return True

    if os.environ.get("CI"):
        return False

    target_stream = stream if stream is not None else sys.stderr
    is_tty = getattr(target_stream, "isatty", None)
    return bool(is_tty and is_tty())

class NullProgress:
    """No-op progress object with the subset of tqdm's interface we use."""

    def __init__(self, iterable: Iterable[Any] | None = None):
        self.iterable = iterable
        self.n = 0

    def __iter__(self) -> Iterator[Any]:
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs: Any) -> None:
        return None

    def set_postfix_str(self, s: str | None = None, refresh: bool = True) -> None:
        return None

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        return None

    def write(self, s: str, file: Any | None = None, end: str = "\n") -> None:
        target = file if file is not None else sys.stdout
        print(s, file=target, end=end)

    def close(self) -> None:
        return None

    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

def _default_progress_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "dynamic_ncols": True,
        "mininterval": 0.5,
    }
    merged.update(kwargs)
    return merged

def create_progress(
    *,
    total: int | None = None,
    desc: str | None = None,
    mode: str | None = None,
    iterable: Iterable[Any] | None = None,
    **kwargs: Any,
):
    """Create a tqdm progress bar or a no-op stand-in."""
    if not progress_enabled(mode=mode):
        return NullProgress(iterable=iterable)

    tqdm_kwargs = _default_progress_kwargs(kwargs)
    return _tqdm(iterable=iterable, total=total, desc=desc, **tqdm_kwargs)

def iter_progress(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str | None = None,
    mode: str | None = None,
    **kwargs: Any,
):
    """Wrap an iterable with a live progress bar when enabled."""
    return create_progress(
        iterable=iterable,
        total=total,
        desc=desc,
        mode=mode,
        **kwargs,
    )

@contextmanager
def managed_progress(
    *,
    total: int | None = None,
    desc: str | None = None,
    mode: str | None = None,
    **kwargs: Any,
):
    """Context manager wrapper around :func:`create_progress`."""
    progress = create_progress(total=total, desc=desc, mode=mode, **kwargs)
    try:
        yield progress
    finally:
        progress.close()
