#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import NamedTuple, Sequence


MIN_PYTHON = (3, 10)
REQUIRED_DIRECTORIES = ("outputs", "logs", "configs", "data")
UV_BOOTSTRAP_ERROR_HINTS = (
    "no virtual environment found",
    "virtual environment",
    "venv",
    "interpreter",
    "python was not found",
    "failed to inspect python",
    "failed to locate python",
    "failed to create",
    "could not create",
)


class SetupError(RuntimeError):
    """Raised when bootstrap cannot complete safely."""


class VenvPaths(NamedTuple):
    python: Path
    pip: Path


def project_root_from_script(script_path: Path) -> Path:
    project_root = script_path.resolve().parent.parent
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        raise SetupError(
            f"Could not find pyproject.toml at {pyproject}. "
            "Run this script from the repository checkout and keep it under scripts/."
        )
    return project_root


def is_supported_python(version_info: Sequence[int]) -> bool:
    return tuple(version_info[:2]) >= MIN_PYTHON


def ensure_supported_python() -> None:
    if not is_supported_python(sys.version_info):
        major, minor = MIN_PYTHON
        raise SetupError(
            f"Python {major}.{minor}+ is required, but this script is running under "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )


def format_command(command: Sequence[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def run_command(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print(f"[setup] Running: {format_command(command)}")
    try:
        return subprocess.run(
            list(command),
            cwd=str(cwd),
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise SetupError(f"Command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        details = render_process_failure(exc)
        raise SetupError(details) from exc


def render_process_failure(exc: subprocess.CalledProcessError) -> str:
    stdout = (exc.stdout or "").strip()
    stderr = (exc.stderr or "").strip()
    lines = [f"Command failed with exit code {exc.returncode}: {format_command(exc.cmd)}"]
    if stderr:
        lines.append(f"stderr:\n{stderr}")
    if stdout:
        lines.append(f"stdout:\n{stdout}")
    return "\n".join(lines)


def uv_is_available() -> bool:
    return shutil.which("uv") is not None


def should_fallback_from_uv(stderr: str, stdout: str) -> bool:
    combined = f"{stderr}\n{stdout}".lower()
    return any(hint in combined for hint in UV_BOOTSTRAP_ERROR_HINTS)


def sync_with_uv(project_root: Path) -> bool:
    print("[setup] Using uv to sync dependencies.")
    try:
        run_command(["uv", "--version"], cwd=project_root)
        run_command(["uv", "sync", "--all-extras"], cwd=project_root)
    except SetupError as exc:
        message = str(exc)
        if should_fallback_from_uv(message, ""):
            print("[setup] uv is available but could not bootstrap this environment.")
            print("[setup] Falling back to a standard library virtualenv + pip install.")
            return False
        raise SetupError(
            "uv was found, but dependency sync failed in a way that should be fixed "
            "instead of masked by a fallback.\n"
            f"{message}"
        ) from exc
    print("[setup] uv sync completed successfully.")
    return True


def active_virtualenv_python() -> Path | None:
    if sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV"):
        return Path(sys.executable)
    return None


def venv_paths(venv_dir: Path) -> VenvPaths:
    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    python_name = "python.exe" if os.name == "nt" else "python"
    pip_name = "pip.exe" if os.name == "nt" else "pip"
    return VenvPaths(python=bin_dir / python_name, pip=bin_dir / pip_name)


def ensure_virtualenv(venv_dir: Path) -> VenvPaths:
    paths = venv_paths(venv_dir)
    pyvenv_cfg = venv_dir / "pyvenv.cfg"
    needs_create = not venv_dir.exists()
    needs_repair = venv_dir.exists() and (not pyvenv_cfg.is_file() or not paths.python.is_file())

    if needs_create:
        print(f"[setup] Creating virtual environment at {venv_dir}.")
    elif needs_repair:
        print(f"[setup] Existing virtual environment at {venv_dir} is incomplete. Rebuilding it.")
    else:
        print(f"[setup] Reusing virtual environment at {venv_dir}.")

    if needs_create or needs_repair:
        builder = venv.EnvBuilder(with_pip=True, clear=needs_repair)
        builder.create(str(venv_dir))

    paths = venv_paths(venv_dir)
    if not paths.python.is_file():
        raise SetupError(f"Virtual environment is missing its Python executable: {paths.python}")
    if not paths.pip.is_file():
        raise SetupError(
            f"Virtual environment is missing pip: {paths.pip}. "
            "Make sure ensurepip is available for this Python installation."
        )
    return paths


def install_with_pip(project_root: Path) -> Path:
    active_python = active_virtualenv_python()
    if active_python is not None:
        python_path = active_python
        print(f"[setup] Using the active virtual environment: {python_path}")
    else:
        env_paths = ensure_virtualenv(project_root / ".venv")
        python_path = env_paths.python

    run_command(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=project_root,
    )
    run_command([str(python_path), "-m", "pip", "install", "-e", ".[dev]"], cwd=project_root)
    print("[setup] Editable install completed successfully.")
    return python_path


def ensure_directories(project_root: Path) -> None:
    print("[setup] Ensuring project directories exist.")
    for directory_name in REQUIRED_DIRECTORIES:
        directory_path = project_root / directory_name
        existed = directory_path.exists()
        directory_path.mkdir(parents=True, exist_ok=True)
        state = "exists" if existed else "created"
        print(f"[setup] {state}: {directory_path}")


def activation_hint(project_root: Path, python_path: Path | None) -> str:
    if python_path is None:
        return "Run commands with uv, for example: uv run python scripts/execution/experiment_runner.py policy"

    venv_dir = project_root / ".venv"
    if python_path == Path(sys.executable):
        return f"Your current virtual environment is ready. Interpreter: {python_path}"

    if python_path.parent.name == "Scripts":
        activate_cmd = f"{venv_dir}\\Scripts\\Activate.ps1"
    else:
        activate_cmd = f"source {venv_dir}/bin/activate"
    return f"Activate the environment with: {activate_cmd}"


def bootstrap_environment(script_path: Path) -> int:
    ensure_supported_python()
    project_root = project_root_from_script(script_path)
    os.chdir(project_root)

    python_path: Path | None = None
    if uv_is_available():
        if not sync_with_uv(project_root):
            python_path = install_with_pip(project_root)
    else:
        print("[setup] uv is not available. Falling back to a standard library virtualenv + pip install.")
        python_path = install_with_pip(project_root)

    ensure_directories(project_root)
    print("[setup] Environment setup is complete.")
    print(f"[setup] {activation_hint(project_root, python_path)}")
    return 0


def main() -> int:
    try:
        return bootstrap_environment(Path(__file__))
    except SetupError as exc:
        print(f"[setup] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
