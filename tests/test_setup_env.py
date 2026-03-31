from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "setup_env.py"
SPEC = importlib.util.spec_from_file_location("setup_env_script", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
setup_env = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = setup_env
SPEC.loader.exec_module(setup_env)


def make_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "repo with spaces"
    (project_root / "scripts").mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
    return project_root


def test_is_supported_python() -> None:
    assert setup_env.is_supported_python((3, 10, 0))
    assert setup_env.is_supported_python((3, 12, 1))
    assert not setup_env.is_supported_python((3, 9, 18))


def test_project_root_from_script_requires_pyproject(tmp_path: Path) -> None:
    script_path = tmp_path / "scripts" / "setup_env.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("", encoding="utf-8")

    with pytest.raises(setup_env.SetupError, match="pyproject.toml"):
        setup_env.project_root_from_script(script_path)


def test_should_fallback_from_uv_bootstrap_errors() -> None:
    assert setup_env.should_fallback_from_uv("No virtual environment found", "")
    assert setup_env.should_fallback_from_uv("Failed to inspect Python interpreter", "")
    assert not setup_env.should_fallback_from_uv("Resolution failed because dependencies conflict", "")


def test_venv_paths_resolve_windows_and_posix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    venv_dir = tmp_path / ".venv"

    monkeypatch.setattr(setup_env.os, "name", "nt")
    windows_paths = setup_env.venv_paths(venv_dir)
    assert windows_paths.python == venv_dir / "Scripts" / "python.exe"
    assert windows_paths.pip == venv_dir / "Scripts" / "pip.exe"

    monkeypatch.setattr(setup_env.os, "name", "posix")
    posix_paths = setup_env.venv_paths(venv_dir)
    assert posix_paths.python == venv_dir / "bin" / "python"
    assert posix_paths.pip == venv_dir / "bin" / "pip"


def test_ensure_directories_is_idempotent(tmp_path: Path) -> None:
    project_root = make_project(tmp_path)
    setup_env.ensure_directories(project_root)
    setup_env.ensure_directories(project_root)

    for name in setup_env.REQUIRED_DIRECTORIES:
        assert (project_root / name).is_dir()


def test_bootstrap_environment_falls_back_when_uv_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = make_project(tmp_path)
    script_path = project_root / "scripts" / "setup_env.py"
    script_path.write_text("", encoding="utf-8")
    fallback_python = project_root / ".venv" / "Scripts" / "python.exe"
    calls: list[str] = []

    monkeypatch.setattr(setup_env, "ensure_supported_python", lambda: None)
    monkeypatch.setattr(setup_env, "uv_is_available", lambda: False)
    monkeypatch.setattr(
        setup_env,
        "install_with_pip",
        lambda root: calls.append("pip") or fallback_python,
    )
    monkeypatch.setattr(setup_env.os, "chdir", lambda path: calls.append(f"chdir:{path}"))

    exit_code = setup_env.bootstrap_environment(script_path)

    assert exit_code == 0
    assert "pip" in calls
    for name in setup_env.REQUIRED_DIRECTORIES:
        assert (project_root / name).exists()


def test_bootstrap_environment_uses_uv_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = make_project(tmp_path)
    script_path = project_root / "scripts" / "setup_env.py"
    script_path.write_text("", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(setup_env, "ensure_supported_python", lambda: None)
    monkeypatch.setattr(setup_env, "uv_is_available", lambda: True)
    monkeypatch.setattr(setup_env, "sync_with_uv", lambda root: calls.append("uv") or True)
    monkeypatch.setattr(
        setup_env,
        "install_with_pip",
        lambda root: calls.append("pip") or (project_root / ".venv" / "bin" / "python"),
    )
    monkeypatch.setattr(setup_env.os, "chdir", lambda path: calls.append(f"chdir:{path}"))

    exit_code = setup_env.bootstrap_environment(script_path)

    assert exit_code == 0
    assert "uv" in calls
    assert "pip" not in calls
