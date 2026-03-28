import io

import pytest

from gibbsq.utils import progress as progress_mod


class _FakeStream:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


def test_progress_enabled_auto_uses_tty(monkeypatch):
    monkeypatch.setattr(progress_mod, "_tqdm", object())
    monkeypatch.delenv("CI", raising=False)
    assert progress_mod.progress_enabled(mode="auto", stream=_FakeStream(True)) is True
    assert progress_mod.progress_enabled(mode="auto", stream=_FakeStream(False)) is False


def test_progress_enabled_respects_ci(monkeypatch):
    monkeypatch.setattr(progress_mod, "_tqdm", object())
    monkeypatch.setenv("CI", "1")
    assert progress_mod.progress_enabled(mode="auto", stream=_FakeStream(True)) is False


def test_progress_enabled_on_requires_tqdm(monkeypatch):
    monkeypatch.setattr(progress_mod, "_tqdm", None)
    assert progress_mod.progress_enabled(mode="on", stream=_FakeStream(True)) is False


def test_create_progress_returns_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(progress_mod, "_tqdm", object())
    progress = progress_mod.create_progress(total=3, desc="x", mode="off")
    assert isinstance(progress, progress_mod.NullProgress)


def test_iter_progress_wraps_with_tqdm(monkeypatch):
    calls = {}

    def fake_tqdm(*, iterable=None, total=None, desc=None, **kwargs):
        calls["iterable"] = list(iterable)
        calls["total"] = total
        calls["desc"] = desc
        calls["kwargs"] = kwargs
        return calls["iterable"]

    monkeypatch.setattr(progress_mod, "_tqdm", fake_tqdm)
    wrapped = progress_mod.iter_progress([1, 2], total=2, desc="demo", mode="on")
    assert list(wrapped) == [1, 2]
    assert calls["total"] == 2
    assert calls["desc"] == "demo"


def test_null_progress_write_emits_text(capsys):
    progress = progress_mod.NullProgress()
    progress.write("hello")
    captured = capsys.readouterr()
    assert captured.out == "hello\n"


def test_configure_progress_mode_sets_env(monkeypatch):
    env = {}
    assert progress_mod.configure_progress_mode("on", env=env) == "on"
    assert env[progress_mod.PROGRESS_ENV_VAR] == "on"
