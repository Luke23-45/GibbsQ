from unittest.mock import Mock

import pytest

from gibbsq.core.config import JAXConfig
from gibbsq.utils import device


def test_setup_jax_auto_falls_back_to_cpu_when_device_discovery_fails(monkeypatch):
    cfg = JAXConfig(enabled=True, platform="auto", precision="float32", fallback_to_cpu=True)

    update_mock = Mock()
    monkeypatch.setattr(device.jax.config, "update", update_mock)

    call_state = {"calls": 0}

    def fake_devices(*args, **kwargs):
        call_state["calls"] += 1
        backend = kwargs.get("backend")
        if call_state["calls"] == 1 and backend is None:
            raise RuntimeError("no supported devices found for platform CUDA")
        if backend == "cpu":
            return [Mock(device_kind="cpu")]
        return [Mock(device_kind="cpu")]

    monkeypatch.setattr(device.jax, "devices", fake_devices)
    monkeypatch.setattr(device.jax, "default_backend", lambda: "cpu")

    device.setup_jax(cfg)

    assert call_state["calls"] >= 2
    assert any(args == ("jax_platforms", "cpu") for args, _ in update_mock.call_args_list)
    assert any(args == ("jax_platform_name", "cpu") for args, _ in update_mock.call_args_list)


def test_setup_jax_cpu_sets_platforms_before_device_query(monkeypatch):
    cfg = JAXConfig(enabled=True, platform="cpu", precision="float32", fallback_to_cpu=True)

    update_mock = Mock()
    monkeypatch.setattr(device.jax.config, "update", update_mock)

    devices_mock = Mock(return_value=[Mock(device_kind="cpu")])
    monkeypatch.setattr(device.jax, "devices", devices_mock)

    device.setup_jax(cfg)

    assert update_mock.call_args_list[0].args == ("jax_platforms", "cpu")
    devices_mock.assert_any_call(backend="cpu")


def test_setup_jax_auto_raises_when_fallback_disabled(monkeypatch):
    cfg = JAXConfig(enabled=True, platform="auto", precision="float32", fallback_to_cpu=False)

    monkeypatch.setattr(device.jax, "devices", Mock(side_effect=RuntimeError("broken backend")))

    with pytest.raises(RuntimeError, match="broken backend"):
        device.setup_jax(cfg)
