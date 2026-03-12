"""
JAX device and precision configuration.
"""

import logging
import jax
from gibbsq.core.config import JAXConfig

log = logging.getLogger(__name__)

def setup_jax(cfg: JAXConfig) -> None:
    """
    Configure JAX runtime based on the provided configuration.
    
    This handles:
    1. Precision (float64 vs float32)
    2. Platform selection (Auto-detection with fallback)
    """
    if not cfg.enabled:
        return

    # 1. Precision Setup
    # CRITICAL: This MUST be done before any JAX operations are executed.
    if cfg.precision == "float64":
        log.info("[JAX] Enabling 64-bit precision (FP64) for high-stability simulation.")
        jax.config.update("jax_enable_x64", True)
    else:
        log.info("[JAX] Using default 32-bit precision (FP32) for performance.")

    # 2. Platform Management
    target_platform = cfg.platform.lower()
    if target_platform == "cuda":
        target_platform = "gpu"

    if target_platform == "auto":
        # Prefer JAX auto-detection, but guard against broken accelerator plugins
        # (e.g., CUDA plugin installed but no supported GPU in the runtime).
        try:
            devices = jax.devices()
            log.info(f"[JAX] Auto-selected platform: {jax.default_backend().upper()} ({len(devices)} devices)")
        except (RuntimeError, ValueError) as e:
            if cfg.fallback_to_cpu:
                log.warning(f"[JAX] Auto platform detection failed: {e}")
                log.warning("[JAX] Restricting runtime to CPU backend.")
                jax.config.update("jax_platforms", "cpu")
                jax.config.update("jax_platform_name", "cpu")
                devices = jax.devices(backend="cpu")
            else:
                raise
    else:
        try:
            if target_platform == "cpu":
                # Skip probing unavailable accelerator plugins in CPU-only environments.
                jax.config.update("jax_platforms", "cpu")
            devices = jax.devices(backend=target_platform)
            jax.config.update("jax_platform_name", target_platform)
            log.info(f"[JAX] Successfully bound to platform: {target_platform.upper()} ({len(devices)} devices)")
        except (RuntimeError, ValueError) as e:
            if cfg.fallback_to_cpu and target_platform != "cpu":
                log.warning(f"[JAX] FAILED to bind to {target_platform.upper()}: {e}")
                log.warning("[JAX] Falling back to CPU as per configuration.")
                jax.config.update("jax_platforms", "cpu")
                jax.config.update("jax_platform_name", "cpu")
            else:
                log.error(f"[JAX] CRITICAL: Target platform {target_platform.upper()} unavailable and fallback disabled.")
                raise e

    # Log device details
    for i, d in enumerate(jax.devices()):
        log.debug(f"  - Device {i}: {d.device_kind}")
