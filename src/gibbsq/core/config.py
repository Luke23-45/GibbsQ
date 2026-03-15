"""
Structured configuration for the softmax-routed queueing system.

Uses Hydra's ConfigStore to register typed schemas. Each sub-config
is a typed dataclass validated by :func:`validate`, so invalid
configurations are caught before experiment execution.

Derived quantities mirror the proof exactly:

    Λ  = Σ μ_i                             (total capacity)
    ρ  = λ / Λ                             (load factor)
    R  = (λ log N) / α + (λ + Λ) / 2       (drift constant)
    ε  = min((Λ − λ) / N, min_i μ_i)       (contraction rate)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Sequence

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from gibbsq.core import constants

__all__ = [
    "PolicyName",
    "SystemConfig",
    "SSAConfig",
    "DGAConfig",
    "SimulationConfig",
    "PolicyConfig",
    "DriftConfig",
    "ExperimentConfig",
    "hydra_to_config",
    "validate",
    "total_capacity",
    "load_factor",
    "drift_constant_R",
    "drift_rate_epsilon",
    "compact_set_radius",
]


# ──────────────────────────────────────────────────────────────
#  Enum for policy selection  (type-safe, IDE-completable)
# ──────────────────────────────────────────────────────────────

class PolicyName(str, Enum):
    """Supported routing policies."""

    SOFTMAX       = "softmax"
    UNIFORM       = "uniform"
    PROPORTIONAL  = "proportional"
    JSQ           = "jsq"
    POWER_OF_D    = "power_of_d"


# ──────────────────────────────────────────────────────────────
#  Sub-configurations
# ──────────────────────────────────────────────────────────────

@dataclass
class SystemConfig:
    """
    Physical parameters of the queueing system.

    Fields
    ------
    num_servers : int
        Number of parallel servers  N ≥ 1.
    arrival_rate : float
        Poisson arrival rate  λ > 0.
    service_rates : list of float
        Per-server exponential service rates  (μ₁, …, μ_N),  each > 0.
    alpha : float
        Inverse temperature of the softmax routing  α > 0.
    """

    num_servers:   int         = MISSING
    arrival_rate:  float       = MISSING
    service_rates: List[float] = MISSING
    alpha:         float       = MISSING


@dataclass
class SSAConfig:
    """
    Parameters for the Gillespie SSA engine (jax_engine.py / numpy simulate).

    Fields
    ------
    sim_time : float
        Total simulation horizon  T > 0  (continuous time seconds).
    sample_interval : float
        Time between trajectory snapshot samples  Δt > 0.
    max_events : int
        Event budget — hard ceiling on Gillespie events to prevent runaway.
    """

    sim_time:        float = 5000.0
    sample_interval: float = 1.0
    max_events:      int   = 100_000


@dataclass
class DGAConfig:
    """
    Parameters for the Differentiable Gillespie Approximation engine.

    Fields
    ------
    sim_steps : int
        Number of lax.scan iterations per trajectory.
    temperature : float
        Gumbel-Softmax temperature for relaxed event selection.
    """

    sim_steps:   int   = 5000
    temperature: float = 0.5


@dataclass
class SimulationConfig:
    """
    Engine-agnostic simulation parameters plus engine-specific sub-configs.

    Fields
    ------
    num_replications : int
        Number of independent replications  R ≥ 1.
    seed : int
        Base seed for the PRNG.  Replication *r* uses  seed + r.
    burn_in_fraction : float
        Fraction of the trajectory discarded before computing steady-state
        statistics.  Must satisfy  0 ≤ burn_in_fraction < 1.
    ssa : SSAConfig
        Gillespie SSA engine parameters.
    dga : DGAConfig
        Differentiable Gillespie Approximation engine parameters.
    """

    num_replications:    int   = 5
    seed:                int   = 42
    burn_in_fraction:    float = 0.2
    export_trajectories: bool  = False
    ssa:  SSAConfig = field(default_factory=SSAConfig)
    dga:  DGAConfig = field(default_factory=DGAConfig)


@dataclass
class PolicyConfig:
    """
    Routing policy parameters.

    ``name`` must be one of the values in :class:`PolicyName`.
    ``d`` is used only when ``name == "power_of_d"``.
    """

    name: str = PolicyName.SOFTMAX.value
    d:    int = 2


@dataclass
class DriftConfig:
    """
    Parameters for Lyapunov drift verification.

    ``q_max``   controls the grid upper bound  {0, …, q_max}^N.
    ``use_grid`` selects grid evaluation (True) vs trajectory evaluation.
    """

    q_max:    int  = 50
    use_grid: bool = True


@dataclass
class JAXEngineConfig:
    """Hardware acceleration tunables."""
    max_events_safety_multiplier: float = 1.5
    max_events_additive_buffer: int = 1000
    scan_sampling_chunk: int = 4

@dataclass(frozen=True)
class NeuralConfig:
    """Neural routing architecture & preprocessing."""
    hidden_size: int = 64
    preprocessing: str = "log1p"  # Phase 10: Superior for scale-invariance
    capacity_bound: float = constants.NEURAL_LINEAR_CAPACITY_BOUND
    init_type: str = "zero_final" # Standard safety requirement

@dataclass
class VerificationThresholds:
    """Research success boundaries."""
    parity_threshold_percent: float = 25.0
    jacobian_rel_tol: float = 1e-1  # Loosened from 5e-2 due to 50.0 sigmoid steepness
    alpha_significance: float = 0.05
    confidence_interval: float = 0.95

@dataclass
class WandbConfig:
    """
    Weights & Biases integration.
    """

    enabled:  bool = False
    project:  str  = "GibbsQ"
    entity:   str | None = None
    group:    str | None = None
    tags:     List[str] = field(default_factory=list)
    mode:     str = "online"   # online, offline, or disabled
    run_name: str | None = None


@dataclass
class JAXConfig:
    """
    Hardware acceleration via JAX.

    Fields
    ------
    enabled : bool
        Whether to use JAX for simulation (required for massive-N scale).
    platform : str
        Target device platform: "auto", "cpu", "gpu", or "tpu".
    precision : str
        Numerical precision: "float32" (faster) or "float64" (more stable).
    fallback_to_cpu : bool
        If True, fall back to CPU if the target platform is unavailable.
    """

    enabled:         bool = False
    platform:        str  = "auto"
    precision:       str  = "float64"
    fallback_to_cpu: bool = True


@dataclass
class GeneralizationConfig:
    rho_boundary_vals: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.98, 0.99, 0.999])
    scale_vals: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0])
    rho_grid_vals: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.85, 0.95, 0.98])


@dataclass
class StressConfig:
    n_values: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    critical_rhos: List[float] = field(default_factory=lambda: [0.95, 0.98, 0.99, 0.995])


@dataclass
class StabilitySweepConfig:
    alpha_vals: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    rho_vals: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.9, 0.95, 0.99])


@dataclass
class NeuralTrainingConfig:
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    dga_learning_rate: float = 0.5
    curriculum: List[List[int]] = field(default_factory=lambda: [[20, 500], [30, 2000], [50, 5000]])


# ──────────────────────────────────────────────────────────────
#  Top-level experiment config
# ──────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    Root configuration node.  Composed from the four sub-configs
    plus an output directory.
    """

    system:     SystemConfig     = field(default_factory=SystemConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    policy:     PolicyConfig     = field(default_factory=PolicyConfig)
    drift:      DriftConfig      = field(default_factory=DriftConfig)
    wandb:      WandbConfig      = field(default_factory=WandbConfig)
    jax:        JAXConfig        = field(default_factory=JAXConfig)
    jax_engine: JAXEngineConfig  = field(default_factory=JAXEngineConfig)
    neural:     NeuralConfig     = field(default_factory=NeuralConfig)
    verification: VerificationThresholds = field(default_factory=VerificationThresholds)
    generalization: GeneralizationConfig = field(default_factory=GeneralizationConfig)
    stress:         StressConfig         = field(default_factory=StressConfig)
    stability_sweep: StabilitySweepConfig = field(default_factory=StabilitySweepConfig)
    neural_training: NeuralTrainingConfig = field(default_factory=NeuralTrainingConfig)
    output_dir:   str              = "outputs"
    log_dir:      str              = "logs"
    debug:        bool             = False
    train_epochs: int              = 30


# ──────────────────────────────────────────────────────────────
#  Validation  (fail-fast with precise diagnostics)
# ──────────────────────────────────────────────────────────────


def _normalize_service_rates(system_cfg: dict) -> dict:
    """Expand scalar service-rate shorthand to match num_servers when needed."""
    cfg = dict(system_cfg)
    rates = cfg.get("service_rates")
    num_servers = cfg.get("num_servers")
    if isinstance(rates, list) and len(rates) == 1 and isinstance(num_servers, int) and num_servers > 1:
        cfg["service_rates"] = rates * num_servers
    return cfg


def _validate_jax_config(jax_cfg: JAXConfig) -> None:
    """Validate JAX backend config values."""
    valid_platforms = {"auto", "cpu", "gpu", "tpu", "cuda"}
    if jax_cfg.platform.lower() not in valid_platforms:
        raise ValueError(
            f"jax.platform must be one of {sorted(valid_platforms)}, got '{jax_cfg.platform}'"
        )
    valid_precisions = {"float32", "float64"}
    if jax_cfg.precision not in valid_precisions:
        raise ValueError(
            f"jax.precision must be one of {sorted(valid_precisions)}, got '{jax_cfg.precision}'"
        )

def validate(cfg: ExperimentConfig) -> None:
    """
    Validate every constraint on *cfg* and raise ``ValueError`` with a
    precise diagnostic message on the first violation.

    Called at the beginning of every experiment script.
    """
    s = cfg.system

    # ── Structural ────────────────────────────────────────────
    if s.num_servers < 1:
        raise ValueError(f"num_servers must be ≥ 1, got {s.num_servers}")
    if len(s.service_rates) != s.num_servers:
        raise ValueError(
            f"|service_rates| = {len(s.service_rates)} ≠ num_servers = {s.num_servers}"
        )

    # ── Positivity ────────────────────────────────────────────
    if s.arrival_rate <= 0:
        raise ValueError(f"arrival_rate (λ) must be > 0, got {s.arrival_rate}")
    if s.alpha <= 0:
        raise ValueError(f"alpha (α) must be > 0, got {s.alpha}")
    for i, mu_i in enumerate(s.service_rates):
        if mu_i <= 0:
            raise ValueError(f"service_rates[{i}] (μ_{i+1}) must be > 0, got {mu_i}")

    # ── Capacity condition  Λ > λ  ────────────────────────────
    cap = math.fsum(s.service_rates)          # Kahan-accurate sum
    if cap <= s.arrival_rate:
        raise ValueError(
            f"Capacity condition violated: "
            f"Λ = Σμ_i = {cap} must be strictly > λ = {s.arrival_rate}"
        )

    # ── Simulation bounds ─────────────────────────────────────
    sim = cfg.simulation
    if sim.num_replications < 1:
        raise ValueError(f"num_replications must be ≥ 1, got {sim.num_replications}")
    if not (0 <= sim.burn_in_fraction < 1):
        raise ValueError(
            f"burn_in_fraction must be in [0, 1), got {sim.burn_in_fraction}"
        )

    # ── SSA engine bounds ─────────────────────────────────────
    ssa = sim.ssa
    if ssa.sim_time <= 0:
        raise ValueError(f"ssa.sim_time must be > 0, got {ssa.sim_time}")
    if ssa.sample_interval <= 0:
        raise ValueError(f"ssa.sample_interval must be > 0, got {ssa.sample_interval}")
    max_samples = ssa.sim_time / ssa.sample_interval
    if max_samples > 50_000:
        import logging
        logging.getLogger(__name__).warning(
            f"SSA max_samples={max_samples:.0f} is very large "
            f"(sim_time={ssa.sim_time}/sample_interval={ssa.sample_interval}). "
            f"Consider increasing sample_interval.")

    # ── DGA engine bounds ─────────────────────────────────────
    dga = sim.dga
    if dga.sim_steps < 1:
        raise ValueError(f"dga.sim_steps must be ≥ 1, got {dga.sim_steps}")
    if dga.sim_steps > 10_000:
        import logging
        logging.getLogger(__name__).warning(
            f"dga.sim_steps={dga.sim_steps} is very large. "
            f"Consider ≤ 5000 for reasonable runtime.")
    if not (0 < dga.temperature <= 10.0):
        raise ValueError(f"dga.temperature must be in (0, 10], got {dga.temperature}")

    # ── Policy name ───────────────────────────────────────────
    valid_names = {e.value for e in PolicyName}
    if cfg.policy.name not in valid_names:
        raise ValueError(
            f"Unknown policy '{cfg.policy.name}'.  Choose from: {sorted(valid_names)}"
        )

    if cfg.policy.name == PolicyName.POWER_OF_D.value and cfg.policy.d < 1:
        raise ValueError(f"policy.d must be >= 1 for power_of_d, got {cfg.policy.d}")

    # ── Neural architecture bounds ────────────────────────────
    neu = cfg.neural
    if neu.hidden_size < 1:
        raise ValueError(f"neural.hidden_size must be ≥ 1, got {neu.hidden_size}")
    valid_pre = {"none", "log1p", "standardize", "linear_min_max"}
    if neu.preprocessing not in valid_pre:
        raise ValueError(f"Unknown neural.preprocessing '{neu.preprocessing}'. Choose from {valid_pre}")
    if neu.capacity_bound <= 0:
        raise ValueError(f"neural.capacity_bound must be > 0, got {neu.capacity_bound}")

    # ── JAX engine safety bounds ──────────────────────────────
    jen = cfg.jax_engine
    if jen.max_events_safety_multiplier < 1.0:
        raise ValueError(f"jax_engine.max_events_safety_multiplier must be ≥ 1.0, got {jen.max_events_safety_multiplier}")
    if jen.max_events_additive_buffer < 0:
        raise ValueError(f"jax_engine.max_events_additive_buffer must be ≥ 0, got {jen.max_events_additive_buffer}")
    if jen.scan_sampling_chunk < 1:
        raise ValueError(f"jax_engine.scan_sampling_chunk must be ≥ 1, got {jen.scan_sampling_chunk}")

    # ── Verification thresholds ───────────────────────────────
    ver = cfg.verification
    if not (0 < ver.parity_threshold_percent < 100):
        raise ValueError(f"verification.parity_threshold_percent must be in (0, 100), got {ver.parity_threshold_percent}")
    if ver.jacobian_rel_tol <= 0:
        raise ValueError(f"verification.jacobian_rel_tol must be > 0, got {ver.jacobian_rel_tol}")
    if not (0 < ver.alpha_significance < 0.5):
        raise ValueError(f"verification.alpha_significance must be in (0, 0.5), got {ver.alpha_significance}")

    _validate_jax_config(cfg.jax)


# ──────────────────────────────────────────────────────────────
#  Derived quantities  (from the proof, §2)
# ──────────────────────────────────────────────────────────────

def total_capacity(cfg: ExperimentConfig) -> float:
    """Λ = Σ_{i=1}^{N} μ_i"""
    return math.fsum(cfg.system.service_rates)


def load_factor(cfg: ExperimentConfig) -> float:
    """ρ = λ / Λ   ∈ (0, 1)  under the capacity condition."""
    return cfg.system.arrival_rate / total_capacity(cfg)


def drift_constant_R(cfg: ExperimentConfig) -> float:
    """
    R = (λ log N) / α  +  C₁

    where  C₁ = (λ + Λ) / 2.    [proof, Step 4]
    """
    s = cfg.system
    C1 = (s.arrival_rate + total_capacity(cfg)) / 2.0
    return (s.arrival_rate * math.log(s.num_servers)) / s.alpha + C1


def drift_rate_epsilon(cfg: ExperimentConfig) -> float:
    """
    ε = min( (Λ − λ) / N,  min_i μ_i )   > 0    [proof, Step 5]
    """
    s = cfg.system
    cap = total_capacity(cfg)
    return min(
        (cap - s.arrival_rate) / s.num_servers,
        min(s.service_rates),
    )


def compact_set_radius(cfg: ExperimentConfig) -> float:
    """
    |Q|₁  threshold defining the "compact set" C in the Foster criterion.

    Outside C,  LV(Q) ≤ −1.    Radius = ⌈(R + 1) / ε⌉.
    """
    R = drift_constant_R(cfg)
    eps = drift_rate_epsilon(cfg)
    return math.ceil((R + 1.0) / eps)


# ──────────────────────────────────────────────────────────────
#  Hydra conversion
# ──────────────────────────────────────────────────────────────

def _build_simulation_config(sim_dict: dict) -> SimulationConfig:
    """Build SimulationConfig with nested SSA/DGA from a flat or nested dict."""
    sim_dict = dict(sim_dict)  # shallow copy
    ssa_raw = sim_dict.pop("ssa", {})
    dga_raw = sim_dict.pop("dga", {})
    return SimulationConfig(
        ssa=SSAConfig(**ssa_raw),
        dga=DGAConfig(**dga_raw),
        **sim_dict,
    )

def hydra_to_config(raw: DictConfig) -> ExperimentConfig:
    """
    Convert a Hydra ``DictConfig`` (loaded from YAML at runtime) into
    our validated :class:`ExperimentConfig`.

    Uses ``OmegaConf.to_container(resolve=True)`` to interpolate all
    Hydra variable references before constructing the dataclass.
    """
    d: dict = OmegaConf.to_container(raw, resolve=True)  # type: ignore[assignment]
    system_cfg = _normalize_service_rates(d.get("system", {}))
    return ExperimentConfig(
        system=SystemConfig(**system_cfg),
        simulation=_build_simulation_config(d.get("simulation", {})),
        policy=PolicyConfig(**d.get("policy", {})),
        drift=DriftConfig(**d.get("drift", {})),
        wandb=WandbConfig(**d.get("wandb", {})),
        jax=JAXConfig(**d.get("jax", {})),
        jax_engine=JAXEngineConfig(**d.get("jax_engine", {})),
        neural=NeuralConfig(**d.get("neural", {})),
        verification=VerificationThresholds(**d.get("verification", {})),
        generalization=GeneralizationConfig(**d.get("generalization", {})),
        stress=StressConfig(**d.get("stress", {})),
        stability_sweep=StabilitySweepConfig(**d.get("stability_sweep", {})),
        neural_training=NeuralTrainingConfig(**d.get("neural_training", {})),
        output_dir=d.get("output_dir", "outputs"),
        log_dir=d.get("log_dir", "logs"),
        debug=d.get("debug", False),
        train_epochs=d.get("train_epochs", 30),
    )


# ──────────────────────────────────────────────────────────────
#  Register with Hydra's ConfigStore
# ──────────────────────────────────────────────────────────────

cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)
