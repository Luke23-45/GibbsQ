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
    "critical_load_required_sim_time",
    "critical_load_sim_time",
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
    JSSQ          = "jssq"
    UAS           = "uas"


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
    """

    sim_time:        float = 5000.0
    sample_interval: float = 1.0


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
    # 16 gives P(miss | a0=1.0, interval=1.0) = exp(-16) < 2e-7 per event —
    # covers the worst-case quick validation configs while adding negligible no-op iterations.
    scan_sampling_chunk: int = 16

@dataclass(frozen=True)
class NeuralConfig:
    """Neural routing architecture & preprocessing."""
    hidden_size: int = 64
    preprocessing: str = "log1p"  # Phase 10: Superior for scale-invariance
    capacity_bound: float = constants.NEURAL_LINEAR_CAPACITY_BOUND
    init_type: str = "zero_final" # Standard safety requirement
    use_rho: bool = True          # PI-V4: Append rho to state features
    use_service_rates: bool = True # SG#13 FIX: Restore heterogeneity awareness
    rho_input_scale: float = 10.0  # PI-V4.1: Increased scale for better feature saliency
    entropy_bonus: float = 0.01    # PI-V4.1: Prevent premature determinism
    entropy_final: float = 0.001
    clip_global_norm: float = 1.0  # PI-V4.1: Prevent loss explosions
    actor_lr: float = 3e-4         # PI-V4.1: Separate LRs for stability
    critic_lr: float = 1e-3
    lr_decay_rate: float = 0.9
    weight_decay: float = 1e-4     # L2 regularization for AdamW

@dataclass
class VerificationThresholds:
    """Research success boundaries."""
    parity_threshold_percent: float = 25.0
    # SG#3 FIX: Reverted dangerous 10% tolerance for gradient checks.
    # With the softer DGA steepness (Constants.DGA_INDICATOR_STEEPNESS = 5.0),
    # the finite-difference check passes reliably at the mathematically sound 5% threshold.
    jacobian_rel_tol: float = 0.05
    stationarity_threshold: float = 1.0
    alpha_significance: float = 0.05
    confidence_interval: float = 0.95
    parity_z_score: float = 1.96  # Z-score for parity margins
    # Gradient Analysis Constants
    gradient_check_chunk_size: int = 1500
    gradient_check_max_steps: int = 300
    gradient_check_n_test: int = 50
    gradient_check_hidden_size: int | None = None
    gradient_check_sim_time: float = 15.0
    gradient_check_n_samples: int = 15000
    gradient_check_epsilon: float = 0.05
    gradient_check_cosine_threshold: float = 0.9
    gradient_check_error_threshold: float = 0.40
    gradient_shake_scale: float = 0.10

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
    rho_boundary_vals: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.98, 0.99])
    scale_vals: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0])
    rho_grid_vals: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.85, 0.95, 0.98])


@dataclass
class StressConfig:
    n_values: list[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    critical_rhos: list[float] = field(default_factory=lambda: [0.95, 0.98, 0.99])
    mu_het: list[float] = field(default_factory=lambda: [100.0, 1.0, 1.0, 1.0])
    
    # Phase 3 Hardcode Extractions
    sample_interval: float = 1.0
    massive_n_rho: float = 0.8
    massive_n_sim_time: float = 500.0
    critical_load_n: int = 10
    critical_load_base_rho: float = 0.8
    critical_load_max_sim_time: float = 100000.0
    heterogeneity_rho: float = 0.5
    heterogeneity_sim_time: float = 1000.0


@dataclass
class StabilitySweepConfig:
    # SOTA defaults from Phase 1 verification (ledger/01_sota_targets.md)
    # Alpha sweep: Full temperature sensitivity [0.01, 100]
    # Rho sweep: Heavy-traffic regime [0.80, 0.98]
    alpha_vals: List[float] = field(default_factory=lambda: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
    rho_vals: List[float] = field(default_factory=lambda: [0.80, 0.85, 0.90, 0.95, 0.98])


@dataclass
class NeuralTrainingConfig:
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    dga_learning_rate: float = 0.5
    min_temperature: float = 0.1   # SG-1 FIX: was in YAML but missing from dataclass
    gamma: float = 0.99            # HP-2 FIX: avoid hardcoded discount
    gae_lambda: float = 0.95       # PATCH-P1: GAE λ for bias-variance tradeoff (Schulman 2015)
    curriculum: List[List[int]] = field(default_factory=lambda: [[20, 500], [30, 2000], [50, 5000]])
    eval_batches: int = 5          # Extracted from train_reinforce.py
    eval_trajs_per_batch: int = 10 # Extracted from train_reinforce.py
    bc_num_steps: int = 1000       # Extracted from pretrain_bc.py
    bc_lr: float = 0.002
    bc_label_smoothing: float = 0.1
    perf_index_min_denom: float = 0.5
    perf_index_jsq_margin: float = 0.05
    shake_scale: float = 0.01
    checkpoint_freq: int = 25
    squash_scale: float = 100.0
    squash_threshold: float = 500.0



@dataclass
class ReinforceConfig:
    """
    Configuration for REINFORCE policy gradient training.
    
    This is the corrected training pipeline that uses true Gillespie SSA
    instead of the broken DGA surrogate.
    
    Fields
    ------
    learning_rate : float
        AdamW learning rate for policy network.
    weight_decay : float
        L2 regularization coefficient.
    entropy_bonus : float
        Coefficient for entropy regularization (encourages exploration).
    batch_size : int
        Number of parallel trajectories per gradient step.
    n_epochs : int
        Total training epochs.
    sim_time : float
        SSA simulation horizon per trajectory.
    """
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    entropy_bonus: float = 0.01
    batch_size: int = 8
    n_epochs: int = 100
    sim_time: float = 5000.0


@dataclass
class DomainRandomizationPhase:
    """Single phase of domain randomization curriculum."""
    # SOTA: Start from higher load for critical regime training
    rho_min: float = 0.60
    rho_max: float = 0.85
    epochs: int = 20
    horizon: int = 1000  # INCREASED: Longer horizons for stability


@dataclass
class DomainRandomizationConfig:
    """
    Configuration for domain randomization training.

    Domain randomization exposes the neural network to diverse load
    conditions, preventing the curriculum failure where training on
    low-load regimes (ρ < 0.4) produces policies that cannot handle
    critical load conditions.

    Fields
    ------
    enabled : bool
        Whether to use domain randomization.
    phases : list of DomainRandomizationPhase
        Curriculum phases with increasing load ranges.
    """
    enabled: bool = True
    rho_min: float = 0.4
    rho_max: float = 0.95
    phases: List[DomainRandomizationPhase] = field(
        default_factory=lambda: [
            DomainRandomizationPhase(rho_min=0.45, rho_max=0.70, epochs=20, horizon=500),
            DomainRandomizationPhase(rho_min=0.50, rho_max=0.85, epochs=30, horizon=2000),
            DomainRandomizationPhase(rho_min=0.60, rho_max=0.95, epochs=50, horizon=5000),
        ]
    )



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
    domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    neural_training: NeuralTrainingConfig = field(default_factory=NeuralTrainingConfig)
    output_dir:   str              = "outputs"
    log_dir:      str              = "logs"
    train_epochs: int              = 30
    batch_size:   int              = 16


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


def _validate_rho_list(name: str, values: Sequence[float]) -> None:
    """Validate that every configured load factor is strictly within (0, 1)."""
    for i, rho in enumerate(values):
        if not (0.0 < float(rho) < 1.0):
            raise ValueError(f"{name}[{i}] must be in (0, 1), got {rho}")


def _validate_positive_list(name: str, values: Sequence[float]) -> None:
    """Validate that every configured scalar is strictly positive."""
    for i, x in enumerate(values):
        if not (float(x) > 0.0):
            raise ValueError(f"{name}[{i}] must be > 0, got {x}")


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
    
    # ── New neural configuration validation (PI-V4) ────────────────────────────
    if neu.use_rho and not isinstance(neu.use_rho, bool):
        raise ValueError(f"neural.use_rho must be boolean, got {type(neu.use_rho)}")
    if neu.rho_input_scale <= 0:
        raise ValueError(f"neural.rho_input_scale must be > 0, got {neu.rho_input_scale}")
    if neu.rho_input_scale > 100.0:
        raise ValueError(f"neural.rho_input_scale must be <= 100.0, got {neu.rho_input_scale}")
    if neu.entropy_bonus < 0:
        raise ValueError(f"neural.entropy_bonus must be ≥ 0, got {neu.entropy_bonus}")
    if neu.entropy_bonus > 1.0:
        raise ValueError(f"neural.entropy_bonus must be <= 1.0, got {neu.entropy_bonus}")
    if neu.clip_global_norm <= 0:
        raise ValueError(f"neural.clip_global_norm must be > 0, got {neu.clip_global_norm}")
    if neu.clip_global_norm > 10.0:
        raise ValueError(f"neural.clip_global_norm must be <= 10.0, got {neu.clip_global_norm}")
    if neu.actor_lr <= 0 or neu.critic_lr <= 0:
        raise ValueError(f"neural.actor_lr and neural.critic_lr must be > 0, got {neu.actor_lr}, {neu.critic_lr}")
    if neu.actor_lr > 1e-1 or neu.critic_lr > 1e-1:
        raise ValueError(f"neural.learning rates must be <= 1e-1, got {neu.actor_lr}, {neu.critic_lr}")
    if neu.actor_lr < 1e-6 or neu.critic_lr < 1e-6:
        raise ValueError(f"neural.learning rates must be >= 1e-6, got {neu.actor_lr}, {neu.critic_lr}")
    if neu.weight_decay < 0:
        raise ValueError(f"neural.weight_decay must be >= 0, got {neu.weight_decay}")
    if neu.weight_decay > 1.0:
        raise ValueError(f"neural.weight_decay must be <= 1.0, got {neu.weight_decay}")
    
    # PATCH: Validate gamma (discount factor) is in valid range
    ntc = cfg.neural_training
    if not (0 <= ntc.gamma <= 1):
        raise ValueError(f"neural_training.gamma must be in [0, 1], got {ntc.gamma}")

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
    if not (0 < ver.stationarity_threshold <= 1.0):
        raise ValueError(f"verification.stationarity_threshold must be in (0, 1.0], got {ver.stationarity_threshold}")
    if not (0.5 < ver.confidence_interval < 1.0):
        raise ValueError(f"verification.confidence_interval must be in (0.5, 1.0), got {ver.confidence_interval}")

    # ── Domain Randomization ──────────────────────────────────
    dr = cfg.domain_randomization
    if dr.enabled:
        if dr.phases:
            for i, phase in enumerate(dr.phases):
                if not (0 < phase.rho_min < phase.rho_max < 1.0):
                    raise ValueError(f"DR phase {i} has invalid rho range [{phase.rho_min}, {phase.rho_max}]")
                if phase.epochs < 1:
                    raise ValueError(f"DR phase {i} must have epochs ≥ 1, got {phase.epochs}")
        else:
            # SG#1 PATCH: Validate top-level rho bounds when phases are absent
            if not (0 < dr.rho_min < dr.rho_max < 1.0):
                raise ValueError(
                    f"domain_randomization has invalid rho range "
                    f"[{dr.rho_min}, {dr.rho_max}]; need 0 < rho_min < rho_max < 1"
                )

    # ── Sweep/generalization/stress domain guards ─────────────
    # These values are transformed into λ = ρ·Λ in experiment loops,
    # so invalid ρ would bypass the strict capacity condition (Λ > λ).
    _validate_rho_list("stability_sweep.rho_vals", cfg.stability_sweep.rho_vals)
    _validate_positive_list("stability_sweep.alpha_vals", cfg.stability_sweep.alpha_vals)
    _validate_rho_list("generalization.rho_grid_vals", cfg.generalization.rho_grid_vals)
    _validate_rho_list("generalization.rho_boundary_vals", cfg.generalization.rho_boundary_vals)
    _validate_rho_list("stress.critical_rhos", cfg.stress.critical_rhos)
    if not (0.0 < float(cfg.stress.massive_n_rho) < 1.0):
        raise ValueError(f"stress.massive_n_rho must be in (0, 1), got {cfg.stress.massive_n_rho}")
    if not (0.0 < float(cfg.stress.heterogeneity_rho) < 1.0):
        raise ValueError(f"stress.heterogeneity_rho must be in (0, 1), got {cfg.stress.heterogeneity_rho}")

    for name, rho_values in (
        ("generalization.rho_boundary_vals", cfg.generalization.rho_boundary_vals),
        ("stress.critical_rhos", cfg.stress.critical_rhos),
    ):
        for i, rho in enumerate(rho_values):
            required_sim_time = critical_load_required_sim_time(
                base_sim_time=cfg.simulation.ssa.sim_time,
                rho=float(rho),
                base_rho=float(cfg.stress.critical_load_base_rho),
            )
            if required_sim_time > float(cfg.stress.critical_load_max_sim_time):
                raise ValueError(
                    f"{name}[{i}]={rho} requires critical-load sim_time "
                    f"{required_sim_time:.0f}, which exceeds "
                    f"stress.critical_load_max_sim_time={cfg.stress.critical_load_max_sim_time:.0f}. "
                    "Raise the cap or lower the rho grid."
                )

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

    Standard:    C₁ = (λ + Λ) / 2
    Archimedean: C₁ = λ/(2·min_μ) + N/2
    """
    s = cfg.system
    if cfg.policy.name == PolicyName.UAS.value:
        min_mu = min(s.service_rates)
        C1 = (s.arrival_rate / (2.0 * min_mu)) + (s.num_servers / 2.0)
    else:
        C1 = (s.arrival_rate + total_capacity(cfg)) / 2.0
    return (s.arrival_rate * math.log(s.num_servers)) / s.alpha + C1


def drift_rate_epsilon(cfg: ExperimentConfig) -> float:
    """
    ε = min( (Λ − λ) / N,  min_i μ_i )   > 0    [proof, Step 5]

    For UAS, the effective service rate in the weighted space is 1.0.
    """
    s = cfg.system
    cap = total_capacity(cfg)
    if cfg.policy.name == PolicyName.UAS.value:
        return (cap - s.arrival_rate) / cap  # Roughly matching service-rate of 1.0
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


def critical_load_required_sim_time(
    *,
    base_sim_time: float,
    rho: float,
    base_rho: float,
) -> float:
    """Return the uncapped simulation horizon required by the critical-load scaling rule."""
    return base_sim_time * max(1.0, ((1.0 - base_rho) / max(1.0 - rho, 1e-6)))


def critical_load_sim_time(cfg: ExperimentConfig, rho: float) -> float:
    """Return a fail-closed critical-load horizon derived from the active config."""
    required = critical_load_required_sim_time(
        base_sim_time=cfg.simulation.ssa.sim_time,
        rho=float(rho),
        base_rho=float(cfg.stress.critical_load_base_rho),
    )
    cap = float(cfg.stress.critical_load_max_sim_time)
    if required > cap:
        raise ValueError(
            f"rho={rho} requires critical-load sim_time {required:.0f}, "
            f"which exceeds stress.critical_load_max_sim_time={cap:.0f}. "
            "Refusing to truncate the horizon because that biases near-critical results."
        )
    return required


# ──────────────────────────────────────────────────────────────
#  Hydra conversion
# ──────────────────────────────────────────────────────────────

def _build_simulation_config(sim_dict: dict) -> SimulationConfig:
    """Build SimulationConfig with nested SSA/DGA from a flat or nested dict."""
    sim_dict = dict(sim_dict)  # shallow copy
    ssa_raw = sim_dict.pop("ssa", {})
    dga_raw = sim_dict.pop("dga", {})
    # SG-1 FIX: max_events was removed from SSAConfig; pop it from any YAML that
    # still carries it so SSAConfig(**ssa_raw) does not raise TypeError on upgrade.
    # SG-6 PATCH: Emit deprecation warning instead of silently dropping.
    if "max_events" in ssa_raw:
        import logging as _lg
        _lg.getLogger(__name__).warning(
            "DEPRECATED: simulation.ssa.max_events in YAML is ignored. "
            "The JAX and NumPy engines compute max_events dynamically from "
            "(arrival_rate + sum(mu)) * sim_time * 1.5 + 1000. "
            "Remove max_events from your config to suppress this warning."
        )
        ssa_raw.pop("max_events")
    return SimulationConfig(
        ssa=SSAConfig(**ssa_raw),
        dga=DGAConfig(**dga_raw),
        **sim_dict,
    )

def _build_dr_config(dr_dict: dict) -> DomainRandomizationConfig:
    """Build DomainRandomizationConfig from YAML dict.

    SG#1 PATCH: When 'phases' is absent from the YAML, explicitly set
    phases=[] so the dataclass default curriculum is NOT injected.
    This ensures the top-level rho_min/rho_max are honored by the
    training loop.  When 'phases' IS present, parse and pass them
    through (existing behavior).
    """
    dr_dict = dict(dr_dict)  # shallow copy
    phases_raw = dr_dict.pop("phases", None)
    if phases_raw is not None:
        phases = [DomainRandomizationPhase(**p) for p in phases_raw]
        return DomainRandomizationConfig(**dr_dict, phases=phases)
    # SG#1 PATCH: Pass phases=[] to prevent dataclass default injection
    return DomainRandomizationConfig(**dr_dict, phases=[])


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
        verification=VerificationThresholds(**d.get("verification", {})),  # type: ignore[arg-type]
        generalization=GeneralizationConfig(**d.get("generalization", {})),
        stress=StressConfig(**d.get("stress", {})),  # type: ignore[arg-type]
        stability_sweep=StabilitySweepConfig(**d.get("stability_sweep", {})),
        domain_randomization=_build_dr_config(d.get("domain_randomization", {})),  # SG-1 FIX
        neural_training=NeuralTrainingConfig(**d.get("neural_training", {})),
        output_dir=d.get("output_dir", "outputs"),
        log_dir=d.get("log_dir", "logs"),
        train_epochs=d.get("train_epochs", 30),
        batch_size=d.get("batch_size", 16),
    )


# ──────────────────────────────────────────────────────────────
#  Register with Hydra's ConfigStore
# ──────────────────────────────────────────────────────────────

cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)
