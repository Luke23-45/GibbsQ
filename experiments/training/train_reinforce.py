"""
REINFORCE Policy Gradient Training for N-GibbsQ.

This module implements the corrected training pipeline using REINFORCE
policy gradient on true Gillespie SSA simulations, replacing the broken
DGA-based training that violated Jensen's Inequality and erased the
hard zero-boundary condition.

The key insight: REINFORCE computes gradients through the score function
estimator (likelihood ratio method), which preserves the discrete nature
of the SSA forward simulation. Only the backward pass requires
differentiability, operating on log-likelihoods of routing decisions.

References:
    - Williams, R. J. (1992). Simple statistical gradient-following
      algorithms for connectionist reinforcement learning.
    - Schulman, J. et al. (2017). Gradient Estimation Using Stochastic
      Computation Graphs.
"""

import json
import logging
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from jaxtyping import PRNGKeyArray, Array, Float
from jax.flatten_util import ravel_pytree
from omegaconf import DictConfig

from gibbsq.analysis.plot_profiles import ExperimentPlotContext
from gibbsq.core.config import ExperimentConfig, NeuralConfig, load_experiment_config
from gibbsq.core.neural_policies import NeuralRouter, ValueNetwork
from gibbsq.core.pretraining import extract_bc_data_config
from gibbsq.core.policy_distribution import compute_numpy_policy_probs
from gibbsq.core.reinforce_objective import compute_action_interval_returns_from_trajectory_numpy
from gibbsq.core.features import look_ahead_potential
from gibbsq.core.policies import JSQRouting, JSSQRouting
from gibbsq.core import constants
# Local simulate not needed if defined/used via engines.numpy_engine directly if ever needed,
# but here we use the local collect_trajectory_ssa.
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.model_io import (
    BC_POINTER,
    get_bc_metadata_path,
    save_model_pointer,
    validate_bc_reuse_metadata,
    validate_neural_model_shape,
)
from gibbsq.utils.progress import create_progress, iter_progress
from gibbsq.utils.run_artifacts import artifacts_dir, figure_path, metrics_path

log = logging.getLogger(__name__)



class TrajectoryResult(NamedTuple):
    """Result from a single SSA trajectory with routing decisions."""
    log_probs: list[float]  # log π_θ(a_t | s_t) for each routing decision
    states: list[np.ndarray]  # Queue state BEFORE each decision
    actions: list[int]  # Server index chosen for each arrival
    total_integrated_queue: float  # Integrated queue length over time (True Reward)
    arrival_count: int
    departure_count: int
    jump_times: list[float]  # Time of each SSA event
    action_step_indices: list[int]  # Indices where routing decisions were made
    all_states: list[np.ndarray]  # All states (not just at decision points)
    rho: float = 0.4  # Load factor used during trajectory collection


def try_load_bc_actor_warm_start(
    policy_net: NeuralRouter,
    *,
    cfg: ExperimentConfig,
    bc_data_config: dict | None,
    project_root: Path,
    output_root: Path,
) -> tuple[NeuralRouter, dict[str, str | bool | None]]:
    """Load BC actor warm-start weights when a compatible pointer exists."""
    pointer_path = output_root / BC_POINTER
    metadata: dict[str, str | bool | None] = {
        "source": "expert_bootstrap",
        "pointer_path": str(pointer_path),
        "model_path": None,
        "loaded": False,
        "reason": "bc_pointer_missing",
    }
    if not pointer_path.exists():
        return policy_net, metadata

    try:
        raw_pointer = pointer_path.read_text(encoding="utf-8").strip()
        if not raw_pointer:
            metadata["reason"] = "bc_pointer_empty"
            return policy_net, metadata
        model_path = Path(raw_pointer)
        if not model_path.is_absolute():
            model_path = project_root / model_path
        metadata["model_path"] = str(model_path)
        if not model_path.exists():
            metadata["reason"] = "bc_model_missing"
            return policy_net, metadata

        compatibility = validate_bc_reuse_metadata(
            model_path,
            cfg=cfg,
            bc_data_config=bc_data_config,
        )
        candidate = eqx.tree_deserialise_leaves(model_path, policy_net)
        validate_neural_model_shape(candidate, cfg.neural, cfg.system.num_servers)
        metadata.update(
            {
                "source": "bc_pointer",
                "loaded": True,
                "reason": "loaded",
                "fingerprint": str(compatibility["fingerprint"]),
                "metadata_path": str(get_bc_metadata_path(model_path)),
            }
        )
        log.info("Reusing BC warm-start actor weights from %s", model_path)
        return candidate, metadata
    except Exception as exc:
        metadata["reason"] = f"bc_load_failed:{exc}"
        log.warning(
            "BC warm-start reuse failed; falling back to expert bootstrap. Reason: %s",
            exc,
        )
        return policy_net, metadata


def compute_causal_returns_to_go(
    states: list[np.ndarray],
    jump_times: list[float],
    action_step_indices: list[int],
    sim_time: float,
    gamma: float = 0.99
) -> np.ndarray:
    """Computes Discounted Causal Returns, stripping non-stationary time dependence."""
    return compute_action_interval_returns_from_trajectory_numpy(
        states=states,
        jump_times=jump_times,
        action_step_indices=action_step_indices,
        sim_time=sim_time,
        gamma=gamma,
    )


def compute_gae(
    states: list[np.ndarray],
    jump_times: list[float],
    action_step_indices: list[int],
    sim_time: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    value_net=None,
    service_rates: np.ndarray = None,
    rho: float = 0.5,
    jsq_limit: float = 1.0,
    random_limit: float = 10.0,
    denom: float = 5.0,
    squash_scale: float = 100.0,
    squash_threshold: float = 100.0,
    decision_states: list[np.ndarray] | None = None,
):
    """
    Compute Generalized Advantage Estimation (GAE) for CTMC trajectories.

    Implements Schulman et al. (2015) GAE for continuous-time queueing.

    GAE balances bias-variance tradeoff in advantage estimation:
    - λ=0: Low variance, high bias (TD(0))
    - λ=1: High variance, low bias (Monte Carlo returns)
    - λ=0.95: Optimal tradeoff (Schulman recommendation)

    For CTMC, we compute TD residuals using continuous-time discounting:
        δ_t = r_t + γ^(Δt) * V(s_{t+1}) - V(s_t)

    Then GAE is the exponentially-weighted sum:
        A_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}

    Parameters
    ----------
    states : list of np.ndarray
        Post-event states aligned with ``jump_times``. These states determine
        the queue integral over each inter-event interval.
    jump_times : list of float
        Time of each SSA event.
    action_step_indices : list of int
        Indices where routing decisions were made.
    sim_time : float
        Total simulation horizon.
    gamma : float
        Discount factor (default 0.99).
    gae_lambda : float
        GAE λ parameter (default 0.95).
    value_net : ValueNetwork
        Critic network for V(s) estimation.
    service_rates : np.ndarray
        Service rates for sojourn-time feature computation.
    decision_states : list of np.ndarray, optional
        Pre-decision states aligned with routing actions. Required when a
        critic is provided so value targets match the actor input contract.

    Returns
    -------
    np.ndarray
        GAE advantages for each action.
    """
    if not states or not action_step_indices or value_net is None:
        # Fallback to returns-to-go if no critic
        return compute_causal_returns_to_go(states, jump_times, action_step_indices, sim_time, gamma)

    if decision_states is None:
        raise ValueError("decision_states is required when compute_gae uses a critic.")
    if len(states) != len(jump_times):
        raise ValueError("states and jump_times must have the same length.")
    if len(decision_states) != len(action_step_indices):
        raise ValueError("decision_states and action_step_indices must have the same length.")
    if any(idx < 0 or idx >= len(states) for idx in action_step_indices):
        raise ValueError("action_step_indices must refer to valid post-event states.")
    if list(action_step_indices) != sorted(action_step_indices):
        raise ValueError("action_step_indices must be sorted in ascending order.")

    # 1. Compute event-level interval costs from the post-event state that
    # governs the queue between consecutive jumps.
    q_totals = np.array([np.sum(s) for s in states], dtype=np.float64)
    dt = np.diff(np.array(jump_times, dtype=np.float64), append=sim_time)

    # 2. Convert event-level queue integrals into action-interval rewards.
    # The critic is trained on pre-decision states, so Bellman targets must be
    # defined on arrival decision epochs rather than post-arrival event states.
    G_idx = 100.0 * (random_limit - q_totals) / denom
    action_rewards = np.zeros(len(action_step_indices), dtype=np.float64)
    action_dt = np.zeros(len(action_step_indices), dtype=np.float64)

    for i, action_idx in enumerate(action_step_indices):
        next_action_idx = action_step_indices[i + 1] if i + 1 < len(action_step_indices) else len(states)
        elapsed = 0.0
        reward = 0.0
        duration = 0.0
        for t in range(action_idx, next_action_idx):
            reward += (gamma ** elapsed) * (1.0 - (gamma ** dt[t])) * G_idx[t]
            elapsed += dt[t]
            duration += dt[t]
        action_rewards[i] = reward
        action_dt[i] = duration

    # 3. Compute V(s) on decision states plus a terminal bootstrap value.
    import jax.numpy as jnp

    decision_states_arr = np.array(decision_states)
    decision_v_preds = np.array(
        jax.vmap(value_net, in_axes=(0, None, None))(
            jnp.asarray(decision_states_arr),
            jnp.asarray(service_rates),
            rho,
        )
    )
    terminal_v_pred = float(
        value_net(
            jnp.asarray(states[-1]),
            jnp.asarray(service_rates),
            rho,
        )
    )

    # 4. Compute TD residuals on action intervals.
    td_residuals = np.zeros(len(action_step_indices), dtype=np.float64)
    for i in range(len(action_step_indices)):
        next_v = decision_v_preds[i + 1] if i + 1 < len(action_step_indices) else terminal_v_pred
        td_residuals[i] = (
            action_rewards[i]
            + (gamma ** action_dt[i]) * next_v
            - decision_v_preds[i]
        )

    # 5. Compute action-level GAE: A_k = Σ_l (γλ)^{ΔT_{k:k+l}} δ_{k+l}
    advantages = np.zeros(len(action_step_indices))

    for i in range(len(action_step_indices)):
        gae = 0.0
        decay = 1.0
        import math
        beta = -math.log(max(gamma * gae_lambda, 1e-10))
        for t in range(i, len(td_residuals)):
            gae += decay * td_residuals[t]
            decay *= math.exp(-beta * action_dt[t])

        advantages[i] = gae

    # 6. The critic target must align with the pre-decision value contract.
    returns = advantages + decision_v_preds

    return advantages, returns


def compute_performance_index(
    mean_queue: float,
    jsq_limit: float,
    random_limit: float,
    random_empirical: float,
    cfg: ExperimentConfig
) -> float:
    """Computes Performance Index where 0% = Random and 100% = JSQ."""
    fixed_random = max(jsq_limit * 1.01, random_limit, random_empirical)
    denom = max(cfg.neural_training.perf_index_min_denom, jsq_limit * cfg.neural_training.perf_index_jsq_margin, fixed_random - jsq_limit)
    perf_index = 100.0 * (fixed_random - mean_queue) / denom
    return perf_index


def collect_trajectory_ssa(
    policy_net: NeuralRouter,
    num_servers: int,
    arrival_rate: float,
    service_rates: np.ndarray,
    sim_time: float,
    rng: np.random.Generator,
    use_jsq: bool = False,
    rho: float | None = None,
    deterministic: bool = False,
) -> TrajectoryResult:
    """
    Run one Gillespie SSA trajectory, recording routing decisions and queue evolution.

    This is the core REINFORCE trajectory collection. Unlike DGA, the forward
    simulation runs as the true CTMC—no smoothing, no relaxation.

    Parameters
    ----------
    policy_net : NeuralRouter
        Neural routing policy that maps sojourn-time features to routing logits.
    num_servers : int
        Number of servers N.
    arrival_rate : float
        Arrival rate λ.
    service_rates : np.ndarray
        Service rates μ_i for each server.
    sim_time : float
        Simulation horizon T.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    TrajectoryResult
        Contains log_probs, states, total_queue, and event counts.
    """
    N = num_servers
    lam = float(arrival_rate)
    mu = np.asarray(service_rates, dtype=np.float64)

    Q = np.zeros(N, dtype=np.int64)
    t = 0.0

    log_probs = []
    states = []  # States at decision points (arrivals)
    actions = []
    total_queue = 0.0
    arrival_count = 0
    departure_count = 0

    jump_times = []  # Time of each SSA event
    action_step_indices = []  # Indices where routing decisions were made
    all_states = []  # All states (not just at decision points)
    step_counter = 0  # Global step counter for indexing

    rates = np.empty(2 * N, dtype=np.float64)
    jsq_policy = JSQRouting() if use_jsq else None

    while t < sim_time:
        if use_jsq:
            # Reuse the shared JSQ implementation so tie handling matches the rest
            # of the codebase (equal split across all shortest queues).
            probs = jsq_policy(Q, rng)
        else:
            probs = compute_numpy_policy_probs(policy_net, Q, mu, rho, deterministic=deterministic)

        # [arrival_to_0, ..., arrival_to_{N-1}, departure_from_0, ..., departure_from_{N-1}]
        rates[:N] = lam * probs
        rates[N:] = mu * (Q > 0).astype(np.float64)

        a0 = rates.sum()
        if a0 <= constants.RATE_GUARD_EPSILON:
            break  # Degenerate—no events possible

        tau = rng.exponential(1.0 / a0)
        t += tau
        if t >= sim_time:
            break

        u = rng.uniform(0.0, a0)
        cumrates = np.cumsum(rates)
        event = int(np.searchsorted(cumrates, u, side='right'))
        event = min(event, 2 * N - 1)  # Safety clamp

        if event < N:
            states.append(Q.copy())
            actions.append(int(event))

            log_probs.append(float(np.log(probs[event] + constants.NUMERICAL_STABILITY_EPSILON)))
            Q[event] += 1
            arrival_count += 1

            action_step_indices.append(step_counter)
        else:
            srv = event - N
            Q[srv] -= 1
            departure_count += 1

        jump_times.append(t)
        all_states.append(Q.copy())
        step_counter += 1

    if jump_times:
        jump_times_arr = np.array(jump_times)
        dt = np.diff(jump_times_arr, append=sim_time)
        q_totals = np.array([np.sum(s) for s in all_states])
        integrated_queue = np.sum(q_totals * dt)
    else:
        integrated_queue = 0.0

    effective_rho = rho if rho is not None else (lam / np.sum(mu))

    return TrajectoryResult(
        log_probs=log_probs,
        states=states,
        actions=actions,
        total_integrated_queue=integrated_queue,
        arrival_count=arrival_count,
        departure_count=departure_count,
        jump_times=jump_times,
        action_step_indices=action_step_indices,
        all_states=all_states,
        rho=effective_rho,  # Store sampled rho
    )







class ReinforceTrainer:
    """
    REINFORCE training loop for N-GibbsQ.

    Implements the corrected training pipeline:
    1. Collect trajectories using true Gillespie SSA
    2. Compute REINFORCE gradients with baseline subtraction
    3. Update policy and value networks
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        run_dir: Path,
        run_logger=None,
        bc_data_config: dict | None = None,
    ):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.bc_data_config = bc_data_config

        self.num_servers = cfg.system.num_servers
        self.service_rates = np.array(cfg.system.service_rates, dtype=np.float64)
        self.service_rates_jax = jnp.array(self.service_rates)

        self.batch_size = cfg.batch_size
        self.sim_time = cfg.simulation.ssa.sim_time

    def bootstrap_from_expert(self, policy_net, value_net, key, jsq_limit, random_limit, denom):
        """Pretrain the neural network using centralized BC logic."""
        from gibbsq.core.pretraining import train_robust_bc_policy, train_robust_bc_value

        project_root = Path(__file__).resolve().parents[2]
        output_root = self.run_dir.parent.parent
        policy_net, warm_start_meta = try_load_bc_actor_warm_start(
            policy_net,
            cfg=self.cfg,
            bc_data_config=self.bc_data_config,
            project_root=project_root,
            output_root=output_root,
        )

        if not warm_start_meta["loaded"]:
            policy_net = train_robust_bc_policy(
                policy_net=policy_net,
                service_rates=self.service_rates,
                key=key,
                seed=self.cfg.simulation.seed,
                alpha=self.cfg.system.alpha,
                num_steps=self.cfg.neural_training.bc_num_steps,
                lr=self.cfg.neural_training.bc_lr,
                weight_decay=self.cfg.neural_training.weight_decay,
                label_smoothing=self.cfg.neural_training.bc_label_smoothing,
                bc_data_config=self.bc_data_config,
            )

        # Initializes the baseline to expert performance level to prevent early gradient noise
        value_net, _ = train_robust_bc_value(
            value_net=value_net,
            service_rates=self.service_rates,
            key=key,
            seed=self.cfg.simulation.seed,
            alpha=self.cfg.system.alpha,
            num_steps=self.cfg.neural_training.bc_num_steps,
            lr=self.cfg.neural_training.bc_lr,
            weight_decay=self.cfg.neural_training.weight_decay,
            jsq_limit=jsq_limit,
            random_limit=random_limit,
            denom=denom,
            squash_scale=self.cfg.neural_training.squash_scale,
            squash_threshold=self.cfg.neural_training.squash_threshold,
            bc_data_config=self.bc_data_config,
        )

        return policy_net, value_net, warm_start_meta

    def execute(self, key: PRNGKeyArray, n_epochs: int = 100):
        """Execute the REINFORCE training loop."""
        execute_start = time.perf_counter()
        stage_profile: dict[str, object] = {
            "oracle_rollout_engine": "numpy_ssa",
            "profile": {
                "batch_size": int(self.batch_size),
                "train_epochs": int(n_epochs),
                "sim_time": float(self.sim_time),
                "jax_enabled": bool(self.cfg.jax.enabled),
            },
            "setup": {},
            "epochs": [],
        }

        key, actor_key, critic_key = jax.random.split(key, 3)

        policy_net = NeuralRouter(
            num_servers=self.num_servers,
            config=self.cfg.neural,
            service_rates=self.service_rates,
            key=actor_key,
        )
        shake_key = jax.random.PRNGKey(self.cfg.simulation.seed + 99999)
        params, static = eqx.partition(policy_net, eqx.is_array)
        flat_params, unravel = jax.flatten_util.ravel_pytree(params)
        shake_scale = self.cfg.neural_training.shake_scale
        flat_params = flat_params + shake_scale * jax.random.normal(shake_key, flat_params.shape)
        new_params = unravel(flat_params)
        policy_net = eqx.combine(new_params, static)


        value_net = ValueNetwork(
            num_servers=self.num_servers,
            config=self.cfg.neural,
            hidden_size=self.cfg.neural.hidden_size,
            key=critic_key,
        )

        policy_opt = optax.chain(
            optax.clip_by_global_norm(self.cfg.neural.clip_global_norm),
            optax.adamw(learning_rate=self.cfg.neural.actor_lr, weight_decay=self.cfg.neural.weight_decay)
        )
        value_opt = optax.chain(
            optax.clip_by_global_norm(self.cfg.neural.clip_global_norm),
            optax.adamw(learning_rate=self.cfg.neural.critic_lr, weight_decay=self.cfg.neural.weight_decay)
        )

        policy_state = policy_opt.init(eqx.filter(policy_net, eqx.is_array))
        value_state = value_opt.init(eqx.filter(value_net, eqx.is_array))

        jsq_queues = []
        with create_progress(total=3, desc="reinforce: setup", unit="stage", leave=False) as setup_bar:
            jsq_started = time.perf_counter()
            for _ in iter_progress(
                range(self.batch_size),
                total=self.batch_size,
                desc="reinforce: JSQ baseline",
                unit="traj",
                leave=False,
            ):
                rng = np.random.default_rng(self.cfg.simulation.seed + _)
                traj_jsq = collect_trajectory_ssa(
                    policy_net=policy_net,  # Not used in JSQ mode
                    num_servers=self.num_servers,
                    arrival_rate=self.cfg.system.arrival_rate,
                    service_rates=self.service_rates,
                    sim_time=self.sim_time,
                    rng=rng,
                    use_jsq=True
                )
                jsq_queues.append(traj_jsq.total_integrated_queue / self.sim_time)
            setup_bar.update(1)
            stage_profile["setup"] = {
                "jsq_baseline_seconds": time.perf_counter() - jsq_started,
                "jsq_baseline_mean_queue": float(np.mean(jsq_queues)) if jsq_queues else 0.0,
                "jsq_baseline_trajectories": int(len(jsq_queues)),
            }

            jsq_limit = np.mean(jsq_queues)

            # Mean queue under uniform random routing when lambda / N < mu_i.
            # E[Q_random] = sum_i (lambda / N) / (mu_i - lambda / N)
            q_rand_analytical = 0.0
            is_unstable = False
            lam_i = self.cfg.system.arrival_rate / self.num_servers
            for mu in self.service_rates:
                if lam_i >= mu - 1e-4:
                    is_unstable = True
                    break
                q_rand_analytical += lam_i / (mu - lam_i)

            if is_unstable:
                analytical_random_limit = 50.0 * self.num_servers
            else:
                analytical_random_limit = q_rand_analytical

            random_limit = analytical_random_limit
            random_queue = analytical_random_limit  # For consistency with logging

            log.info(f"  JSQ Mean Queue (Target): {jsq_limit:.4f}")
            log.info(f"  Random Mean Queue (Analytical): {analytical_random_limit:.4f}")

            fixed_random = max(jsq_limit * 1.01, random_limit, random_queue)
            denom = max(self.cfg.neural_training.perf_index_min_denom, jsq_limit * self.cfg.neural_training.perf_index_jsq_margin, fixed_random - jsq_limit)

            # Uses pre-bootstrap baselines when computing the initial denominator.
            key, bootstrap_key = jax.random.split(key)
            bootstrap_started = time.perf_counter()
            policy_net, value_net, warm_start_meta = self.bootstrap_from_expert(
                policy_net, value_net, bootstrap_key, jsq_limit, random_limit, denom
            )
            setup_profile = dict(stage_profile["setup"])
            setup_profile.update(
                {
                    "bootstrap_seconds": time.perf_counter() - bootstrap_started,
                    "warm_start": warm_start_meta,
                }
            )
            stage_profile["setup"] = setup_profile
            setup_bar.update(1)

            policy_state = policy_opt.init(eqx.filter(policy_net, eqx.is_array))
            value_state = value_opt.init(eqx.filter(value_net, eqx.is_array))

            log.info("=" * 60)
            log.info("  REINFORCE Training (SSA-Based Policy Gradient)")
            log.info("=" * 60)
            log.info(f"  Epochs: {n_epochs}, Batch size: {self.batch_size}")
            log.info(f"  Simulation time: {self.sim_time}")
            log.info("-" * 60)
            setup_bar.update(1)

        history_loss = []
        history_reward = []

        def save_checkpoint(epoch_idx):
            ckpt_path = artifacts_dir(self.run_dir) / f"policy_net_epoch_{epoch_idx:03d}.eqx"
            eqx.tree_serialise_leaves(ckpt_path, policy_net)
            log.info(f"  [Checkpoint] Saved epoch {epoch_idx} model to {ckpt_path.name}")

            # Update pointer in real-time for comparison scripts via model_io
            if getattr(self, "save_global_pointer", True):
                _PROJECT_ROOT = Path(__file__).resolve().parents[2]
                pointer_dir = self.run_dir.parent.parent
                save_model_pointer(
                    model_path=ckpt_path,
                    project_root=_PROJECT_ROOT,
                    output_root=pointer_dir,
                    pointer_name="latest_reinforce_weights.txt"
                )

        ema_base_idx = 0.0
        ema_corr = 0.0
        ema_ev = 0.0
        alpha = 0.33  # ~Window 5

        base_actor_lr = self.cfg.neural.actor_lr

        training_complete = False
        last_successful_epoch = -1

        # Create optimizer ONCE to preserve Adam momentum/variance.
        # Use inject_hyperparams so we can update LR each epoch without resetting state.
        policy_opt = optax.chain(
            optax.clip_by_global_norm(self.cfg.neural.clip_global_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=base_actor_lr,
                weight_decay=self.cfg.neural.weight_decay,
            ),
        )
        policy_state = policy_opt.init(eqx.filter(policy_net, eqx.is_array))

        try:
            with create_progress(total=n_epochs, desc="reinforce_train", unit="epoch") as epoch_bar:
                for epoch in range(n_epochs):
                    # Apply Adaptive Linear Decay to Actor LR (prevents unlearning)
                    epoch_actor_lr = base_actor_lr * (1.0 - (epoch / n_epochs) * self.cfg.neural.lr_decay_rate)

                    policy_state[1].hyperparams["learning_rate"] = jnp.asarray(
                        epoch_actor_lr, dtype=jnp.float32
                    )

                    entropy_anneal_epochs = n_epochs  # Decay over full training
                    entropy_final = self.cfg.neural.entropy_final
                    entropy_initial = self.cfg.neural.entropy_bonus
                    if epoch < entropy_anneal_epochs:
                        progress = epoch / entropy_anneal_epochs
                        current_entropy_coef = entropy_initial * (1.0 - progress) + entropy_final * progress
                    else:
                        current_entropy_coef = entropy_final
                    if epoch > 0 and epoch % self.cfg.neural_training.checkpoint_freq == 0:
                        save_checkpoint(epoch)

                    trajectories = []
                    epoch_rewards = []

                    dr_cfg = getattr(self.cfg, 'domain_randomization', None)
                    if dr_cfg and getattr(dr_cfg, 'enabled', False):
                        if hasattr(dr_cfg, 'phases') and len(dr_cfg.phases) > 0:
                            phase_idx = 0
                            epochs_accum = 0
                            for i, p in enumerate(dr_cfg.phases):
                                epochs_accum += p.epochs
                                if epoch < epochs_accum:
                                    phase_idx = i
                                    break
                                phase_idx = i
                            active_phase = dr_cfg.phases[phase_idx]
                            rho_min = active_phase.rho_min
                            rho_max = active_phase.rho_max
                        else:
                            rho_min = getattr(dr_cfg, 'rho_min', 0.4)
                            rho_max = getattr(dr_cfg, 'rho_max', 0.85)
                    else:
                        config_rho = self.cfg.system.arrival_rate / np.sum(self.service_rates)
                        rho_min = float(config_rho)
                        rho_max = float(config_rho)

                    rollout_started = time.perf_counter()
                    for _ in iter_progress(
                        range(self.batch_size),
                        total=self.batch_size,
                        desc=f"reinforce epoch {epoch + 1}/{n_epochs}",
                        unit="traj",
                        leave=False,
                    ):
                        rng = np.random.default_rng(self.cfg.simulation.seed + epoch * self.batch_size + _)
                        sampled_rho = rng.uniform(rho_min, rho_max)
                        total_capacity = np.sum(self.service_rates)
                        sampled_arrival_rate = sampled_rho * total_capacity

                        traj = collect_trajectory_ssa(
                            policy_net=policy_net,
                            num_servers=self.num_servers,
                            arrival_rate=sampled_arrival_rate,
                            service_rates=self.service_rates,
                            sim_time=self.sim_time,
                            rng=rng,
                            rho=sampled_rho
                        )
                        trajectories.append(traj)
                        epoch_rewards.append(traj.total_integrated_queue)
                    rollout_seconds = time.perf_counter() - rollout_started

                    mean_queue = np.mean(epoch_rewards) / self.sim_time
                    base_regime_index = compute_performance_index(
                        mean_queue, jsq_limit, random_limit, random_queue, self.cfg
                    )

                    fixed_random = max(jsq_limit * 1.01, random_limit, random_queue)
                    denom = max(self.cfg.neural_training.perf_index_min_denom, jsq_limit * self.cfg.neural_training.perf_index_jsq_margin, fixed_random - jsq_limit)

                    all_advantages = []
                    gae_lambda = getattr(self.cfg.neural_training, 'gae_lambda', 0.95)
                    use_gae = gae_lambda > 0.0 and gae_lambda < 1.0

                    advantage_started = time.perf_counter()
                    for traj in trajectories:
                        G_raw = compute_causal_returns_to_go(
                            traj.all_states, traj.jump_times, traj.action_step_indices,
                            sim_time=self.sim_time,
                            gamma=self.cfg.neural_training.gamma
                        )

                        if use_gae:
                            gae_adv, gae_ret = compute_gae(
                                traj.all_states, traj.jump_times, traj.action_step_indices,
                                sim_time=self.sim_time,
                                gamma=self.cfg.neural_training.gamma,
                                gae_lambda=gae_lambda,
                                value_net=value_net,
                                service_rates=self.service_rates,
                                rho=traj.rho,
                                jsq_limit=jsq_limit,
                                random_limit=random_limit,
                                denom=denom,
                                squash_scale=self.cfg.neural_training.squash_scale,
                                squash_threshold=self.cfg.neural_training.squash_threshold,
                                decision_states=traj.states,
                            )
                            all_advantages.append((G_raw, gae_adv, gae_ret))
                        else:
                            all_advantages.append((G_raw, None, None))
                    advantage_seconds = time.perf_counter() - advantage_started

                    batch_S = []
                    batch_A = []
                    batch_G = []
                    batch_G_raw = []  # Track raw returns for GAE Critic Target
                    batch_gae_adv = []
                    batch_rho = []
                    batch_traj_indices = []

                    batch_Ret = []
                    for i, traj in enumerate(trajectories):
                        G_raw, gae_adv, gae_ret = all_advantages[i]
                        if len(G_raw) > 0:
                            t_actions = np.array([traj.jump_times[idx] for idx in traj.action_step_indices])
                            t_rem = np.maximum(1e-3, self.sim_time - t_actions)

                            G_idx = 100.0 * (random_limit * t_rem - G_raw) / (denom * t_rem)

                            batch_S.extend(traj.states)
                            batch_A.extend(traj.actions)
                            batch_G.extend(np.atleast_1d(G_idx).tolist())
                            batch_G_raw.extend(np.atleast_1d(G_raw).tolist())
                            if gae_adv is not None:
                                batch_gae_adv.extend(np.atleast_1d(gae_adv).tolist())
                                batch_Ret.extend(np.atleast_1d(gae_ret).tolist())

                            batch_rho.extend([traj.rho] * int(np.array(G_idx).size))
                            batch_traj_indices.extend([i] * int(np.array(G_idx).size))

                    policy_loss = 0.0
                    value_loss = 0.0
                    avg_entropy = 0.0
                    policy_aux = {"entropy": 0.0, "mean_logp": 0.0, "mean_adv": 0.0}

                    if len(batch_G) > 0:
                        update_started = time.perf_counter()
                        S_tensor = jnp.asarray(np.stack(batch_S), dtype=jnp.float32)
                        A_tensor = jnp.asarray(batch_A, dtype=jnp.int32)
                        G_tensor = jnp.asarray(batch_G, dtype=jnp.float32)
                        G_raw_tensor = jnp.asarray(batch_G_raw, dtype=jnp.float32)
                        rho_tensor = jnp.asarray(batch_rho, dtype=jnp.float32)
                        idx_tensor = jnp.asarray(batch_traj_indices, dtype=jnp.int32)

                        G_scaled = G_tensor

                        v_preds = jax.lax.stop_gradient(
                            jax.vmap(value_net, in_axes=(0, None, 0))(
                                S_tensor, self.service_rates_jax, rho_tensor
                            )
                        )

                        if use_gae:
                            raw_adv = jnp.asarray(batch_gae_adv, dtype=jnp.float32)
                            Ret_tensor = jnp.asarray(batch_Ret, dtype=jnp.float32)
                            critic_target = Ret_tensor
                        else:
                            raw_adv = G_scaled - v_preds
                            critic_target = G_scaled

                        def advantage_norm_fn(advs):
                            return (advs - jnp.mean(advs)) / (jnp.std(advs) + 1e-8)

                        norm_adv = advantage_norm_fn(raw_adv)

                        def policy_loss_fn(model, Q_feat, mu_feat, rho_feat, actions, advs, traj_indices, ent_coef):
                            logits = jax.vmap(model, in_axes=(0, None, 0))(Q_feat, mu_feat, rho_feat)

                            log_probs = jax.nn.log_softmax(logits, axis=-1)
                            chosen_log_probs = log_probs[jnp.arange(len(actions)), actions]

                            action_signals = chosen_log_probs * advs

                            probs = jax.nn.softmax(logits, axis=-1)
                            entropy = -jnp.sum(probs * log_probs, axis=-1)

                            num_trajs = self.batch_size
                            traj_sums = jax.ops.segment_sum(action_signals, traj_indices, num_segments=num_trajs)
                            traj_counts = jax.ops.segment_sum(
                                jnp.ones_like(action_signals),
                                traj_indices,
                                num_segments=num_trajs,
                            )
                            traj_means = traj_sums / jnp.maximum(1.0, traj_counts)

                            traj_entropy = jax.ops.segment_sum(
                                entropy, traj_indices, num_segments=num_trajs
                            ) / jnp.maximum(1.0, traj_counts)

                            total_loss = -jnp.mean(traj_means) - ent_coef * jnp.mean(traj_entropy)

                            def corr_fn(logp, advantages):
                                neg_logp = -logp.flatten()
                                adv_flat = advantages.flatten()
                                return jnp.corrcoef(jnp.stack([neg_logp, adv_flat]))[0, 1]

                            corr = corr_fn(chosen_log_probs, advs)

                            def explained_variance(y_true, y_pred):
                                var_y = jnp.var(y_true)
                                ev = 1.0 - jnp.var(y_true - y_pred) / (var_y + 1e-8)
                                return ev

                            ev = explained_variance(critic_target, v_preds)

                            mean_logp = jnp.mean(chosen_log_probs)
                            mean_adv = jnp.mean(advs)

                            return total_loss, {
                                "entropy": jnp.mean(traj_entropy),
                                "mean_logp": mean_logp,
                                "mean_adv": mean_adv,
                                "corr": corr,
                                "ev": ev,
                            }

                        (policy_loss, policy_aux), policy_grads = eqx.filter_value_and_grad(
                            policy_loss_fn, has_aux=True
                        )(
                            policy_net,
                            S_tensor,
                            self.service_rates_jax,
                            rho_tensor,
                            A_tensor,
                            norm_adv,
                            idx_tensor,
                            current_entropy_coef,
                        )
                        policy_grad_norm = jnp.sqrt(
                            sum(
                                jnp.sum(jnp.square(g))
                                for g in jax.tree_util.tree_leaves(policy_grads)
                                if g is not None
                            )
                        )

                        updates, policy_state = policy_opt.update(
                            policy_grads, policy_state, eqx.filter(policy_net, eqx.is_array)
                        )
                        policy_net = eqx.apply_updates(policy_net, updates)

                        def value_loss_fn(model, Q_feat, mu_feat, rho_feat, g_targets):
                            preds = jax.vmap(model, in_axes=(0, None, 0))(Q_feat, mu_feat, rho_feat)
                            return jnp.mean((preds - g_targets) ** 2)

                        value_loss, value_grads = eqx.filter_value_and_grad(value_loss_fn)(
                            value_net, S_tensor, self.service_rates_jax, rho_tensor, critic_target
                        )

                        value_grad_norm = jnp.sqrt(
                            sum(
                                jnp.sum(jnp.square(g))
                                for g in jax.tree_util.tree_leaves(value_grads)
                                if g is not None
                            )
                        )

                        updates, value_state = value_opt.update(
                            value_grads, value_state, eqx.filter(value_net, eqx.is_array)
                        )
                        value_net = eqx.apply_updates(value_net, updates)
                        update_seconds = time.perf_counter() - update_started

                        if epoch % 10 == 0:
                            avg_ent = policy_aux["entropy"]
                            mean_logp = policy_aux["mean_logp"]
                            mean_adv = policy_aux["mean_adv"]
                            corr = policy_aux.get("corr", 0.0)
                            log.info(
                                f"    [Sign Check] mean_adv: {mean_adv:.4f} | mean_loss: {float(policy_loss):.4f} | mean_logp: {mean_logp:.4f} | corr: {corr:.4f}"
                            )
                            log.info(
                                f"    [Grad Check] P-Grad Norm: {policy_grad_norm:.4f} | V-Grad Norm: {value_grad_norm:.4f}"
                            )
                    else:
                        update_seconds = 0.0

                    epoch_profile = {
                        "epoch": int(epoch),
                        "rollout_seconds": float(rollout_seconds),
                        "advantage_seconds": float(advantage_seconds),
                        "update_seconds": float(update_seconds),
                        "mean_queue": float(mean_queue),
                        "mean_integrated_queue": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
                        "mean_event_steps": float(np.mean([len(t.jump_times) for t in trajectories])) if trajectories else 0.0,
                        "mean_actions": float(np.mean([len(t.actions) for t in trajectories])) if trajectories else 0.0,
                        "mean_arrivals": float(np.mean([t.arrival_count for t in trajectories])) if trajectories else 0.0,
                        "mean_departures": float(np.mean([t.departure_count for t in trajectories])) if trajectories else 0.0,
                        "num_trajectories": int(len(trajectories)),
                        "base_regime_index": float(base_regime_index),
                    }
                    cast_epochs = stage_profile["epochs"]
                    assert isinstance(cast_epochs, list)
                    if len(cast_epochs) < 10 or epoch == n_epochs - 1:
                        cast_epochs.append(epoch_profile)

                    if epoch == 0:
                        ema_base_idx = base_regime_index
                        ema_corr = float(policy_aux.get("corr", 0.0))
                        ema_ev = float(policy_aux.get("ev", 0.0))
                    else:
                        ema_base_idx = alpha * base_regime_index + (1 - alpha) * ema_base_idx
                        ema_corr = alpha * float(policy_aux.get("corr", 0.0)) + (1 - alpha) * ema_corr
                        ema_ev = alpha * float(policy_aux.get("ev", 0.0)) + (1 - alpha) * ema_ev

                    history_loss.append(float(policy_loss))
                    history_reward.append(base_regime_index)
                    if epoch % 5 == 0:
                        log.info(
                            f"Epoch {epoch:4d} | mean_queue: {mean_queue:5.3f} | "
                            f"base_regime_index: {base_regime_index:6.1f}% (EMA: {ema_base_idx:6.1f}%) | "
                            f"Loss: {float(policy_loss):.4f} | V-Loss: {float(value_loss):.4f}"
                        )
                        log.info(f"   -> Signaling | EV: {ema_ev:6.3f} [EMA], Corr: {ema_corr:6.4f} [EMA]")

                    metrics = {
                        "epoch": epoch,
                        "policy_loss": float(policy_loss),
                        "value_loss": float(value_loss),
                        "base_regime_index": float(base_regime_index),
                        "base_regime_index_ema": float(ema_base_idx),
                        "performance_index": float(base_regime_index),
                        "performance_index_ema": float(ema_base_idx),
                        "mean_queue": float(mean_queue),
                        "ev_ema": float(ema_ev),
                        "corr_ema": float(ema_corr),
                        "jsq_limit": float(jsq_limit),
                        "random_analytical": float(random_limit),
                        "random_empirical": float(random_queue),
                        "arrival_count": int(np.mean([t.arrival_count for t in trajectories])),
                        "mean_adv": float(policy_aux.get("mean_adv", 0.0)),
                        "mean_logp": float(policy_aux.get("mean_logp", 0.0)),
                        "policy_grad_norm": float(policy_grad_norm) if 'policy_grad_norm' in locals() else 0.0,
                        "value_grad_norm": float(value_grad_norm) if 'value_grad_norm' in locals() else 0.0,
                        "entropy": float(policy_aux.get("entropy", 0.0)),
                    }
                    append_metrics_jsonl(metrics, metrics_path(self.run_dir, "reinforce_metrics.jsonl"))
                    if self.run_logger:
                        self.run_logger.log(metrics)

                    epoch_bar.set_postfix(
                        {
                            "queue": f"{mean_queue:.3f}",
                            "pi": f"{base_regime_index:.1f}%",
                        },
                        refresh=False,
                    )
                    epoch_bar.update(1)

                    last_successful_epoch = epoch

            training_complete = True

        except Exception as e:
            log.error(f"Training interrupted at epoch {last_successful_epoch}: {e}")
            if last_successful_epoch >= 0:
                save_checkpoint(last_successful_epoch)
                log.info(f"Saved emergency checkpoint at epoch {last_successful_epoch}")
            raise

        stage_profile["training_loop_seconds"] = time.perf_counter() - execute_start
        self._save_assets(
            policy_net,
            value_net,
            history_loss,
            history_reward,
            jsq_limit,
            random_limit,
            random_queue,
            stage_profile=stage_profile,
            execute_start=execute_start,
        )

    def _save_assets(
        self,
        policy_net: NeuralRouter,
        value_net: ValueNetwork,
        history_loss: list,
        history_reward: list,
        jsq_limit: float,
        random_limit: float,
        random_queue: float,
        stage_profile: dict | None = None,
        execute_start: float | None = None,
    ):
        """Persist model weights and training history."""
        import matplotlib.pyplot as plt
        from gibbsq.analysis.theme import apply_theme, THEMES
        from gibbsq.utils.chart_exporter import save_chart

        artifacts = artifacts_dir(self.run_dir)
        policy_path = artifacts / "n_gibbsq_reinforce_weights.eqx"
        eqx.tree_serialise_leaves(policy_path, policy_net)

        value_path = artifacts / "value_network_weights.eqx"
        eqx.tree_serialise_leaves(value_path, value_net)

        from gibbsq.analysis.plotting import plot_training_dashboard
        import json

        metrics_file = metrics_path(self.run_dir, "reinforce_metrics.jsonl")
        dashboard_metrics: dict = {
            "epoch": [], "base_regime_index": [], "base_regime_index_ema": [],
            "policy_loss": [], "value_loss": [], "ev_ema": [], "corr_ema": [],
            "policy_grad_norm": [], "value_grad_norm": [], "entropy": [],
        }
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                for line in f:
                    row = json.loads(line.strip())
                    for key in dashboard_metrics:
                        if key in row:
                            dashboard_metrics[key].append(row[key])

        pointer_dir = self.run_dir.parent.parent
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
        save_model_pointer(
            model_path=policy_path,
            project_root=_PROJECT_ROOT,
            output_root=pointer_dir,
            pointer_name="latest_reinforce_weights.txt"
        )

        log.info("-" * 55)
        log.info("-------------------------------------------------------")
        log.info(f"Running Final Deterministic Evaluation (N={self.cfg.neural_training.eval_batches * self.cfg.neural_training.eval_trajs_per_batch})...")
        eval_started = time.perf_counter()
        eval_indices = []
        with create_progress(
            total=self.cfg.neural_training.eval_batches,
            desc="reinforce: final eval",
            unit="batch",
            leave=False,
        ) as eval_bar:
            for b in range(self.cfg.neural_training.eval_batches):
                eval_trajs = []
                for i in iter_progress(
                    range(self.cfg.neural_training.eval_trajs_per_batch),
                    total=self.cfg.neural_training.eval_trajs_per_batch,
                    desc=f"reinforce eval batch {b + 1}/{self.cfg.neural_training.eval_batches}",
                    unit="traj",
                    leave=False,
                ):
                    eval_rng = np.random.default_rng(self.cfg.simulation.seed + 10000 + b * 10 + i)
                    t = collect_trajectory_ssa(
                        policy_net=policy_net,
                        num_servers=self.num_servers,
                        arrival_rate=self.cfg.system.arrival_rate,
                        service_rates=self.service_rates,
                        sim_time=self.sim_time,
                        rng=eval_rng,
                        rho=self.cfg.system.arrival_rate / np.sum(self.service_rates),
                        deterministic=True,
                    )
                    eval_trajs.append(t)

                batch_indices = []
                for t in eval_trajs:
                    p_idx = compute_performance_index(
                        t.total_integrated_queue / self.sim_time,
                        jsq_limit,
                        random_limit,
                        random_queue,
                        self.cfg,
                    )
                    batch_indices.append(p_idx)

                eval_indices.extend(batch_indices)
                eval_bar.update(1)

        final_mean = np.mean(eval_indices)
        final_std = np.std(eval_indices)
        if stage_profile is not None:
            stage_profile["final_eval"] = {
                "seconds": float(time.perf_counter() - eval_started),
                "mean_score": float(final_mean),
                "std_score": float(final_std),
                "num_scores": int(len(eval_indices)),
            }
            if execute_start is not None:
                stage_profile["total_wall_seconds"] = float(time.perf_counter() - execute_start)
            stage_profile_path = metrics_path(self.run_dir, "reinforce_stage_profile.json")
            stage_profile_path.write_text(json.dumps(stage_profile, indent=2), encoding="utf-8")
            log.info("Stage profile written to %s", stage_profile_path)
        log.info(f"Deterministic Policy Score: {final_mean:.2f}% ± {final_std:.2f}%")
        log.info(f"JSQ Target: 100.0% | Random Floor: 0.0% (Performance Index Scale)")
        log.info("-------------------------------------------------------")

        dashboard_metrics.update(
            {
                "final_eval_mean": float(final_mean),
                "final_eval_std": float(final_std),
                "final_eval_count": int(len(eval_indices)),
                "train_epochs": int(self.cfg.train_epochs),
                "run_label": "debug" if int(self.cfg.train_epochs) <= 5 else "training",
            }
        )
        plot_path = figure_path(self.run_dir, "reinforce_training_curve")
        fig = plot_training_dashboard(
            metrics=dashboard_metrics,
            jsq_baseline=100.0,  # JSQ = 100% on Performance Index scale
            random_baseline=0.0,  # Random = 0% on Performance Index scale
            save_path=plot_path,
            theme="publication",
            formats=["png", "pdf"],
            context=ExperimentPlotContext(
                experiment_id="training",
                chart_name="plot_training_dashboard",
            ),
        )
        plt.close(fig)

        log.info(f"Training Complete! Final Loss: {history_loss[-1]:.4f}")
        log.info(f"Final Base-Regime Index Proxy: {history_reward[-1]:.2f}")
        log.info(f"Policy weights: {policy_path}")
        log.info(f"Value weights: {value_path}")



def main(raw_cfg: DictConfig):
    """Main entry point for REINFORCE training."""
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "reinforce_train")
    bc_data_config = extract_bc_data_config(resolved_raw_cfg)

    run_dir, run_id = get_run_config(cfg, "reinforce_train", resolved_raw_cfg)
    run_logger = setup_wandb(cfg, resolved_raw_cfg, default_group="reinforce_training",
                            run_id=run_id, run_dir=run_dir)

    trainer = ReinforceTrainer(cfg, run_dir, run_logger, bc_data_config=bc_data_config)

    seed_key = jax.random.PRNGKey(cfg.simulation.seed)
    trainer.execute(seed_key, n_epochs=cfg.train_epochs)


if __name__ == "__main__":
    import sys
    import hydra
    if len(sys.argv) > 1:
        hydra.main(version_base=None, config_path="../../configs", config_name="default")(main)()
    else:
        from hydra import compose, initialize_config_dir
        import os
        config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raw_cfg = compose(config_name="default")
            main(raw_cfg)
