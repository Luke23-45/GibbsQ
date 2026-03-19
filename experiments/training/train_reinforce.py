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

import logging
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

from gibbsq.core.config import ExperimentConfig, NeuralConfig, hydra_to_config, validate
from gibbsq.core.neural_policies import NeuralRouter, ValueNetwork
from gibbsq.core.features import sojourn_time_features
from gibbsq.core.policies import JSQRouting, JSSQRouting
from gibbsq.core import constants
# Local simulate not needed if defined/used via engines.numpy_engine directly if ever needed,
# but here we use the local collect_trajectory_ssa.
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory Collection Result
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryResult(NamedTuple):
    """Result from a single SSA trajectory with routing decisions."""
    log_probs: list[float]  # log π_θ(a_t | s_t) for each routing decision
    states: list[np.ndarray]  # Queue state BEFORE each decision
    actions: list[int]  # Server index chosen for each arrival
    total_integrated_queue: float  # Integrated queue length over time (True Reward)
    arrival_count: int
    departure_count: int
    # New fields for causal returns-to-go computation
    jump_times: list[float]  # Time of each SSA event
    action_step_indices: list[int]  # Indices where routing decisions were made
    all_states: list[np.ndarray]  # All states (not just at decision points)


def compute_causal_returns_to_go(
    states: list[np.ndarray],
    jump_times: list[float],
    action_step_indices: list[int],
    sim_time: float,
    gamma: float = 0.99  # CRITICAL UPDATE: Stationary Discounting
) -> np.ndarray:
    """Computes Discounted Causal Returns, stripping non-stationary time dependence."""
    if not states or not action_step_indices:
        return np.array([])
    
    # 1. Continuous interval area computation
    q_totals = np.array([np.sum(s) for s in states])
    dt = np.diff(np.array(jump_times), append=sim_time)
    q_integrals = q_totals * dt
    
    # 2. Assign the immediate cost of an action to the interval BEFORE the next action
    action_intervals = []
    n_actions = len(action_step_indices)
    for i in range(n_actions):
        start_idx = action_step_indices[i]
        end_idx = action_step_indices[i+1] if (i + 1 < n_actions) else len(q_integrals)
        
        # Immediate area under the queue before the system must decide routing again
        interval_cost = np.sum(q_integrals[start_idx:end_idx])
        action_intervals.append(interval_cost)
        
    action_intervals = np.array(action_intervals)
    
    # 3. Standard backward discounting (Solves the Critic's impossible T-t prediction task)
    returns = np.zeros_like(action_intervals, dtype=np.float64)
    R = 0.0
    for i in reversed(range(len(action_intervals))):
        R = action_intervals[i] + gamma * R
        returns[i] = R
        
    return returns


def compute_gae(
    states: list[np.ndarray],
    jump_times: list[float],
    action_step_indices: list[int],
    sim_time: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    value_net=None,
    service_rates: np.ndarray = None,
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE) for CTMC trajectories.
    
    PATCH-P1: Implements Schulman et al. (2015) GAE for continuous-time queueing.
    
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
        All states visited during trajectory.
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
    
    Returns
    -------
    np.ndarray
        GAE advantages for each action.
    """
    if not states or not action_step_indices or value_net is None:
        # Fallback to returns-to-go if no critic
        return compute_causal_returns_to_go(states, jump_times, action_step_indices, sim_time, gamma)
    
    # 1. Compute continuous-time intervals
    q_totals = np.array([np.sum(s) for s in states])
    dt = np.diff(np.array(jump_times), append=sim_time)
    
    # 2. Compute rewards: negative queue integral (we want to minimize queue)
    # Each state contributes -Q_total * dt (negative because we minimize)
    rewards = -q_totals * dt
    
    # 3. Compute V(s) for all states using critic
    import jax.numpy as jnp
    from gibbsq.core.features import sojourn_time_features
    
    states_arr = np.array(states)
    s_feat = sojourn_time_features(jnp.asarray(states_arr), jnp.asarray(service_rates))
    v_preds = np.array(jax.vmap(value_net)(s_feat, jnp.zeros(len(states))))  # TODO: pass rho
    
    # 4. Compute TD residuals with continuous-time discounting
    # δ_t = r_t + γ^(Δt) * V(s_{t+1}) - V(s_t)
    td_residuals = np.zeros(len(states))
    for t in range(len(states) - 1):
        gamma_dt = gamma ** dt[t]  # Continuous-time discount
        td_residuals[t] = rewards[t] + gamma_dt * v_preds[t + 1] - v_preds[t]
    # Terminal state: no bootstrap
    td_residuals[-1] = rewards[-1] - v_preds[-1]
    
    # 5. Compute GAE: A_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}
    # For actions, we sum TD residuals between action points
    advantages = np.zeros(len(action_step_indices))
    
    for i, action_idx in enumerate(action_step_indices):
        # Sum TD residuals from this action to the next
        next_action_idx = action_step_indices[i + 1] if i + 1 < len(action_step_indices) else len(td_residuals)
        
        gae = 0.0
        decay = 1.0
        for t in range(action_idx, next_action_idx):
            gae += decay * td_residuals[t]
            dt_t = dt[t] if t < len(dt) else 0.0
            decay *= (gamma * gae_lambda) ** dt_t  # Continuous-time decay
        
        advantages[i] = gae
    
    return advantages


def compute_performance_index(
    mean_queue: float,
    jsq_limit: float,
    random_limit: float,
    random_empirical: float
) -> float:
    """Computes Performance Index where 0% = Random and 100% = JSQ."""
    # Use the same robust denominator as the original training loop
    fixed_random = max(jsq_limit * 1.01, random_limit, random_empirical)
    denom = max(0.5, jsq_limit * 0.05, fixed_random - jsq_limit)
    perf_index = 100.0 * (fixed_random - mean_queue) / denom
    return perf_index


# ValueNetwork moved to gibbsq.core.neural_policies


# ─────────────────────────────────────────────────────────────────────────────
# SSA Trajectory Collection with Routing Decisions
# ─────────────────────────────────────────────────────────────────────────────

def collect_trajectory_ssa(
    policy_net: NeuralRouter,
    num_servers: int,
    arrival_rate: float,
    service_rates: np.ndarray,
    sim_time: float,
    rng: np.random.Generator,
    use_jsq: bool = False,
    rho: float = None,  # PI-V4: Optional load factor for neural policy
    deterministic: bool = False, # PI-V5: Support for deterministic evaluation
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
    
    # State
    Q = np.zeros(N, dtype=np.int64)
    t = 0.0
    
    # Optimization: Extract NumPy parameters once to avoid JAX overhead in the loop
    np_params = policy_net.get_numpy_params()
    np_config = policy_net.config
    
    # Recording
    log_probs = []
    states = []  # States at decision points (arrivals)
    actions = []
    total_queue = 0.0
    arrival_count = 0
    departure_count = 0
    
    # New: Track all states and jump times for causal returns
    jump_times = []  # Time of each SSA event
    action_step_indices = []  # Indices where routing decisions were made
    all_states = []  # All states (not just at decision points)
    step_counter = 0  # Global step counter for indexing
    
    # Pre-allocated work arrays
    rates = np.empty(2 * N, dtype=np.float64)
    
    # Main event loop
    while t < sim_time:
        if use_jsq:
            # Join Shortest Queue (JSQ) Logic
            chosen_srv = int(np.argmin(Q))
            probs = np.zeros(N)
            probs[chosen_srv] = 1.0
            logits = np.zeros(N) # Not used but avoids UnboundLocal
        else:
            # 1. Compute routing probabilities using sojourn-time features
            # Use fast NumPy forward pass
            s = (Q + 1.0) / mu
            logits = policy_net.numpy_forward(s, np_params, np_config, rho=rho)
            
            # Log-sum-exp for numerical stability
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / probs.sum()
            
            if deterministic:
                # Greedy selection for evaluation
                best_srv = int(np.argmax(probs))
                probs = np.zeros_like(probs)
                probs[best_srv] = 1.0
        
        # 2. Build event-rate vector
        # [arrival_to_0, ..., arrival_to_{N-1}, departure_from_0, ..., departure_from_{N-1}]
        rates[:N] = lam * probs
        rates[N:] = mu * (Q > 0).astype(np.float64)
        
        # 3. Total propensity
        a0 = rates.sum()
        if a0 <= constants.RATE_GUARD_EPSILON:
            break  # Degenerate—no events possible
        
        # 4. Draw holding time ~ Exp(a0)
        tau = rng.exponential(1.0 / a0)
        t += tau
        if t >= sim_time:
            break
        
        # 5. Select event via inverse CDF
        u = rng.uniform(0.0, a0)
        cumrates = np.cumsum(rates)
        event = int(np.searchsorted(cumrates, u, side='right'))
        event = min(event, 2 * N - 1)  # Safety clamp
        
        # 6. Apply transition
        if event < N:
            # Arrival: record state BEFORE decision
            states.append(Q.copy())
            actions.append(int(event))
            
            # Record log probability of routing decision
            log_probs.append(float(np.log(probs[event] + constants.NUMERICAL_STABILITY_EPSILON)))
            Q[event] += 1
            arrival_count += 1
            
            # Track this as a routing decision point
            action_step_indices.append(step_counter)
        else:
            # Departure
            srv = event - N
            Q[srv] -= 1
            departure_count += 1
        
        # Record state and time for causal returns computation
        jump_times.append(t)
        all_states.append(Q.copy())
        step_counter += 1
        
    # Compute reward: total integrated queue (Patch 1: area under the curve)
    # We use jump_times and all_states to get the true area from t1 to T
    if jump_times:
        jump_times_arr = np.array(jump_times)
        dt = np.diff(jump_times_arr, append=sim_time)
        q_totals = np.array([np.sum(s) for s in all_states])
        integrated_queue = np.sum(q_totals * dt)
    else:
        integrated_queue = 0.0
    
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
    )


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE Gradient Computation
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

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
    ):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        
        self.num_servers = cfg.system.num_servers
        self.service_rates = np.array(cfg.system.service_rates, dtype=np.float64)
        self.service_rates_jax = jnp.array(self.service_rates)
        
        # Hyperparameters are read directly from cfg in execute()
        self.batch_size = cfg.batch_size  # PI-V4.1: Support config override
        self.sim_time = cfg.simulation.ssa.sim_time
    
    def bootstrap_from_expert(self, policy_net, value_net, key):
        """Pretrain the neural network using centralized Ultra-Robust BC logic."""
        from gibbsq.core.pretraining import train_robust_bc_policy, train_robust_bc_value
        
        # 1. Train Policy (Behavior Cloning)
        policy_net = train_robust_bc_policy(
            policy_net=policy_net,
            service_rates=self.service_rates,
            key=key,
            num_steps=500,
            lr=0.002,
            weight_decay=1e-4,
            label_smoothing=0.1
        )

        # 2. Train Value (Critic Warming)
        # Initializes the baseline to expert performance level to prevent early gradient noise
        value_net, _ = train_robust_bc_value(
            value_net=value_net,
            service_rates=self.service_rates,
            key=key,
            num_steps=500,
            lr=0.002,
            weight_decay=1e-4
        )
        
        log.info(f"--- Bootstrapping Complete (Actor-Critic Warmed) ---")
        return policy_net, value_net

    def execute(self, key: PRNGKeyArray, n_epochs: int = 100):
        """Execute the REINFORCE training loop."""
        
        # Initialize networks
        key, actor_key, critic_key = jax.random.split(key, 3)
        
        policy_net = NeuralRouter(
            num_servers=self.num_servers,
            config=self.cfg.neural,
            key=actor_key,
        )
        # SHAKE WEIGHTS to break zero-init symmetry
        shake_key = jax.random.PRNGKey(self.cfg.simulation.seed + 99999)
        params, static = eqx.partition(policy_net, eqx.is_array)
        flat_params, unravel = jax.flatten_util.ravel_pytree(params)
        shake_scale = 0.01  # PI-V4.3: Refined Symmetry BreakingMag (0.1 -> 0.01)
        flat_params = flat_params + shake_scale * jax.random.normal(shake_key, flat_params.shape)
        new_params = unravel(flat_params)
        policy_net = eqx.combine(new_params, static)

        
        value_net = ValueNetwork(
            num_servers=self.num_servers,
            config=self.cfg.neural,
            hidden_size=64,
            key=critic_key,
        )
        
        # PI-V4.3: Refined Global Norm Clipping (Patch H3: 0.5 per PPO literature)
        policy_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(learning_rate=self.cfg.neural.actor_lr, weight_decay=self.cfg.neural.weight_decay)
        )
        value_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(learning_rate=self.cfg.neural.critic_lr, weight_decay=self.cfg.neural.weight_decay)
        )
        
        policy_state = policy_opt.init(eqx.filter(policy_net, eqx.is_array))
        value_state = value_opt.init(eqx.filter(value_net, eqx.is_array))
        
        # Bootstrapping (Phase 0)
        key, bootstrap_key = jax.random.split(key)
        policy_net, value_net = self.bootstrap_from_expert(policy_net, value_net, bootstrap_key)
        
        # Re-initialize states after bootstrapping
        policy_state = policy_opt.init(eqx.filter(policy_net, eqx.is_array))
        value_state = value_opt.init(eqx.filter(value_net, eqx.is_array))
        
        log.info("=" * 60)
        log.info("  REINFORCE Training (SSA-Based Policy Gradient)")
        log.info("=" * 60)
        log.info(f"  Epochs: {n_epochs}, Batch size: {self.batch_size}")
        log.info(f"  Simulation time: {self.sim_time}")
        log.info("-" * 60)
        
        history_loss = []
        history_reward = []
        
        # Periodic Checkpointing helper
        def save_checkpoint(epoch_idx):
            ckpt_path = self.run_dir / f"policy_net_epoch_{epoch_idx:03d}.eqx"
            eqx.tree_serialise_leaves(ckpt_path, policy_net)
            log.info(f"  [Checkpoint] Saved epoch {epoch_idx} model to {ckpt_path.name}")
            
            # SG#11: Update pointer in real-time for comparison scripts
            _PROJECT_ROOT = Path(__file__).resolve().parents[2]
            pointer_dir = self.run_dir.parent.parent
            pointer_dir.mkdir(parents=True, exist_ok=True)
            ptr_path = pointer_dir / "latest_reinforce_weights.txt"
            
            relative_model_path = ckpt_path.resolve().relative_to(_PROJECT_ROOT)
            with open(ptr_path, "w", encoding='utf-8') as f:
                f.write(str(relative_model_path))
        
        # --- Pre-train Benchmark ---
        log.info("Computing JSQ and Random baselines...")
        jsq_queues = []
        random_queues = []
        for _ in range(self.batch_size):
            rng = np.random.default_rng(self.cfg.simulation.seed + _)
            # 1. JSQ Benchmark
            traj_jsq = collect_trajectory_ssa(
                policy_net=policy_net, # Not used in JSQ
                num_servers=self.num_servers,
                arrival_rate=self.cfg.system.arrival_rate,
                service_rates=self.service_rates,
                sim_time=self.sim_time,
                rng=rng,
                use_jsq=True
            )
            jsq_queues.append(traj_jsq.total_integrated_queue / self.sim_time)
            
            # 2. Random Benchmark (Approximate with high-temperature logits or uniform)
            # For simplicity, we'll use the initial weights which are near-zero (Random)
            # PATCH: Pass rho explicitly so the policy uses correct load factor features
            rho = self.cfg.system.arrival_rate / np.sum(self.service_rates)
            traj_rand = collect_trajectory_ssa(
                policy_net=policy_net,
                num_servers=self.num_servers,
                arrival_rate=self.cfg.system.arrival_rate,
                service_rates=self.service_rates,
                sim_time=self.sim_time,
                rng=rng,
                use_jsq=False,
                rho=rho
            )
            random_queues.append(traj_rand.total_integrated_queue / self.sim_time)
            
        jsq_limit = np.mean(jsq_queues)
        random_queue = np.mean(random_queues) 
        
        # 2. Random Benchmark (Analytical Asymmetric M/M/1 Sum)
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
            
        # Standardize the limit used for PI scaling
        random_limit = max(jsq_limit * 1.01, analytical_random_limit, random_queue)

        log.info(f"  JSQ Mean Queue (Target): {jsq_limit:.4f}")
        log.info(f"  Random Mean Queue (Floor): {random_queue:.4f}")
        log.info(f"  Analytical Random Limit: {analytical_random_limit:.4f}")
        
        # PI-V4.3: Exponential Moving Averages for stable diagnostics
        ema_idx = 0.0
        ema_corr = 0.0
        ema_ev = 0.0
        alpha = 0.33  # ~Window 5
        
        # ZERO-TOLERANCE PATCH: Learning Rate Schedule for Actor stability
        base_actor_lr = self.cfg.neural.actor_lr
        
        for epoch in range(n_epochs):
            # Apply Adaptive Linear Decay to Actor LR (prevents unlearning)
            epoch_actor_lr = base_actor_lr * (1.0 - (epoch / n_epochs) * 0.9)
            
            # PATCH-P4: Entropy Annealing - decay from initial to final
            # Early exploration (high entropy) -> Late exploitation (low entropy)
            entropy_anneal_epochs = n_epochs  # Decay over full training
            entropy_final = 0.001  # Final entropy coefficient
            entropy_initial = self.cfg.neural.entropy_bonus
            if epoch < entropy_anneal_epochs:
                progress = epoch / entropy_anneal_epochs
                current_entropy_coef = entropy_initial * (1.0 - progress) + entropy_final * progress
            else:
                current_entropy_coef = entropy_final
            
            # Re-optimizer with decayed LR (AdamW handles the rest)
            # This is a simple way to implement decay without complex optax schedules
            policy_opt = optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adamw(learning_rate=epoch_actor_lr, weight_decay=self.cfg.neural.weight_decay)
            )
            # Save checkpoint every 25 epochs
            if epoch > 0 and epoch % 25 == 0:
                save_checkpoint(epoch)

            # Collect batch of trajectories
            trajectories = []
            epoch_rewards = []
            
            # PATCH-P2: Domain Randomization - sample rho from curriculum
            # This prevents overfitting to a single load factor
            dr_cfg = getattr(self.cfg, 'domain_randomization', None)
            if dr_cfg is not None and getattr(dr_cfg, 'enabled', False):
                rho_min = getattr(dr_cfg, 'rho_min', 0.4)
                rho_max = getattr(dr_cfg, 'rho_max', 0.95)
            else:
                # Default DR range if not configured
                rho_min = 0.4
                rho_max = 0.85
            
            for _ in range(self.batch_size):
                # Use different seed for each trajectory
                rng = np.random.default_rng(self.cfg.simulation.seed + epoch * self.batch_size + _)
                
                # PATCH-P2: Sample random rho for domain randomization
                # Vary arrival rate to sample different load conditions
                sampled_rho = rng.uniform(rho_min, rho_max)
                total_capacity = np.sum(self.service_rates)
                sampled_arrival_rate = sampled_rho * total_capacity
                
                traj = collect_trajectory_ssa(
                    policy_net=policy_net,
                    num_servers=self.num_servers,
                    arrival_rate=sampled_arrival_rate,  # PATCH-P2: Use sampled arrival rate
                    service_rates=self.service_rates,
                    sim_time=self.sim_time,
                    rng=rng,
                    rho=sampled_rho  # PATCH-P2: Pass sampled rho
                )
                trajectories.append(traj)
                epoch_rewards.append(traj.total_integrated_queue)
            
            # PI-V5: Use standardized compute_performance_index
            mean_queue = np.mean(epoch_rewards) / self.sim_time
            perf_index = compute_performance_index(mean_queue, jsq_limit, random_limit, random_queue)
            mean_reward = perf_index
            
            # Robust denominator for G_idx advantage signal
            fixed_random = max(jsq_limit * 1.01, random_limit, random_queue)
            denom = max(0.5, jsq_limit * 0.05, fixed_random - jsq_limit)
            

            all_advantages = []
            gae_lambda = getattr(self.cfg.neural_training, 'gae_lambda', 0.95)
            use_gae = gae_lambda > 0.0 and gae_lambda < 1.0
            
            for traj in trajectories:
                if use_gae:
                    # PATCH-P1: Use GAE for advantage estimation
                    G = compute_gae(
                        traj.all_states, traj.jump_times, traj.action_step_indices,
                        sim_time=self.sim_time,
                        gamma=self.cfg.neural_training.gamma,
                        gae_lambda=gae_lambda,
                        value_net=value_net,
                        service_rates=self.service_rates,
                    )
                else:
                    # Fallback: Standard returns-to-go
                    G = compute_causal_returns_to_go(
                        traj.all_states, traj.jump_times, traj.action_step_indices, 
                        sim_time=self.sim_time,
                        gamma=self.cfg.neural_training.gamma
                    )
                all_advantages.append(G)
            
            # 1. O(1) FLATTENING AND SHAPE CONTROL
            batch_S = []
            batch_A = []
            batch_G = []
            batch_Q_inst = []  # PATCH: Instantaneous queue sums for critic
            batch_rho = []
            batch_traj_indices = []
            
            for i, traj in enumerate(trajectories):
                G_raw = all_advantages[i]
                if len(G_raw) > 0:
                    # 3. PI-V3: Time-Normalized Advantage Calculation
                    # Normalize integral returns by exact remaining simulation time (T-t_k)
                    t_actions = np.array([traj.jump_times[idx] for idx in traj.action_step_indices])
                    t_rem = np.maximum(1e-3, self.sim_time - t_actions)
                    
                    G_idx = 100.0 * (random_limit * t_rem - G_raw) / (denom * t_rem)
                    
                    batch_S.extend(traj.states)
                    batch_A.extend(traj.actions)
                    batch_G.extend(np.atleast_1d(G_idx).tolist())
                    # PATCH: Collect instantaneous queue sums for critic target
                    # This aligns with pretraining which uses state-dependent queue sums
                    for state in traj.states:
                        batch_Q_inst.append(np.sum(state))
                    rho_val = self.cfg.system.arrival_rate / np.sum(self.service_rates)
                    batch_rho.extend([rho_val] * int(np.array(G_idx).size))
                    batch_traj_indices.extend([i] * int(np.array(G_idx).size))
            
            # SG#1 FIX: Initialize losses before conditional to prevent UnboundLocalError
            # when batch_G is empty (all trajectories have no actions)
            policy_loss = 0.0
            value_loss = 0.0
            avg_entropy = 0.0  # Bug #2 fix: Initialize to prevent UnboundLocalError
            policy_aux = {"entropy": 0.0, "mean_logp": 0.0, "mean_adv": 0.0}
            
            if len(batch_G) > 0:
                S_tensor = jnp.asarray(np.stack(batch_S), dtype=jnp.float32)
                A_tensor = jnp.asarray(batch_A, dtype=jnp.int32)
                G_tensor = jnp.asarray(batch_G, dtype=jnp.float32)
                Q_inst_tensor = jnp.asarray(batch_Q_inst, dtype=jnp.float32)  # PATCH: Queue sums
                rho_tensor = jnp.asarray(batch_rho, dtype=jnp.float32)
                idx_tensor = jnp.asarray(batch_traj_indices, dtype=jnp.int32)
                
                # PI-V4: PER-TRAJECTORY STANDARDIZATION (SNR Optimization)
                def local_norm_fn(G, traj_indices):
                    num_trajs = self.batch_size
                    traj_means = jax.ops.segment_sum(G, traj_indices, num_segments=num_trajs)
                    traj_counts = jax.ops.segment_sum(jnp.ones_like(G), traj_indices, num_segments=num_trajs)
                    traj_means = jnp.where(traj_counts > 0, traj_means / traj_counts, 0.0)
                    
                    G_centered = G - traj_means[traj_indices]
                    traj_vars = jax.ops.segment_sum(G_centered**2, traj_indices, num_segments=num_trajs)
                    traj_stds = jnp.sqrt(jnp.where(traj_counts > 0, traj_vars / traj_counts, 1.0) + 1e-8)
                    
                    return G_centered / traj_stds[traj_indices]

                # Target Tanh-Squashing: Prevent exploding V-loss from extreme episodes
                G_scaled = 100.0 * jnp.tanh(G_tensor / 500.0)
                G_norm = local_norm_fn(G_scaled, idx_tensor)
                
                # PATCH: Compute Q_inst_scaled for critic target
                # This aligns with pretraining which uses state-dependent queue sums
                # Critic learns V(s, rho) ≈ E[sum(queue)], not performance index
                Q_inst_scaled = 100.0 * jnp.tanh(Q_inst_tensor / 500.0)
                
                # PI-V5: PROJECT ALIGNMENT - EXPLICIT ADVANTAGE NORMALIZATION
                def advantage_norm_fn(advs):
                    # Global normalization across the batch for variance reduction
                    return (advs - jnp.mean(advs)) / (jnp.std(advs) + 1e-8)

                
                # Precompute Critic Baselines detachably (Rho-Aware)
                s_feat = sojourn_time_features(S_tensor, self.service_rates_jax)
                v_preds = jax.lax.stop_gradient(jax.vmap(value_net)(s_feat, rho_tensor))
                
                # Actor-Critic Advantage
                raw_adv = G_norm - v_preds
                norm_adv = advantage_norm_fn(raw_adv)
                
                # Acting (PI-V4.1: Include Entropy Bonus)
                def policy_loss_fn(model, s_feat, rho_feat, actions, advs, traj_indices, ent_coef):
                    logits = jax.vmap(model)(s_feat, rho_feat)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    chosen_log_probs = log_probs[jnp.arange(len(actions)), actions]
                    
                    # Policy Gradient term
                    action_signals = chosen_log_probs * advs
                    
                    # Entropy term (PI-V4.1: Numerically stable formulation)
                    probs = jax.nn.softmax(logits, axis=-1)
                    entropy = -jnp.sum(probs * log_probs, axis=-1)
                    
                    # Trajectory-Averaged Loss (Equal weight to each sampled load factor)
                    num_trajs = self.batch_size
                    traj_sums = jax.ops.segment_sum(action_signals, traj_indices, num_segments=num_trajs)
                    traj_counts = jax.ops.segment_sum(jnp.ones_like(action_signals), traj_indices, num_segments=num_trajs)
                    traj_means = traj_sums / jnp.maximum(1.0, traj_counts)
                    
                    traj_entropy = jax.ops.segment_sum(entropy, traj_indices, num_segments=num_trajs) / jnp.maximum(1.0, traj_counts)
                    
                    # Maximize (Utility + ent_coef * Entropy) -> Minimize -(Utility) - ent_coef * Entropy
                    # PATCH: Correct entropy bonus formulation
                    total_loss = -jnp.mean(traj_means) - ent_coef * jnp.mean(traj_entropy)
                    
                    # PI-V4.3: Corrected Correlation Metric [Corr(-logp, adv)]
                    def corr_fn(logp, advantages):
                        neg_logp = -logp.flatten()
                        adv_flat = advantages.flatten()
                        # Use jnp.corrcoef for standard peer-reviewed alignment check
                        return jnp.corrcoef(jnp.stack([neg_logp, adv_flat]))[0, 1]
                    
                    corr = corr_fn(chosen_log_probs, advs)
                        
                    # Critic Quality Metric
                    # PATCH: Measure EV against Q_inst_scaled (the critic's actual target)
                    # not against G_norm (the advantage target)
                    def explained_variance(y_true, y_pred):
                        var_y = jnp.var(y_true)
                        ev = 1.0 - jnp.var(y_true - y_pred) / (var_y + 1e-8)
                        return ev
                    ev = explained_variance(Q_inst_scaled, v_preds)
                    
                    mean_logp = jnp.mean(chosen_log_probs)
                    mean_adv = jnp.mean(advs)

                    return total_loss, {
                        "entropy": jnp.mean(traj_entropy),
                        "mean_logp": mean_logp,
                        "mean_adv": mean_adv,
                        "corr": corr,
                        "ev": ev
                    }

                (policy_loss, policy_aux), policy_grads = eqx.filter_value_and_grad(policy_loss_fn, has_aux=True)(
                    policy_net, s_feat, rho_tensor, A_tensor, norm_adv, idx_tensor, current_entropy_coef
                )
                # Calculate Actor Gradient Norm
                policy_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(policy_grads) if g is not None))
                
                updates, policy_state = policy_opt.update(
                    policy_grads, policy_state, eqx.filter(policy_net, eqx.is_array)
                )
                policy_net = eqx.apply_updates(policy_net, updates)
                
                # 4. PURE VECTORIZED CRITIC GRAPH
                # PATCH: Use Q_inst_scaled for critic loss (not G_norm)
                # This aligns with pretraining which trains on state-dependent queue sums
                def value_loss_fn(model, s_feat, rho_feat, queue_targets):
                    preds = jax.vmap(model)(s_feat, rho_feat)
                    return jnp.mean((preds - queue_targets) ** 2)

                value_loss, value_grads = eqx.filter_value_and_grad(value_loss_fn)(
                    value_net, s_feat, rho_tensor, Q_inst_scaled
                )
                
                # Calculate Critic Gradient Norm
                value_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(value_grads) if g is not None))
                
                updates, value_state = value_opt.update(
                    value_grads, value_state, eqx.filter(value_net, eqx.is_array)
                )
                value_net = eqx.apply_updates(value_net, updates)
                
                # Diagnostic Alignment Check (Sign Test)
                if epoch % 10 == 0:
                    avg_ent = policy_aux["entropy"]
                    mean_logp = policy_aux["mean_logp"]
                    mean_adv = policy_aux["mean_adv"]
                    corr = policy_aux.get("corr", 0.0)
                    log.info(f"    [Sign Check] mean_adv: {mean_adv:.4f} | mean_loss: {float(policy_loss):.4f} | mean_logp: {mean_logp:.4f} | corr: {corr:.4f}")
                    log.info(f"    [Grad Check] P-Grad Norm: {policy_grad_norm:.4f} | V-Grad Norm: {value_grad_norm:.4f}")
            
            # Update EMAs
            if epoch == 0:
                ema_idx, ema_corr, ema_ev = mean_reward, float(policy_aux.get("corr", 0.0)), float(policy_aux.get("ev", 0.0))
            else:
                ema_idx = alpha * mean_reward + (1 - alpha) * ema_idx
                ema_corr = alpha * float(policy_aux.get("corr", 0.0)) + (1 - alpha) * ema_corr
                ema_ev = alpha * float(policy_aux.get("ev", 0.0)) + (1 - alpha) * ema_ev
            
            # Record metrics (Using Positive Utility)
            history_loss.append(float(policy_loss))
            history_reward.append(mean_reward)
            # Log progress
            if epoch % 5 == 0:
                log.info(f"Epoch {epoch:4d} | mean_reward: {mean_reward:5.1f}% (EMA: {ema_idx:5.1f}%) | "
                        f"Loss: {float(policy_loss):.4f} | V-Loss: {float(value_loss):.4f}")
                log.info(f"   -> Signaling | EV: {ema_ev:6.3f} [EMA], Corr: {ema_corr:6.4f} [EMA]")
            
            metrics = {
                "epoch": epoch,
                "policy_loss": float(policy_loss),
                "value_loss": float(value_loss),
                "mean_reward": float(mean_reward), # Restored label
                "performance_index": float(perf_index),
                "performance_index_ema": float(ema_idx),
                "mean_queue": float(mean_queue),
                "ev_ema": float(ema_ev),
                "corr_ema": float(ema_corr),
                "jsq_limit": float(jsq_limit),
                "random_analytical": float(random_limit),
                "random_empirical": float(random_queue),
                "arrival_count": int(np.mean([t.arrival_count for t in trajectories])),
                # Added diagnostics for research alignment
                "mean_adv": float(policy_aux.get("mean_adv", 0.0)),
                "mean_logp": float(policy_aux.get("mean_logp", 0.0)),
                "policy_grad_norm": float(policy_grad_norm) if 'policy_grad_norm' in locals() else 0.0,
                "value_grad_norm": float(value_grad_norm) if 'value_grad_norm' in locals() else 0.0,
                "entropy": float(policy_aux.get("entropy", 0.0)),
            }
            append_metrics_jsonl(metrics, self.run_dir / "reinforce_metrics.jsonl")
            if self.run_logger:
                self.run_logger.log(metrics)
        
        # Save model and evaluate
        self._save_assets(policy_net, value_net, history_loss, history_reward, jsq_limit, random_limit, random_queue)
    
    def _save_assets(
        self,
        policy_net: NeuralRouter,
        value_net: ValueNetwork,
        history_loss: list,
        history_reward: list,
        jsq_limit: float,
        random_limit: float,
        random_queue: float,
    ):
        """Persist model weights and training history."""
        import matplotlib.pyplot as plt
        
        # Save weights
        policy_path = self.run_dir / "n_gibbsq_reinforce_weights.eqx"
        eqx.tree_serialise_leaves(policy_path, policy_net)
        
        value_path = self.run_dir / "value_network_weights.eqx"
        eqx.tree_serialise_leaves(value_path, value_net)
        
        # Plot training curve
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(history_loss, color='blue', linewidth=2)
        axes[0].set_title('REINFORCE Policy Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history_reward, color='green', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title('Mean Reward (-E[Q])')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reward')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.run_dir / "reinforce_training_curve.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        # Write pointer
        # SG#4 FIX: Write to output_dir from config instead of hardcoded "outputs/small"
        # This ensures policy_comparison.py can find the weights pointer.
        # run_dir = output_dir / experiment_type / run_id, so run_dir.parent.parent = output_dir
        pointer_dir = self.run_dir.parent.parent
        pointer_dir.mkdir(parents=True, exist_ok=True)
        pointer_path = pointer_dir / "latest_reinforce_weights.txt"
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
        _relative_model_path = policy_path.resolve().relative_to(_PROJECT_ROOT)
        pointer_path.write_text(str(_relative_model_path), encoding='utf-8')
        
        log.info("-" * 55)
        log.info("-------------------------------------------------------")
        log.info("Running Final Deterministic Evaluation (N=50)...")
        eval_indices = []
        for b in range(5):
            eval_trajs = []
            for i in range(10):
                # Use a fresh seed for each evaluation trajectory
                eval_rng = np.random.default_rng(self.cfg.simulation.seed + 10000 + b*10 + i)
                t = collect_trajectory_ssa(
                    policy_net=policy_net,
                    num_servers=self.num_servers,
                    arrival_rate=self.cfg.system.arrival_rate,
                    service_rates=self.service_rates,
                    sim_time=self.sim_time,
                    rng=eval_rng,
                    rho=self.cfg.system.arrival_rate / np.sum(self.service_rates),
                    deterministic=True
                )
                eval_trajs.append(t)
            
            batch_indices = []
            for t in eval_trajs:
                p_idx = compute_performance_index(
                    t.total_integrated_queue / self.sim_time, 
                    jsq_limit, 
                    random_limit, 
                    random_queue
                )
                batch_indices.append(p_idx)
            eval_indices.extend(batch_indices)
        
        final_mean = np.mean(eval_indices)
        final_std = np.std(eval_indices)
        log.info(f"Deterministic Policy Score: {final_mean:.2f}% ± {final_std:.2f}%")
        log.info(f"JSQ Target: 100.0% | Random Floor: 0.0% (Performance Index Scale)")
        log.info("-------------------------------------------------------")
        
        log.info(f"Training Complete! Final Loss: {history_loss[-1]:.4f}")
        log.info(f"Final Reward: {history_reward[-1]:.2f}")
        log.info(f"Policy weights: {policy_path}")
        log.info(f"Value weights: {value_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main(raw_cfg: DictConfig):
    """Main entry point for REINFORCE training."""
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)
    
    run_dir, run_id = get_run_config(cfg, "reinforce_training", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="reinforce_training", 
                            run_id=run_id, run_dir=run_dir)
    
    trainer = ReinforceTrainer(cfg, run_dir, run_logger)
    
    seed_key = jax.random.PRNGKey(cfg.simulation.seed)
    trainer.execute(seed_key, n_epochs=cfg.train_epochs)


if __name__ == "__main__":
    import sys
    import hydra

    if len(sys.argv) > 1:
        # Standard Hydra run
        hydra.main(version_base=None, config_path="../../configs", config_name="default")(main)()
    else:
        # Notebook / Manual run
        from hydra import compose, initialize_config_dir
        import os
        config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
        if os.path.exists(config_dir):
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                raw_cfg = compose(config_name="default")
                main(raw_cfg)
        else:
            print(f"Error: Config directory not found at {config_dir}")
