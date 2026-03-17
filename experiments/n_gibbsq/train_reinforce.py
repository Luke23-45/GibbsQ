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
from omegaconf import DictConfig

from gibbsq.core.config import ExperimentConfig, NeuralConfig, hydra_to_config, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.features import sojourn_time_features
from gibbsq.core import constants
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


# ─────────────────────────────────────────────────────────────────────────────
# Value Network (Critic) for Baseline Estimation
# ─────────────────────────────────────────────────────────────────────────────

class ValueNetwork(eqx.Module):
    """
    MLP value function approximator for baseline estimation.
    
    The value network V(s) estimates the expected total queue length
    from state s, used as a baseline in REINFORCE to reduce variance:
    
        ∇_θ J(θ) = E_π [ (R(τ) - V(s)) · ∇_θ log π_θ(a|s) ]
    
    This is the Actor-Critic framework, which provides lower-variance
    gradient estimates than vanilla REINFORCE.
    """
    layers: list[eqx.Module]
    
    def __init__(
        self,
        num_servers: int,
        hidden_size: int = 64,
        key: PRNGKeyArray = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(num_servers, hidden_size, key=keys[0]),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]),
            eqx.nn.Linear(hidden_size, 1, key=keys[2]),
        ]
    
    def __call__(self, s: Float[Array, "dim"]) -> Float[Array, ""]:
        """Forward pass. Returns scalar value estimate V(s)."""
        x = s
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x).squeeze()


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
        # 1. Compute routing probabilities using sojourn-time features
        # Use fast NumPy forward pass
        s = (Q + 1.0) / mu
        logits = policy_net.numpy_forward(s, np_params, np_config)
        
        # Log-sum-exp for numerical stability
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        
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

# The old episodic compute_reinforce_loss was removed and replaced by the 
# Step-Wise Causal PG loss inside the ReinforceTrainer.execute loop.


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
        
        # Training hyperparameters from professor's spec
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4
        self.entropy_bonus = 0.01
        self.batch_size = 8  # Number of parallel trajectories
        self.sim_time = cfg.simulation.ssa.sim_time
    
    def execute(self, key: PRNGKeyArray, n_epochs: int = 100):
        """Execute the REINFORCE training loop."""
        
        # Initialize networks
        key, actor_key, critic_key = jax.random.split(key, 3)
        
        policy_net = NeuralRouter(
            num_servers=self.num_servers,
            config=self.cfg.neural,
            key=actor_key,
        )
        
        value_net = ValueNetwork(
            num_servers=self.num_servers,
            hidden_size=64,
            key=critic_key,
        )
        
        # Optimizers
        policy_opt = optax.adamw(self.learning_rate, weight_decay=self.weight_decay)
        value_opt = optax.adamw(self.learning_rate, weight_decay=self.weight_decay)
        
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
        
        for epoch in range(n_epochs):
            # Collect batch of trajectories
            trajectories = []
            epoch_rewards = []
            
            for _ in range(self.batch_size):
                # Use different seed for each trajectory
                rng = np.random.default_rng(self.cfg.simulation.seed + epoch * self.batch_size + _)
                
                traj = collect_trajectory_ssa(
                    policy_net=policy_net,
                    num_servers=self.num_servers,
                    arrival_rate=self.cfg.system.arrival_rate,
                    service_rates=self.service_rates,
                    sim_time=self.sim_time,
                    rng=rng,
                )
                trajectories.append(traj)
                epoch_rewards.append(traj.total_integrated_queue)  # Reward = Total Cost
            
            # Compute mean reward for logging (negate because we track -Cost)
            mean_reward = -np.mean(epoch_rewards)
            all_returns_to_go = []
            for traj in trajectories:
                # Patch 1/2: Calculate G_k for each action in the trajectory
                G = compute_causal_returns_to_go(
                    traj.all_states, traj.jump_times, traj.action_step_indices, 
                    sim_time=self.sim_time,
                    gamma=self.cfg.neural_training.gamma
                )
                all_returns_to_go.append(G)
            
            # 1. O(1) FLATTENING AND SHAPE CONTROL
            batch_S = []
            batch_A = []
            batch_G =[]
            
            for i, traj in enumerate(trajectories):
                G = all_returns_to_go[i]
                if len(G) > 0:
                    batch_S.extend(traj.states)
                    batch_A.extend(traj.actions)
                    batch_G.extend(G)
            
            # SG#1 FIX: Initialize losses before conditional to prevent UnboundLocalError
            # when batch_G is empty (all trajectories have no actions)
            policy_loss = 0.0
            value_loss = 0.0
            
            if len(batch_G) > 0:
                S_tensor = jnp.asarray(np.stack(batch_S), dtype=jnp.float32)
                A_tensor = jnp.asarray(batch_A, dtype=jnp.int32)
                G_tensor = jnp.asarray(batch_G, dtype=jnp.float32)
                
                # SG#10 FIX: Normalize G before both advantage computation AND critic targets.
                # v_preds should learn E[G_norm | s] ∈ [-3, +3] for rho-invariant stability.
                G_mean = jnp.mean(G_tensor)
                G_std = jnp.std(G_tensor) + 1e-8
                G_norm = (G_tensor - G_mean) / G_std
                
                # Precompute Critic Baselines detachably
                s_feat = (S_tensor + 1.0) / self.service_rates_jax
                v_preds = jax.lax.stop_gradient(jax.vmap(value_net)(s_feat))
                
                # 2. ACTOR-CRITIC ADVANTAGE NORMALIZATION (CRITICAL FOR NaN STABILITY)
                # After patch, norm_adv is still zero-mean relative to the normalized returns.
                norm_adv = G_norm - v_preds
                
                # 3. PURE, FAST, VECTORIZED ACTOR GRAPH 
                # FIX: Use positive sum for cost minimization. Gradient descent on +log_prob * adv
                # gives correct direction: decrease log_prob when adv > 0 (high cost action).
                def policy_loss_fn(model, s_feat, actions, advs):
                    logits = jax.vmap(model)(s_feat)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    chosen_log_probs = log_probs[jnp.arange(len(actions)), actions]
                    return jnp.sum(chosen_log_probs * advs) / max(1.0, float(len(trajectories)))

                policy_loss, policy_grads = eqx.filter_value_and_grad(policy_loss_fn)(
                    policy_net, s_feat, A_tensor, norm_adv
                )
                updates, policy_state = policy_opt.update(
                    policy_grads, policy_state, eqx.filter(policy_net, eqx.is_array)
                )
                policy_net = eqx.apply_updates(policy_net, updates)
                
                # 4. PURE VECTORIZED CRITIC GRAPH
                def value_loss_fn(model, s_feat, returns_to_go):
                    preds = jax.vmap(model)(s_feat)
                    return jnp.mean((preds - returns_to_go) ** 2)

                value_loss, value_grads = eqx.filter_value_and_grad(value_loss_fn)(
                    value_net, s_feat, G_norm
                )
                updates, value_state = value_opt.update(
                    value_grads, value_state, eqx.filter(value_net, eqx.is_array)
                )
                value_net = eqx.apply_updates(value_net, updates)
            
            # Record metrics (Negate total_integrated_queue so Reward increases as Cost decreases)
            mean_reward = -np.mean(epoch_rewards)
            history_loss.append(float(policy_loss))
            history_reward.append(mean_reward)
            
            if epoch % 10 == 0:
                log.info(f"Epoch {epoch:4d} | Loss: {float(policy_loss):.4f} | "
                        f"Reward: {mean_reward:.2f}")
            
            metrics = {
                "epoch": epoch,
                "policy_loss": float(policy_loss),
                "value_loss": float(value_loss),
                "mean_reward": mean_reward,
            }
            append_metrics_jsonl(metrics, self.run_dir / "reinforce_metrics.jsonl")
            
            if self.run_logger:
                self.run_logger.log(metrics)
        
        # Save model
        self._save_assets(policy_net, value_net, history_loss, history_reward)
    
    def _save_assets(
        self,
        policy_net: NeuralRouter,
        value_net: ValueNetwork,
        history_loss: list,
        history_reward: list,
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
        # This ensures corrected_policy_comparison.py can find the weights pointer.
        # run_dir = output_dir / experiment_type / run_id, so run_dir.parent.parent = output_dir
        pointer_dir = self.run_dir.parent.parent
        pointer_dir.mkdir(parents=True, exist_ok=True)
        pointer_path = pointer_dir / "latest_reinforce_weights.txt"
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
        _relative_model_path = policy_path.resolve().relative_to(_PROJECT_ROOT)
        pointer_path.write_text(str(_relative_model_path), encoding='utf-8')
        
        log.info("-" * 55)
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
    from omegaconf import OmegaConf
    import hydra
    from hydra import compose, initialize_config_dir
    import os
    
    # Allow running without Hydra config
    import sys
    if len(sys.argv) > 1:
        # With Hydra
        hydra.main(version_base=None, config_path="../../configs", config_name="default")(main)()
    else:
        # Without Hydra - use default config
        config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raw_cfg = compose(config_name="default")
            main(raw_cfg)
