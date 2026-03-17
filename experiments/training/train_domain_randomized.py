"""
Domain Randomization Training for N-GibbsQ.

This module implements the corrected training pipeline with domain randomization,
which exposes the neural network to diverse load conditions during training.

The key insight: Training on low-load regimes (rho < 0.4) produces policies that
cannot handle critical load conditions. Domain randomization samples load factors
from a distribution [rho_min, rho_max] during training, creating a robust policy.

This is Track 3 of the corrective research plan.
"""

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig

from gibbsq.core.config import ExperimentConfig, hydra_to_config, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.features import sojourn_time_features
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl

log = logging.getLogger(__name__)


class DomainRandomizedTrainer:
    """
    REINFORCE training with domain randomization.
    
    The training samples arrival rates from a distribution that spans
    multiple load conditions, creating a policy that generalizes across
    the entire operating envelope.
    
    Key differences from standard curriculum training:
    1. Load factor rho is sampled per-trajectory from [rho_min, rho_max]
    2. Curriculum phases expand the load range rather than shifting horizon
    3. Final policy is evaluated across the full load spectrum
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
        self.total_capacity = float(np.sum(self.service_rates))
        
        # Training hyperparameters
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4
        self.batch_size = 8
        self.n_epochs = 100
        
        # Domain randomization phases
        # Professor's spec: Start at [0.45, 0.70], expand to [0.60, 0.95]
        self.phases = [
            {"rho_min": 0.45, "rho_max": 0.70, "epochs": 20, "horizon": 500},
            {"rho_min": 0.50, "rho_max": 0.85, "epochs": 30, "horizon": 2000},
            {"rho_min": 0.60, "rho_max": 0.95, "epochs": 50, "horizon": 5000},
        ]
    
    def sample_load_factor(self, rho_min: float, rho_max: float, rng: np.random.Generator) -> float:
        """Sample a load factor uniformly from [rho_min, rho_max]."""
        return rng.uniform(rho_min, rho_max)
    
    def compute_arrival_rate(self, rho: float) -> float:
        """Convert load factor to arrival rate: lambda = rho * Lambda."""
        return rho * self.total_capacity
    
    def execute(self, key: PRNGKeyArray):
        """Execute domain-randomized training."""
        from experiments.training.train_reinforce import (
            collect_trajectory_ssa, ValueNetwork, compute_causal_returns_to_go
        )
        
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
        
        # Optimizers (AdamW with weight decay per professor's spec)
        policy_opt = optax.adamw(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        value_opt = optax.adamw(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        
        policy_state = policy_opt.init(eqx.filter(policy_net, eqx.is_array))
        value_state = value_opt.init(eqx.filter(value_net, eqx.is_array))
        
        log.info("=" * 60)
        log.info("  Domain Randomized REINFORCE Training")
        log.info("=" * 60)
        log.info(f"  Phases: {len(self.phases)}")
        log.info(f"  Batch size: {self.batch_size}")
        log.info("-" * 60)
        
        history_loss = []
        history_reward = []
        history_rho = []
        
        global_epoch = 0
        
        for phase_idx, phase in enumerate(self.phases):
            rho_min = phase["rho_min"]
            rho_max = phase["rho_max"]
            n_epochs = phase["epochs"]
            sim_time = phase["horizon"]
            
            log.info(f"\nPhase {phase_idx + 1}: rho in [{rho_min:.2f}, {rho_max:.2f}], "
                    f"T={sim_time}, epochs={n_epochs}")
            log.info("-" * 40)
            
            for epoch in range(n_epochs):
                # Optimization: Extract NumPy parameters once per batch
                np_params = policy_net.get_numpy_params()
                np_config = policy_net.config
                
                # Collect batch with randomized load factors
                trajectories = []
                epoch_rewards = []
                epoch_rhos = []
                
                for b in range(self.batch_size):
                    # Sample load factor for this trajectory
                    rng = np.random.default_rng(
                        int(self.cfg.simulation.seed) + int(global_epoch) * self.batch_size + int(b)
                    )
                    rho = self.sample_load_factor(rho_min, rho_max, rng)
                    arrival_rate = self.compute_arrival_rate(rho)
                    epoch_rhos.append(rho)
                    
                    # Collect trajectory
                    traj = collect_trajectory_ssa(
                        policy_net=policy_net,
                        num_servers=self.num_servers,
                        arrival_rate=arrival_rate,
                        service_rates=self.service_rates,
                        sim_time=sim_time,
                        rng=rng,
                    )
                    trajectories.append(traj)
                    epoch_rewards.append(-traj.total_integrated_queue)
                
                all_returns_to_go = []
                for traj in trajectories:
                    G = compute_causal_returns_to_go(
                        traj.all_states, traj.jump_times, traj.action_step_indices, 
                        sim_time=sim_time,
                        gamma=self.cfg.neural_training.gamma
                    )
                    all_returns_to_go.append(G)
                
                mean_rho = np.mean(epoch_rhos)

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
                    
                    # SG#10b FIX: Normalize G before both advantage computation AND critic targets.
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
                
                # Record metrics
                mean_reward = np.mean(epoch_rewards)
                history_loss.append(float(policy_loss))
                history_reward.append(mean_reward)
                history_rho.append(mean_rho)
                
                if epoch % 5 == 0:
                    log.info(f"Epoch {global_epoch:4d} | rho={mean_rho:.3f} | "
                            f"Loss: {float(policy_loss):.4f} | Reward: {mean_reward:.2f}")
                
                metrics = {
                    "global_epoch": global_epoch,
                    "phase": phase_idx,
                    "phase_epoch": epoch,
                    "rho": mean_rho,
                    "policy_loss": float(policy_loss),
                    "mean_reward": mean_reward,
                }
                append_metrics_jsonl(metrics, self.run_dir / "domain_randomized_metrics.jsonl")
                
                if self.run_logger:
                    self.run_logger.log(metrics)
                
                global_epoch += 1
        
        # Save model
        self._save_assets(policy_net, value_net, history_loss, history_reward, history_rho)
    
    def _save_assets(self, policy_net, value_net, history_loss, history_reward, history_rho):
        """Persist model weights and training history."""
        import matplotlib.pyplot as plt
        
        # Save weights
        policy_path = self.run_dir / "n_gibbsq_domain_randomized_weights.eqx"
        eqx.tree_serialise_leaves(policy_path, policy_net)
        
        value_path = self.run_dir / "value_network_dr_weights.eqx"
        eqx.tree_serialise_leaves(value_path, value_net)
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(history_loss, color='blue', linewidth=2)
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(history_reward, color='green', linewidth=2)
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Mean Reward (-E[Q])')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(history_rho, color='orange', linewidth=2)
        axes[1, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target rho=0.8')
        axes[1, 0].set_title('Load Factor rho')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('rho')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(history_rho, history_reward, alpha=0.3, s=10)
        axes[1, 1].set_title('Reward vs Load Factor')
        axes[1, 1].set_xlabel('rho')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.run_dir / "domain_randomized_training.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        # Write pointer
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
        # Align pointer location with configured output_dir layout used by other tracks
        # run_dir = output_dir / experiment_type / run_id, so run_dir.parent.parent = output_dir
        pointer_dir = self.run_dir.parent.parent
        pointer_dir.mkdir(parents=True, exist_ok=True)
        # SG#11 FIX: Write both pointers so comparison scripts find the DR model
        _relative_model_path = policy_path.resolve().relative_to(_PROJECT_ROOT)
        
        # 1. DR-specific pointer
        dr_pointer = pointer_dir / "latest_domain_randomized_weights.txt"
        dr_pointer.write_text(str(_relative_model_path), encoding='utf-8')
        
        # 2. Standard pointer (Track 3 usually overwrites Track 1 for evaluation)
        std_pointer = pointer_dir / "latest_reinforce_weights.txt"
        std_pointer.write_text(str(_relative_model_path), encoding='utf-8')
        
        log.info("-" * 55)
        log.info(f"Training Complete! Final Loss: {history_loss[-1]:.4f}")
        log.info(f"Final Reward: {history_reward[-1]:.2f}")
        log.info(f"Policy weights: {policy_path}")


def main(raw_cfg: DictConfig):
    """Main entry point for domain randomized training."""
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)
    
    run_dir, run_id = get_run_config(cfg, "domain_randomized_training", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="domain_randomized_training",
                            run_id=run_id, run_dir=run_dir)
    
    trainer = DomainRandomizedTrainer(cfg, run_dir, run_logger)
    
    seed_key = jax.random.PRNGKey(cfg.simulation.seed)
    trainer.execute(seed_key)


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
