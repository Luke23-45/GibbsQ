"""
Behavior Cloning (BC) utilities.

This module centralizes the pretraining logic for N-GibbsQ:
1. Steady-State Expert Sampling
2. Noisy State Augmentation
3. Label Smoothing (CE Loss)
4. AdamW with Weight Decay
"""

import logging
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from jaxtyping import PRNGKeyArray

from gibbsq.core.config import ExperimentConfig
from gibbsq.core.neural_policies import NeuralRouter, ValueNetwork
from gibbsq.core.features import look_ahead_potential
from gibbsq.core.policies import UASRouting
from gibbsq.engines.numpy_engine import simulate
from gibbsq.utils.progress import create_progress

log = logging.getLogger(__name__)

DEFAULT_BC_DATA_CONFIG = {
    "rhos": [0.45, 0.65, 0.85],
    "mu_scales": [0.5, 1.0, 2.0],
    "samples_per_rho": 1000,
    "expert_sim_time": 1500.0,
    "sample_interval": 1.0,
    "augmentation_noise_min": -1,
    "augmentation_noise_max": 1,
}


def normalize_bc_data_config(
    bc_data_config: dict | None = None,
    *,
    rhos: list[float] | None = None,
    samples_per_rho: int | None = None,
) -> dict:
    """Return validated BC expert-data generation settings."""
    cfg = dict(DEFAULT_BC_DATA_CONFIG)
    if bc_data_config:
        cfg.update(dict(bc_data_config))
    if rhos is not None:
        cfg["rhos"] = list(rhos)
    if samples_per_rho is not None:
        cfg["samples_per_rho"] = int(samples_per_rho)

    rho_values = [float(x) for x in cfg["rhos"]]
    if not rho_values:
        raise ValueError("bc_data.rhos must contain at least one load factor.")
    if not all(0.0 < rho < 1.0 for rho in rho_values):
        raise ValueError(f"bc_data.rhos must lie in (0, 1), got {rho_values}")

    mu_scales = [float(x) for x in cfg["mu_scales"]]
    if not mu_scales:
        raise ValueError("bc_data.mu_scales must contain at least one multiplier.")
    if not all(scale > 0.0 for scale in mu_scales):
        raise ValueError(f"bc_data.mu_scales must be positive, got {mu_scales}")

    samples = int(cfg["samples_per_rho"])
    if samples < 1:
        raise ValueError(f"bc_data.samples_per_rho must be >= 1, got {samples}")

    expert_sim_time = float(cfg["expert_sim_time"])
    if expert_sim_time <= 0.0:
        raise ValueError(f"bc_data.expert_sim_time must be > 0, got {expert_sim_time}")

    sample_interval = float(cfg["sample_interval"])
    if sample_interval <= 0.0:
        raise ValueError(f"bc_data.sample_interval must be > 0, got {sample_interval}")

    noise_min = int(cfg["augmentation_noise_min"])
    noise_max = int(cfg["augmentation_noise_max"])
    if noise_min > noise_max:
        raise ValueError(
            "bc_data.augmentation_noise_min must be <= augmentation_noise_max, "
            f"got [{noise_min}, {noise_max}]"
        )

    cfg["rhos"] = rho_values
    cfg["mu_scales"] = mu_scales
    cfg["samples_per_rho"] = samples
    cfg["expert_sim_time"] = expert_sim_time
    cfg["sample_interval"] = sample_interval
    cfg["augmentation_noise_min"] = noise_min
    cfg["augmentation_noise_max"] = noise_max
    return cfg


def extract_bc_data_config(raw_cfg) -> dict:
    """Extract BC data settings from a raw Hydra/OmegaConf config."""
    if raw_cfg is None:
        return dict(DEFAULT_BC_DATA_CONFIG)
    if hasattr(raw_cfg, "get"):
        candidate = raw_cfg.get("bc_data")
    else:
        candidate = None
    if candidate is None:
        return dict(DEFAULT_BC_DATA_CONFIG)
    if hasattr(candidate, "items"):
        return normalize_bc_data_config(dict(candidate))
    raise TypeError(f"Unsupported bc_data config type: {type(candidate)}")

def compute_value_bootstrap_targets(
    queue_totals: jnp.ndarray,
    random_limit: float,
    denom: float,
) -> jnp.ndarray:
    """Map queue totals onto the same linear PI scale used online."""
    safe_denom = jnp.maximum(jnp.asarray(denom, dtype=jnp.float32), 1e-6)
    return 100.0 * (jnp.asarray(random_limit, dtype=jnp.float32) - queue_totals) / safe_denom

def collect_robust_expert_data(
    num_servers: int,
    service_rates: np.ndarray,
    rhos: list[float] | None = None,
    samples_per_rho: int | None = None,
    seed: int = 42,
    alpha: float = 1.0,
    bc_data_config: dict | None = None,
):
    """Collects steady-state expert data with noisy state augmentation."""
    log.info(f"--- Collecting Robust Expert Data (Steady-State + Augmentation) ---")
    rng = np.random.default_rng(seed)
    bc_cfg = normalize_bc_data_config(
        bc_data_config,
        rhos=rhos,
        samples_per_rho=samples_per_rho,
    )
    
    all_states = []
    all_rhos = []
    all_probs = []
    all_q_targets = []
    all_mus = []
    
    mu_scales = bc_cfg["mu_scales"]
    samples_per_variant = max(100, int(bc_cfg["samples_per_rho"]) // len(mu_scales))
    
    for mu_scale in mu_scales:
        scaled_service_rates = service_rates * mu_scale
        expert = UASRouting(scaled_service_rates, alpha=alpha)
        total_capacity = np.sum(scaled_service_rates)
        
        for rho in bc_cfg["rhos"]:
            arrival_rate = rho * total_capacity
            res = simulate(
                num_servers=num_servers,
                arrival_rate=arrival_rate,
                service_rates=scaled_service_rates,
                policy=expert,
                sim_time=bc_cfg["expert_sim_time"],
                sample_interval=bc_cfg["sample_interval"],
                rng=rng
            )
            states = res.states[len(res.states)//2:]
            
            if len(states) > samples_per_variant:
                indices = rng.choice(len(states), size=samples_per_variant, replace=False)
                states = states[indices]
                
            for s in states:
                all_states.append(s)
                all_rhos.append(rho)
                all_probs.append(expert(s, rng))
                all_q_targets.append(np.sum(s))
                all_mus.append(scaled_service_rates)
                
                noise = rng.integers(
                    bc_cfg["augmentation_noise_min"],
                    bc_cfg["augmentation_noise_max"] + 1,
                    size=num_servers,
                )
                s_aug = np.maximum(0, s + noise)
                all_states.append(s_aug)
                all_rhos.append(rho)
                all_probs.append(expert(s_aug, rng))
                all_q_targets.append(np.sum(s_aug))
                all_mus.append(scaled_service_rates)
                
    return (jnp.array(all_states, dtype=jnp.float32), 
            jnp.array(all_rhos, dtype=jnp.float32), 
            jnp.array(all_probs, dtype=jnp.float32),
            jnp.array(all_q_targets, dtype=jnp.float32),
            jnp.array(all_mus, dtype=jnp.float32))

def train_robust_bc_policy(
    policy_net: NeuralRouter,
    service_rates: np.ndarray,
    key: PRNGKeyArray,
    num_steps: int = 500,
    lr: float = 0.002,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    entropy_bonus: float = 0.01,
    seed: int = 42,
    alpha: float = 1.0,
    bc_data_config: dict | None = None,
):
    """Trains a neural policy using Ultra-Robust BC logic."""
    num_servers = policy_net.layers[-1].weight.shape[0]
    
    X, R, Y, G, MU = collect_robust_expert_data(
        num_servers=num_servers,
        service_rates=service_rates,
        seed=seed,
        alpha=alpha,
        bc_data_config=bc_data_config,
    )
    
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(policy_net, eqx.is_array))
    
    @eqx.filter_jit
    def loss_fn(model, x, r, target_probs, mu_batch):
        logits = jax.vmap(lambda q, m, rh: model(q, mu=m, rho=rh))(x, mu_batch, r)
        soft_labels = target_probs * (1.0 - label_smoothing) + (label_smoothing / num_servers)
        
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce_loss = -jnp.mean(jnp.sum(soft_labels * log_probs, axis=-1))
        
        probs = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        ent_loss = -entropy_bonus * jnp.mean(entropy)
        
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(target_probs, axis=-1))
        return ce_loss + ent_loss, acc

    @eqx.filter_jit
    def step(model, opt_state, x, r, target_probs, mu_batch):
        (loss, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, x, r, target_probs, mu_batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, acc

    log.info(f"--- Bootstrapping Actor (Behavior Cloning) ---")
    with create_progress(total=num_steps + 1, desc="bc_train", unit="step") as progress:
        for i in range(num_steps + 1):
            policy_net, opt_state, loss, acc = step(policy_net, opt_state, X, R, Y, MU)
            progress.update(1)
            progress.set_postfix(
                {"loss": f"{float(loss):.4f}", "acc": f"{float(acc):.2%}"},
                refresh=False,
            )
            if i % 100 == 0:
                log.info(f"  Step {i:4d} | Loss: {loss:.4f} | Acc: {acc:.2%}")
            
    return policy_net

def train_robust_bc_value(
    value_net: ValueNetwork,
    service_rates: np.ndarray,
    key: PRNGKeyArray,
    num_steps: int = 500,
    lr: float = 0.002,
    weight_decay: float = 1e-4,
    jsq_limit: float = 1.0,
    random_limit: float = 2.0,
    denom: float = 1.0,
    squash_scale: float = 100.0,
    squash_threshold: float = 100.0,
    seed: int = 42,
    alpha: float = 1.0,
    bc_data_config: dict | None = None,
):
    """Trains a value network (critic) to estimate steady-state queue lengths."""
    num_servers = len(service_rates)
    
    X, R, Y, G, MU = collect_robust_expert_data(
        num_servers=num_servers,
        service_rates=service_rates,
        seed=seed,
        alpha=alpha,
        bc_data_config=bc_data_config,
    )
    
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(value_net, eqx.is_array))

    G_scaled = compute_value_bootstrap_targets(
        queue_totals=G,
        random_limit=random_limit,
        denom=denom,
    )

    @eqx.filter_jit
    def loss_fn(model, x, r, g, mu_batch):
        preds = jax.vmap(lambda q, m, rh: model(q, mu=m, rho=rh))(x, mu_batch, r)
        loss = jnp.mean((preds - g)**2)
        return loss

    @eqx.filter_jit
    def step(model, opt_state, x, r, g, mu_batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, r, g, mu_batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    log.info(f"--- Bootstrapping Critic (Value Warming) ---")
    with create_progress(total=num_steps + 1, desc="bc_value", unit="step") as progress:
        for i in range(num_steps + 1):
            value_net, opt_state, loss = step(value_net, opt_state, X, R, G_scaled, MU)
            progress.update(1)
            progress.set_postfix({"loss": f"{float(loss):.4f}"}, refresh=False)
            if i % 100 == 0:
                log.info(f"  Step {i:4d} | MSE Loss: {loss:.4f}")
            
    return value_net, opt_state
