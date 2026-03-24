"""
Ultra-Robust Behavior Cloning (BC) Utilities.

This module centralizes the foundational pretraining logic for N-GibbsQ,
incorporating Platinum-grade robustness patches:
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
from gibbsq.core.features import sojourn_time_features
from gibbsq.core.policies import SojournTimeSoftmaxRouting
from gibbsq.engines.numpy_engine import simulate

log = logging.getLogger(__name__)

def collect_robust_expert_data(
    num_servers: int,
    service_rates: np.ndarray,
    rhos: list[float] = [0.45, 0.65, 0.85],
    samples_per_rho: int = 1000,
    seed: int = 42
):
    """Collects steady-state expert data with noisy state augmentation."""
    log.info(f"--- Collecting Robust Expert Data (Steady-State + Augmentation) ---")
    rng = np.random.default_rng(seed)
    
    all_states = []
    all_rhos = []
    all_probs = []
    all_q_targets = []  # Target for Value Network (Mean Queue)
    all_mus = []        # PATCH: Store corresponding mu for generalization training
    
    # PATCH: Domain Randomization mapping not just rho, but also scaling factors for mu!
    # To generalize better, we'll aggressively downscale and upscale capacities
    # to prevent value hallucination out of bounds.
    mu_scales = [0.5, 1.0, 2.0]
    samples_per_variant = max(100, samples_per_rho // len(mu_scales))
    
    for mu_scale in mu_scales:
        scaled_service_rates = service_rates * mu_scale
        expert = SojournTimeSoftmaxRouting(scaled_service_rates, alpha=1.0)
        total_capacity = np.sum(scaled_service_rates)
        
        for rho in rhos:
            arrival_rate = rho * total_capacity
            res = simulate(
                num_servers=num_servers,
                arrival_rate=arrival_rate,
                service_rates=scaled_service_rates,
                policy=expert,
                sim_time=1500.0,
                sample_interval=1.0,
                rng=rng
            )
            # Take steady-state samples (second half)
            states = res.states[len(res.states)//2:]
            
            if len(states) > samples_per_variant:
                indices = rng.choice(len(states), size=samples_per_variant, replace=False)
                states = states[indices]
                
            for s in states:
                # 1. Original
                all_states.append(s)
                all_rhos.append(rho)
                all_probs.append(expert(s, rng))
                all_q_targets.append(np.sum(s))
                all_mus.append(scaled_service_rates)
                
                # 2. Noisy Augmentation
                noise = rng.integers(-1, 2, size=num_servers)
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
):
    """Trains a neural policy using Ultra-Robust BC logic."""
    # Extract num_servers from the model's final layer
    num_servers = policy_net.layers[-1].weight.shape[0]
    
    X, R, Y, G, MU = collect_robust_expert_data(
        num_servers=num_servers,
        service_rates=service_rates
    )
    
    # Optimizer
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(policy_net, eqx.is_array))
    
    @eqx.filter_jit
    def loss_fn(model, x, r, target_probs, mu_batch):
        # PATCH: Use local mu_batch so generated features match the domain randomization
        s_feat = jax.vmap(sojourn_time_features)(x, mu_batch)
        logits = jax.vmap(model)(s_feat, r)
        
        # Softmax over expert exact probability mass
        soft_labels = target_probs * (1.0 - label_smoothing) + (label_smoothing / num_servers)
        
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce_loss = -jnp.mean(jnp.sum(soft_labels * log_probs, axis=-1))
        
        # Entropy bonus
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
    for i in range(num_steps + 1):
        policy_net, opt_state, loss, acc = step(policy_net, opt_state, X, R, Y, MU)
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
):
    """Trains a value network (critic) to estimate steady-state queue lengths."""
    # Extract num_servers from the model's layers
    num_servers = len(service_rates)
    
    X, R, Y, G, MU = collect_robust_expert_data(
        num_servers=num_servers,
        service_rates=service_rates
    )
    
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(value_net, eqx.is_array))
    
    # SOTA FIX: Map instantaneous expert queue lengths to Performance Index (G_idx)
    # This precisely aligns the Critic initialization with the RL advantage target.
    G_idx = 100.0 * (random_limit - G) / denom
    G_scaled = squash_scale * jnp.tanh(G_idx / squash_threshold)

    @eqx.filter_jit
    def loss_fn(model, x, r, g, mu_batch):
        # PATCH: Use local mu_batch so generated features match the domain randomization
        s_feat = jax.vmap(sojourn_time_features)(x, mu_batch)
        preds = jax.vmap(model)(s_feat, r)
        loss = jnp.mean((preds - g)**2)
        return loss

    @eqx.filter_jit
    def step(model, opt_state, x, r, g, mu_batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, r, g, mu_batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    log.info(f"--- Bootstrapping Critic (Value Warming) ---")
    for i in range(num_steps + 1):
        value_net, opt_state, loss = step(value_net, opt_state, X, R, G_scaled, MU)
        if i % 100 == 0:
            log.info(f"  Step {i:4d} | MSE Loss: {loss:.4f}")
            
    return value_net, opt_state
