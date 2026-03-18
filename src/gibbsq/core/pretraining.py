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
from gibbsq.core.policies import JSSQRouting
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
    expert = JSSQRouting(service_rates)
    total_capacity = np.sum(service_rates)
    rng = np.random.default_rng(seed)
    
    all_states = []
    all_rhos = []
    all_actions = []
    all_q_targets = []  # Target for Value Network (Mean Queue)
    
    for rho in rhos:
        arrival_rate = rho * total_capacity
        res = simulate(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates,
            policy=expert,
            sim_time=1500.0,
            sample_interval=1.0,
            rng=rng
        )
        # Take steady-state samples (second half)
        states = res.states[len(res.states)//2:]
        # Analytical or Empirical Mean Queue for this rho
        # We use empirical mean queue from the expert simulation as the target
        mean_q = np.mean(np.sum(states, axis=-1))
        
        if len(states) > samples_per_rho:
            indices = rng.choice(len(states), size=samples_per_rho, replace=False)
            states = states[indices]
            
        for s in states:
            # 1. Original
            all_states.append(s)
            all_rhos.append(rho)
            all_actions.append(np.argmax(expert(s, rng)))
            all_q_targets.append(mean_q)
            
            # 2. Noisy Augmentation
            noise = rng.integers(-1, 2, size=num_servers)
            s_aug = np.maximum(0, s + noise)
            all_states.append(s_aug)
            all_rhos.append(rho)
            all_actions.append(np.argmax(expert(s_aug, rng)))
            all_q_targets.append(mean_q)
            
    return (jnp.array(all_states, dtype=jnp.float32), 
            jnp.array(all_rhos, dtype=jnp.float32), 
            jnp.array(all_actions, dtype=jnp.int32),
            jnp.array(all_q_targets, dtype=jnp.float32))

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
    
    X, R, Y, G = collect_robust_expert_data(
        num_servers=num_servers,
        service_rates=service_rates
    )
    
    # Optimizer
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(policy_net, eqx.is_array))
    
    service_rates_jax = jnp.array(service_rates)
    
    @eqx.filter_jit
    def loss_fn(model, x, r, y):
        s_feat = sojourn_time_features(x, service_rates_jax)
        logits = jax.vmap(model)(s_feat, r)
        
        # Label Smoothing
        one_hot = jax.nn.one_hot(y, num_servers)
        soft_labels = one_hot * (1.0 - label_smoothing) + (label_smoothing / num_servers)
        
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce_loss = -jnp.mean(jnp.sum(soft_labels * log_probs, axis=-1))
        
        # Entropy bonus
        probs = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        ent_loss = -entropy_bonus * jnp.mean(entropy)
        
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return ce_loss + ent_loss, acc

    @eqx.filter_jit
    def step(model, opt_state, x, r, y):
        (loss, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, x, r, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, acc

    log.info(f"--- Bootstrapping Actor (Behavior Cloning) ---")
    for i in range(num_steps + 1):
        policy_net, opt_state, loss, acc = step(policy_net, opt_state, X, R, Y)
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
):
    """Trains a value network (critic) to estimate steady-state queue lengths."""
    # Extract num_servers from the model's layers
    num_servers = len(service_rates)
    
    X, R, Y, G = collect_robust_expert_data(
        num_servers=num_servers,
        service_rates=service_rates
    )
    
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(value_net, eqx.is_array))
    
    service_rates_jax = jnp.array(service_rates)

    @eqx.filter_jit
    def loss_fn(model, x, r, g):
        s_feat = sojourn_time_features(x, service_rates_jax)
        preds = jax.vmap(model)(s_feat, r)
        loss = jnp.mean((preds - g)**2)
        return loss

    @eqx.filter_jit
    def step(model, opt_state, x, r, g):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, r, g)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    log.info(f"--- Bootstrapping Critic (Value Warming) ---")
    for i in range(num_steps + 1):
        value_net, opt_state, loss = step(value_net, opt_state, X, R, G)
        if i % 100 == 0:
            log.info(f"  Step {i:4d} | MSE Loss: {loss:.4f}")
            
    return value_net, opt_state
