#!/usr/bin/env python3
"""
Diagnostic Script: Explained Variance (EV) Analysis.
Tests the correlation between Value Network predictions and different reward scales.
"""

import sys
import os
from pathlib import Path

# Fix python path
# If script is in scripts/research/debug_ev.py, parent is research, parent.parent is scripts, parent.parent.parent is root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import json
import equinox as eqx
from omegaconf import OmegaConf

from gibbsq.core.config import ExperimentConfig
from gibbsq.core.neural_policies import NeuralRouter, ValueNetwork
from gibbsq.core.features import sojourn_time_features
from experiments.training.train_reinforce import collect_trajectory_ssa, compute_causal_returns_to_go

def load_latest_model():
    # Find latest weights pointer
    ptr_path = Path("outputs/small/latest_reinforce_weights.txt")
    if not ptr_path.exists():
        ptr_path = Path("C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/MoEQ/outputs/small/latest_reinforce_weights.txt")
    
    if not ptr_path.exists():
        raise FileNotFoundError("Could not find latest_reinforce_weights.txt")
        
    rel_path = ptr_path.read_text().strip()
    abs_path = Path("C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/MoEQ") / rel_path
    
    # Load config (try run dir, then root)
    run_dir = abs_path.parent
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = ROOT / "configs" / "small.yaml"
        print(f"Using fallback config: {cfg_path}")
    
    raw_cfg = OmegaConf.load(cfg_path)
    
    # Initialize from config
    num_servers = raw_cfg.system.num_servers
    
    # Use actual config objects if possible, else mock
    from gibbsq.core.config import NeuralConfig
    nc = NeuralConfig(
        layer_sizes=list(raw_cfg.neural.layer_sizes),
        use_rho=raw_cfg.neural.use_rho,
        rho_input_scale=raw_cfg.neural.rho_input_scale
    )
    
    policy_net = NeuralRouter(num_servers=num_servers, config=nc)
    value_net = ValueNetwork(num_servers=num_servers, config=nc)
    
    policy_net = eqx.tree_deserialise_leaves(abs_path, policy_net)
    
    # Value weights are usually value_network_weights.eqx in results dir
    v_path = run_dir / "value_network_weights.eqx"
    if v_path.exists():
        value_net = eqx.tree_deserialise_leaves(v_path, value_net)
        print(f"Loaded Critic: {v_path.name}")
    else:
        print("Warning: value_network_weights.eqx not found. Using randomly initialized critic (may be after bootstrap).")
        
    return policy_net, value_net, raw_cfg

def analyze_ev():
    print("=" * 60)
    print(" EXPLAINED VARIANCE (EV) DIAGNOSTIC")
    print("=" * 60)
    
    policy_net, value_net, cfg = load_latest_model()
    num_servers = cfg.system.num_servers
    service_rates = np.array(cfg.system.service_rates)
    service_rates_jax = jnp.array(service_rates)
    
    # Collect 10 trajectories
    print(f"Simulating 10 trajectories (sim_time={cfg.simulation.ssa.sim_time})...")
    np_rng = np.random.default_rng(cfg.simulation.seed + 1337)
    
    all_V = []
    all_G_raw = []
    all_G_scaled = []
    
    for i in range(10):
        rho = cfg.system.arrival_rate / np.sum(service_rates)
        traj = collect_trajectory_ssa(
            policy_net=policy_net,
            num_servers=num_servers,
            arrival_rate=cfg.system.arrival_rate,
            service_rates=service_rates,
            sim_time=cfg.simulation.ssa.sim_time,
            rng=np_rng,
            rho=rho
        )
        
        # Compute G_raw
        G_raw = compute_causal_returns_to_go(
             traj.all_states, traj.jump_times, traj.action_step_indices, 
             sim_time=cfg.simulation.ssa.sim_time,
             gamma=cfg.get("neural_training", {}).get("gamma", 0.99)
        )
        
        if len(G_raw) == 0: continue
        
        # Compute Vpreds
        S_tensor = jnp.array(np.stack(traj.states))
        rho_tensor = jnp.full((len(S_tensor),), rho)
        s_feat = sojourn_time_features(S_tensor, service_rates_jax)
        v_preds = jax.vmap(value_net)(s_feat, rho_tensor)
        
        # Scaling logic exactly as in train_reinforce.py
        # random_limit logic extraction
        q_rand_analytical = 0.0
        lam_i = cfg.system.arrival_rate / num_servers
        for mu in service_rates:
            q_rand_analytical += lam_i / (mu - lam_i)
        
        # Note: In real run, jsq_limit is measured. Here we estimate for diagnostic purposes.
        jsq_est = q_rand_analytical * 0.4 # Expert usually improves by 60%
        random_limit = max(jsq_est * 1.01, q_rand_analytical)
        
        # denominators
        denom = max(0.5, jsq_est * 0.05, random_limit - jsq_est)
        
        t_actions = np.array([traj.jump_times[idx] for idx in traj.action_step_indices])
        t_rem = np.maximum(1e-3, cfg.simulation.ssa.sim_time - t_actions)
        
        # G_scaled (Time-Normalized PI-V3)
        G_pi = 100.0 * (random_limit * t_rem - G_raw) / (denom * t_rem)
        
        all_V.extend(v_preds.tolist())
        all_G_raw.extend(G_raw.tolist())
        all_G_scaled.extend(G_pi.tolist())

    V = np.array(all_V)
    Gr = np.array(all_G_raw)
    Gs = np.array(all_G_scaled)
    
    # Correlations
    corr_raw = np.corrcoef(V, Gr)[0, 1]
    corr_scaled = np.corrcoef(V, Gs)[0, 1]
    
    print("-" * 60)
    print(f"Sample Size: {len(V)} decision points")
    print(f"Mean V_pred: {np.mean(V):.4f}  | Std V_pred: {np.std(V):.4f}")
    print(f"Mean G_raw:  {np.mean(Gr):.4f} | Std G_raw:  {np.std(Gr):.4f}")
    print(f"Mean G_scaled: {np.mean(Gs):.4f} | Std G_scaled: {np.std(Gs):.4f}")
    print("-" * 60)
    print(f"Correlation(V, G_raw):    {corr_raw:.4%}")
    print(f"Correlation(V, G_scaled): {corr_scaled:.4%}")
    print("-" * 60)
    
    if abs(corr_raw) > abs(corr_scaled) + 0.1:
        print("💡 INSIGHT: Critic aligns better with Raw returns than Normalized returns.")
        print("This confirms the normalization mismatch hypothesis.")
    elif np.std(V) < 0.01:
        print("💡 INSIGHT: Critic Stagnation confirmed. Predicts nearly constant value.")
    e