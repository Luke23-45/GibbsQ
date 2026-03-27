# Final Experiment Config Audit (Configurable vs Hardcoded)

This file records the parameter audit used to design `configs/final_experiment.yaml`.

## 1) Configurable (moved/kept in YAML)

The following are directly controlled through `ExperimentConfig` and are explicitly set in `final_experiment.yaml`:
- system, simulation, policy, drift, stress, generalization, stability_sweep,
  neural, neural_training, domain_randomization, jax, jax_engine,
  verification, wandb, output/log paths, train_epochs, batch_size.

## 2) Hardcoded constants found in code paths (not currently config-backed)

### Pretraining hardcoded values
- `src/gibbsq/core/pretraining.py`
  - `rhos=[0.45, 0.65, 0.85]`
  - `mu_scales=[0.5, 1.0, 2.0]`
  - expert routing `alpha=1.0`
  - data collection `sim_time=1500.0`, `sample_interval=1.0`

### JAX SSA helper defaults
- `src/gibbsq/engines/jax_ssa.py`
  - Poisson tail sigma default `sigma=6.0`
  - default `max_steps=5000`, `gamma=0.99`

### Training code defaults/fallbacks
- `experiments/training/train_reinforce.py`
  - fallback domain-randomization bounds `rho_min=0.4`, `rho_max=0.85`
  - EMA smoothing factor `alpha=0.33`
  - analytical fallback caps and constants in reward-index formulas

## 3) Final-config design policy

Given the above, `final_experiment.yaml` sets every available config-backed parameter
conservatively for reproducibility and stability while documenting remaining hardcoded
constants here for transparency.
