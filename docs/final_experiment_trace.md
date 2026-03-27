# Final Experiment Parameter Trace (final_experiment.yaml)

This document traces runtime-critical parameters from `configs/final_experiment.yaml`
into the code that consumes them.

## Entry path
- `scripts/execution/experiment_runner.py` forwards all Hydra overrides to the module command (`python -m <experiment>`).
- `scripts/execution/reproduction_pipeline.py` forwards global Hydra args to each experiment via `run_experiment(...)`.

## Parameter trace

| Config key | Consumed in code | Notes |
|---|---|---|
| `system.num_servers`, `system.arrival_rate`, `system.service_rates`, `system.alpha` | `src/gibbsq/core/config.py::validate`, all experiment `main(...)` after `hydra_to_config(...)` | Capacity and positivity constraints enforced before execution. |
| `simulation.num_replications` | stress/sweep/policy/stats/generalize/critical/ablation scripts | Controls replication loops and statistical aggregation. |
| `simulation.ssa.sim_time`, `simulation.ssa.sample_interval` | NumPy/JAX SSA calls across all evaluation/sweep/testing scripts | Drives trajectory horizon and sample count. |
| `policy.name`, `policy.d` | `src/gibbsq/core/builders.py`, policy maps in sweep/stress/stats/generalize/critical | Routing policy selector. |
| `jax.enabled` | `src/gibbsq/core/builders.py::select_engine`, stress script guard | Stress test requires JAX-enabled path. |
| `jax.precision`, `jax.platform`, `jax.fallback_to_cpu` | `gibbsq.utils.device.setup_jax` via `get_run_config(...)` | Applied before run capsule initialization. |
| `jax_engine.*` | `src/gibbsq/engines/jax_engine.py` | Controls scan/event safety bounds. |
| `stability_sweep.alpha_vals`, `stability_sweep.rho_vals` | `experiments/sweeps/stability_sweep.py` | Grid dimensions and workload points. |
| `stress.*` | `experiments/testing/stress_test.py` | Massive-N, critical-load, heterogeneity test axes. |
| `generalization.rho_grid_vals`, `generalization.scale_vals`, `generalization.rho_boundary_vals` | `gen_sweep.py`, `critical_load.py`, `baselines_comparison.py` | Controls transfer and boundary evaluation points. |
| `neural.*` | `src/gibbsq/core/neural_policies.py`, train/eval scripts | Network architecture and optimizer-adjacent knobs. |
| `neural_training.*` | `pretrain_bc.py`, `train_reinforce.py`, gradient-check script | BC/REINFORCE schedules and reward-index scalars. |
| `verification.*` | `reinforce_gradient_check.py`, stats/policy analyses | Gate thresholds and significance settings. |
| `output_dir`, `log_dir`, `wandb.*` | `src/gibbsq/utils/logging.py` and artifact-saving code | Run capsule structure and telemetry mode. |

## Validation status
- `final_experiment.yaml` composes and passes schema validation (`hydra_to_config + validate`).
- All 12 experiment entry scripts were smoke-run with `--config-name final_experiment` and reduced runtime overrides.
