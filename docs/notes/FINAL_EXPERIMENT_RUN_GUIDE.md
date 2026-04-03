# Final Experiment Run Guide

This guide is the authoritative execution order for the finalized
`final_experiment` profile.

It assumes:

- the final config in `configs/final_experiment.yaml` is already frozen
- expensive jobs are run standalone
- lighter jobs are bundled through the execution scripts
- execution happens from the repository root

## Final execution split

### Phase 1: bundled core pipeline

Run the lighter pre-train and validation jobs together:

```bash
python scripts/execution/final/phase1_pipeline.py --config-name final_experiment --progress off
```

This phase runs:

- `check_configs`
- `reinforce_check`
- `drift`
- `sweep`
- `bc_train`

`stress` is intentionally not part of this bundled phase because it is finalized as a standalone expensive run.

### Phase 2: standalone expensive jobs

Run these one by one, in this order:

```bash
python scripts/execution/experiment_runner.py --config-name final_experiment --progress off reinforce_train
python scripts/execution/experiment_runner.py --config-name final_experiment --progress off stress
python scripts/execution/experiment_runner.py --config-name final_experiment --progress off generalize
python scripts/execution/experiment_runner.py --config-name final_experiment --progress off critical
python scripts/execution/experiment_runner.py --config-name final_experiment --progress off ablation
```

Why this order:

1. `reinforce_train` must run before model-dependent evaluations.
2. `stress` is independent, expensive, and should get its own session.
3. `generalize` and `critical` require trained weights from `reinforce_train`.
4. `ablation` is expensive but self-contained once the training budget is finalized.

### Phase 3: bundled post-train light jobs

After `reinforce_train` succeeds, run:

```bash
python scripts/execution/final/phase3_pipeline.py --config-name final_experiment --progress off
```

This phase runs:

- `policy`
- `stats`

## Resume rules

If a phase is interrupted:

- for Phase 1, rerun the same pipeline command with `--start-from <alias>` if needed
- for Phase 2, rerun only the failed standalone job
- for Phase 3, rerun the post-train pipeline with `--start-from <alias>` if needed

Examples:

```bash
python scripts/execution/final/phase1_pipeline.py --config-name final_experiment --progress off --start-from sweep
python scripts/execution/final/phase3_pipeline.py --config-name final_experiment --progress off --start-from stats
```

## Final locked budgets that matter during execution

The standalone-heavy jobs are expected to resolve as:

- `reinforce_train`
  - `train_epochs: 15`
  - `batch_size: 16`
  - `simulation.ssa.sim_time: 1000.0`
- `generalize`
  - `simulation.num_replications: 2`
  - `simulation.ssa.sim_time: 10000.0`
  - `generalization.scale_vals: [0.5, 1.0, 2.0]`
  - `generalization.rho_grid_vals: [0.5, 0.7, 0.85]`
- `critical`
  - `simulation.num_replications: 2`
  - `generalization.rho_boundary_vals: [0.90, 0.92, 0.95, 0.97, 0.98, 0.985, 0.99]`
- `stress`
  - unchanged from the frozen final profile
- `ablation`
  - inherits the finalized `reinforce_train` budget

## Recommended Colab workflow

For Colab, treat each standalone expensive job as its own session if needed.

Recommended sequence:

1. start a fresh runtime
2. run Phase 1
3. run `reinforce_train`
4. run Phase 3
5. run `stress`
6. run `generalize`
7. run `critical`
8. run `ablation`

This minimizes the risk of losing trained-weight dependencies before `policy` and `stats`.

## Command checklist

Run these from the repository root:

```bash
python scripts/execution/final/phase1_pipeline.py --config-name final_experiment --progress off
python scripts/execution/experiment_runner.py --config-name final_experiment --progress off reinforce_train
python scripts/execution/final/phase3_pipeline.py --config-name final_experiment --progress off
python scripts/execution/experiment_runner.py --config-name final_experiment --progress off stress
python scripts/execution/experiment_runner.py --config-name final_experiment --progress off generalize
python scripts/execution/experiment_runner.py --config-name final_experiment --progress off critical
python scripts/execution/experiment_runner.py --config-name final_experiment --progress off ablation
```
