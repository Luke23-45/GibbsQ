# Runner Guide

This directory contains the run-facing entrypoints for the experiment pipeline.
Use these scripts when you want a controlled, repeatable execution path rather
than running experiment modules one by one.

There are four different layers:

- main `z2` thesis pipeline
- benchmark-only or verification-only slices of that pipeline
- neural training pipeline
- neural applicability support pipeline

The rule is:

- run the main `z2` pipeline first
- run the training pipeline to generate model weights
- run the neural support pipeline only after the training pipeline is complete
- do not mix archived experiments into the active thesis workflow

## 1. Files in this directory

### Main thesis runners

- [run_verification.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/studies/runners/run_verification.py)
  Runs the active `z2` verification experiments:
  - boundary equilibrium verification
  - reflected-ODE convergence
  - CTMC boundary mismatch demo
  - direct CTMC validation
  - exhaustive drift audit
  - theorem constant sweep
  - consolidated CTMC support summary

- [run_benchmarks.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/studies/runners/run_benchmarks.py)
  Runs the active `H5` benchmark anchor:
  - independent-seed benchmark rerun

- [run_all_z2.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/studies/runners/run_all_z2.py)
  Runs the main thesis pipeline end to end:
  - verification first
  - benchmarks second

### Supporting neural runners

- [run_training.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/studies/runners/run_training.py)
  Runs the neural learning pipeline to generate weights required for `H7`:
  - platinum BC pretraining
  - REINFORCE SSA training

- [run_neural_support.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/studies/runners/run_neural_support.py)
  Runs the neural applicability studies that support `H7`:
  - corrected policy comparison
  - statistical benchmark
  - generalization sweep
  - critical-load study
  - SSA ablation study

These neural runs are supporting only. They are not theorem evidence.

## 2. Recommended run order

### Fast safety check

Run the execution guards first:

```powershell
python -m gibbsq.experiments.testing.check_configs
python -m gibbsq.experiments.verification.engine_parity --config-name final_experiment
```

### Main thesis run

```powershell
python -m studies.runners.run_all_z2 --output-dir outputs/data --report-dir outputs/reports
```

### Verification only

Use this when you want only the deterministic and stochastic support layer.

```powershell
python -m studies.runners.run_verification --output-dir outputs/data --report-dir outputs/reports
```

### Benchmark only

Use this when verification is already complete and you want just the empirical
anchor rerun.

```powershell
python -m studies.runners.run_benchmarks --output-dir outputs/data --report-dir outputs/reports
```

### Neural training

Use this to generate the required weights before running neural support.

```powershell
python -m studies.runners.run_training --config-name final_experiment --report-dir outputs/reports
```

### Neural support only

Only run this after you have prepared the required neural model pointer via the training runner.

```powershell
python -m studies.runners.run_neural_support --config-name final_experiment --report-dir outputs/reports
```

## 3. Dry runs

Use dry runs first whenever you want to check what will execute.

```powershell
python -m studies.runners.run_verification --dry-run
python -m studies.runners.run_benchmarks --dry-run
python -m studies.runners.run_all_z2 --dry-run
python -m studies.runners.run_training --dry-run
python -m studies.runners.run_neural_support --dry-run
```

## 4. Detailed coverage

### `run_verification.py`

Purpose:
- supports `H1`, `H2`, `H3`, and `H4`
- covers publication-facing verification experiments only

Experiments covered:
- `Boundary Equilibrium Verification`
- `Reflected-ODE Multi-Start Convergence`
- `CTMC Generator Boundary-Mismatch Demo`
- `Direct CTMC Validation Capsule`
- `Exhaustive Small-Grid Drift Audit`
- `Theorem-Constant Parameter Sweep`
- `Consolidated CTMC Support Summary`

Outputs:
- experiment artifacts under the selected `--output-dir`
- pipeline report under the selected `--report-dir`

Notes:
- this runner covers the publication-facing verification layer
- it does not run neural studies
- it does not run archived experiments

### `run_verification_checks.py`

Purpose:
- runs support-only verification checks that are not publication-facing experiments

Checks covered:
- `Configuration Sanity Checks`
- `Engine Parity Check (NumPy vs JAX)`
- `Theorem-Backed Drift Verification`
- `Reflected UAS Exploratory Proof Search`

Outputs:
- check artifacts under the selected `--output-dir`
- pipeline report under the selected `--report-dir`

Notes:
- use this runner when you want operational consistency checks without mixing them into the paper-facing experiment rollup

### `run_benchmarks.py`

Purpose:
- supports `H5`

Experiments covered:
- `Independent-Seed Benchmark Rerun`

Outputs:
- benchmark artifacts under the selected `--output-dir`
- benchmark report under the selected `--report-dir`

Notes:
- this runner is simulation-heavy
- use it after verification if you want a narrower rerun

### `run_all_z2.py`

Purpose:
- master runner for the active thesis experiment package

Stages:
1. verification
2. benchmark

Options:
- `--skip-benchmarks`
  Run only the verification stage

Example:

```powershell
python -m studies.runners.run_all_z2 --skip-benchmarks
```

### `run_training.py`

Purpose:
- generate neural model weights for `H7`

Experiments covered:
- `Platinum BC Pretraining`
- `REINFORCE SSA Training`

Notes:
- this handles the deep learning portion of the project
- models are dumped into the configured artifact directory

### `run_neural_support.py`

Purpose:
- supports `H7` only

Experiments covered:
- `Corrected Policy Comparison`
- `Neural Statistical Benchmark`
- `Neural Generalization Sweep`
- `Neural Critical-Load Study`
- `Neural SSA Ablation`

Prerequisite:
- a valid trained neural model pointer must already exist (run `run_training.py` first)

Notes:
- these runs are empirical applicability studies
- they must not be cited as theorem evidence

## 5. Direct module commands

Use these when you want to run one experiment directly instead of using a
runner.

### Main verification modules

```powershell
python -m gibbsq.experiments.verification.boundary_equilibrium_verification --output-dir outputs/data
python -m gibbsq.experiments.verification.reflected_ode_convergence --output-dir outputs/data
python -m gibbsq.experiments.verification.ctmc_boundary_mismatch_demo --output-dir outputs/data
python -m gibbsq.experiments.verification.direct_ctmc_validation --output-dir outputs/data --mode audit
python -m gibbsq.experiments.verification.exhaustive_drift_audit --output-dir outputs/data
python -m gibbsq.experiments.verification.theorem_constant_sweep --output-dir outputs/data
python -m gibbsq.experiments.verification.ctmc_support_summary --output-dir outputs/data
```

### Benchmark module

```powershell
python -m gibbsq.experiments.benchmark.independent_seed_rerun --output-dir outputs/data
```

### Supporting modules

```powershell
python -m gibbsq.experiments.testing.check_configs
python -m gibbsq.experiments.verification.engine_parity --config-name final_experiment
python -m gibbsq.experiments.verification.drift_verification --config-name final_experiment
python -m gibbsq.experiments.verification.reflected_uas_proof_search --config-name final_experiment
```

### Neural applicability modules

```powershell
python -m gibbsq.experiments.evaluation.baselines_comparison --config-name final_experiment
python -m gibbsq.experiments.evaluation.n_gibbsq_evals.stats_bench --config-name final_experiment
python -m gibbsq.experiments.evaluation.n_gibbsq_evals.gen_sweep --config-name final_experiment
python -m gibbsq.experiments.evaluation.n_gibbsq_evals.critical_load --config-name final_experiment
python -m gibbsq.experiments.evaluation.n_gibbsq_evals.ablation_ssa --config-name final_experiment
```

### Neural training prerequisites

These are not thesis evidence objects. Use them only to prepare weights for the
neural applicability study.

```powershell
python -m gibbsq.experiments.training.pretrain_bc --config-name final_experiment
python -m gibbsq.experiments.training.train_reinforce --config-name final_experiment
```

## 6. What is intentionally not run here

These are not part of the active thesis runner layer:

- `sruas_validation.py`
- `direction12_probe.py`
- `state_dependent_uas_probe.py`
- `smvr_probe.py`
- `compare_softmax_variants.py`
- `compare_rl_variants.py`
- `reinforce_gradient_check.py`
- `reconciliation_proof.py`
- `hyperparameter_qualification.py`
- `stability_sweep.py`
- `stress_test.py`

Those files remain archived research history.

## 7. Final guardrails

- Run `run_all_z2.py` for the main thesis package.
- Run `run_training.py` to bake neural model weights.
- Run `run_neural_support.py` only after the training is complete.
- Do not cite neural-study outputs as theorem evidence.
- Do not revive archived experiments into the active runner workflow.
