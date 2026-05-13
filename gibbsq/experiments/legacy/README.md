# Legacy / Archived Experiments

This directory contains experiments that are **not** part of the active `z2`
thesis program.  They are preserved for historical reference and
reproducibility, but they should not be used in the final thesis experiment
pipeline.

## Why these were archived

Per the z2 implementation plan
(`docs/implemention/z2_experiment_implementation_plan.md`), the active thesis
experiments are limited to those that directly support the z2 hypothesis ladder:

- H1/H2: Deterministic reflected-ODE theory
- H3: Stochastic obstruction of the old route
- H4: Direct CTMC certification route
- H5: Benchmark empirical performance

Experiments that serve exploratory, superseded, or neural-policy objectives
are archived here.

## Contents

| File | Original Location | Reason Archived |
|------|------------------|-----------------|
| `sruas_validation.py` | `gibbsq/experiments/verification/` | Superseded by direct CTMC route |
| `smvr_probe.py` | `gibbsq/experiments/testing/` | Failed exploratory direction |
| `state_dependent_uas_probe.py` | `gibbsq/experiments/testing/` | Exploratory; did not beat benchmark |
| `compare_softmax_variants.py` | `gibbsq/experiments/testing/` | Not part of z2 theorem program |
| `compare_rl_variants.py` | `gibbsq/experiments/testing/` | Not part of z2 theorem program |
| `reinforce_gradient_check.py` | `gibbsq/experiments/testing/` | Not part of z2 theorem program |
| `stress_test.py` | `gibbsq/experiments/testing/` | Not part of z2 theorem program |
| `reconciliation_proof.py` | `gibbsq/experiments/testing/` | Superseded by z2 proof structure |
| `direction12_probe.py` | `gibbsq/experiments/testing/` | Direction 1 superseded by `reflected_ode_convergence.py`; Direction 2 (sparsemax) not z2 |
| `pretrain_bc.py` | `gibbsq/experiments/training/` | Neural policy; supporting only |
| `train_reinforce.py` | `gibbsq/experiments/training/` | Neural policy; supporting only |
| `hyperparameter_qualification.py` | `gibbsq/experiments/studies/` | Not part of z2 theorem program |
| `stability_sweep.py` | `gibbsq/experiments/sweeps/` | Not part of z2 theorem program |

## Can I still run these?

Yes.  These files are preserved with their original logic intact.  However,
they are not maintained as part of the active z2 experiment pipeline and may
reference deprecated configurations or output paths.
