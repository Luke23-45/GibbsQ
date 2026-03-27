# Sources Consulted

## Files Read (codebase)

- scripts/execution/experiment_runner.py — canonical 12-experiment mapping.
- scripts/execution/reproduction_pipeline.py — pipeline ordering and launch behavior.
- docs/gibbsq.md — theoretical foundation and routing/stability statements.
- src/gibbsq/core/config.py — validation, capacity/alpha enforcement, derived constants.
- src/gibbsq/core/policies.py — softmax/UAS policy math and numerical stabilization.
- src/gibbsq/core/neural_policies.py — neural policy/value forward definitions.
- src/gibbsq/core/features.py — look-ahead potential and softmax variants.
- src/gibbsq/core/builders.py — policy construction and engine selection.
- src/gibbsq/engines/numpy_engine.py — SSA semantics and max_events behavior.
- src/gibbsq/engines/jax_engine.py — JAX SSA policy routing and validation.
- src/gibbsq/engines/jax_ssa.py — trajectory collection and Poisson max-step bound.
- src/gibbsq/analysis/metrics.py — stationarity and summary metric definitions.
- src/gibbsq/utils/exporter.py — metrics/trajectory persistence paths.
- src/gibbsq/utils/logging.py — run capsule and logging initialization.
- configs/default.yaml — baseline experiment domains and defaults.
- configs/small.yaml — small profile domains.
- configs/large.yaml — large profile domains.
- configs/debug.yaml — debug profile domains.
- configs/drift.yaml — drift-focused profile domains.
- experiments/testing/check_configs.py — experiment #1 entry.
- experiments/testing/reinforce_gradient_check.py — experiment #2 entry.
- experiments/verification/drift_verification.py — experiment #3 entry.
- experiments/sweeps/stability_sweep.py — experiment #4 entry.
- experiments/testing/stress_test.py — experiment #5 entry.
- experiments/evaluation/baselines_comparison.py — experiment #6 entry.
- experiments/training/pretrain_bc.py — experiment #7 entry.
- experiments/training/train_reinforce.py — experiment #8 entry.
- experiments/evaluation/n_gibbsq_evals/stats_bench.py — experiment #9 entry.
- experiments/evaluation/n_gibbsq_evals/gen_sweep.py — experiment #10 entry.
- experiments/evaluation/n_gibbsq_evals/ablation_ssa.py — experiment #11 entry.
- experiments/evaluation/n_gibbsq_evals/critical_load.py — experiment #12 entry.

## External References
None required for confirmed-fault adjudication in this pass.
