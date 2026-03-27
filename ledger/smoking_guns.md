# Smoking Gun Registry

## Summary Table
| # | Experiment | File | Severity | Status |
| --- | --- | --- | --- | --- |
| 1 | check_configs | experiments/testing/check_configs.py | HIGH | CONFIRMED |
| 2 | reinforce_check | experiments/testing/reinforce_gradient_check.py | HIGH | CONFIRMED |
| 3 | drift | src/gibbsq/core/drift.py | CRITICAL | CONFIRMED |
| 4 | stress | experiments/testing/stress_test.py | HIGH | CONFIRMED |
| 5 | policy | experiments/evaluation/baselines_comparison.py | HIGH | CONFIRMED |
| 6 | bc_train | experiments/training/pretrain_bc.py | CRITICAL | CONFIRMED |
| 7 | reinforce_train | experiments/training/train_reinforce.py | CRITICAL | CONFIRMED |
| 8 | stats | experiments/evaluation/n_gibbsq_evals/stats_bench.py | HIGH | CONFIRMED |
| 9 | generalize | experiments/evaluation/n_gibbsq_evals/gen_sweep.py | HIGH | CONFIRMED |
| 10 | critical | experiments/evaluation/n_gibbsq_evals/critical_load.py | HIGH | CONFIRMED |

## Cross-Experiment Findings
- SMOKING GUN #6 propagates into `policy`, `stats`, `generalize`, and `critical` because their model-resolution path prefers `latest_domain_randomized_weights.txt` over `latest_reinforce_weights.txt`.
- SMOKING GUN #7 propagates into `ablation` because the ablation trainer subclasses `ReinforceTrainer.execute` and therefore inherits the same sampled-policy versus optimized-policy mismatch.
- SMOKING GUN #1 masks the broken `configs/drift.yaml` profile, which directly affects the drift verification experiment when that profile is selected.

## Confirmed Findings
SMOKING GUN #1
  Experiment    : check_configs
  File          : experiments/testing/check_configs.py
  Function      : main
  Lines         : 13-17
  Observed      : The validation loop composes only default, small, and large configs.
  Source        : experiments/testing/check_configs.py:13-17; src/gibbsq/core/config.py:751-757 + configs/drift.yaml:74-84 verified by local compose/hydra_to_config run returning "TypeError: NeuralConfig.__init__() got an unexpected keyword argument 'grad_clip'"
  Impact        : The repository contains a broken config profile that the pre-flight experiment does not exercise, so the reported success message can exclude a failing experiment configuration.
  Severity      : HIGH
  Severity basis: The experiment is intended to catch configuration faults before execution, but it misses a real conversion failure in a shipped experiment profile.
  Status        : CONFIRMED

SMOKING GUN #2
  Experiment    : reinforce_check
  File          : experiments/testing/reinforce_gradient_check.py
  Function      : compute_reinforce_gradient
  Lines         : 115-146
  Observed      : The REINFORCE loss uses `batch.returns` as the advantage signal and then multiplies that signal by an additional `gamma ** action_idx` factor before summing `adv * log_prob`.
  Source        : experiments/testing/reinforce_gradient_check.py:115-146; src/gibbsq/engines/jax_ssa.py:86-107,237 plus experiments/testing/reinforce_gradient_check.py:261-268
  Impact        : `compute_causal_returns_jax` already returns discounted action returns, while the finite-difference reference extracts `G_0` from those same returns without the extra factor, so the experiment compares a double-discounted score-function objective against a different finite-difference target.
  Severity      : HIGH
  Severity basis: This is the core validity check for the REINFORCE estimator; using mismatched objectives can invalidate a pass/fail conclusion even when the underlying implementation is otherwise correct.
  Status        : CONFIRMED

SMOKING GUN #3
  Experiment    : drift
  File          : src/gibbsq/core/drift.py
  Function      : _vectorised_softmax
  Lines         : 225-237
  Observed      : The vectorized softmax helper has no `mode == "uas"` branch, so any call with `mode='uas'` falls through to raw-queue logits instead of `-alpha * (Q+1)/mu + log(mu)`.
  Source        : src/gibbsq/core/drift.py:83-87,225-237; experiments/verification/drift_verification.py:68-72,133-137 verified by local comparison showing `verify_single(..., mode='uas')` exact drift != `_vectorised_drift(..., mode='uas')` exact drift on the same state
  Impact        : The drift experiment's grid and trajectory paths use the vectorized evaluator, so UAS drift verification is computed with the wrong routing probabilities and can misreport proof violations or proof success.
  Severity      : CRITICAL
  Severity basis: This experiment is the direct theorem-validation surface; a wrong exact generator in the audited mode invalidates the trustworthiness of its conclusions.
  Status        : CONFIRMED

SMOKING GUN #4
  Experiment    : stress
  File          : experiments/testing/stress_test.py
  Function      : main
  Lines         : 225-270
  Observed      : The critical-load path computes MSER-5 truncation points on padded `total_q_trajectories`, then converts each `d_star` to a fraction and applies that fraction to separately truncated `SimResult` arrays.
  Source        : experiments/testing/stress_test.py:225-270; src/gibbsq/analysis/metrics.py:30-37,40-53,147-199
  Impact        : `time_averaged_queue_lengths` and `stationarity_diagnostic` trim by `int(len(result.states) * fraction)`, so when `_vl < max_samples` the effective burn-in index is smaller than the MSER-5 cutoff that was estimated, biasing the reported critical-load queue and stationarity metrics.
  Severity      : HIGH
  Severity basis: This affects the experiment's near-critical statistics directly and can undercut the claimed steady-state evidence without producing an explicit runtime failure.
  Status        : CONFIRMED

SMOKING GUN #5
  Experiment    : policy
  File          : experiments/evaluation/baselines_comparison.py
  Function      : run_corrected_comparison
  Lines         : 198-204
  Observed      : On neural model shape mismatch, the code logs that it is skipping neural evaluation and then `return`s from `run_corrected_comparison`.
  Source        : experiments/evaluation/baselines_comparison.py:198-204; experiments/evaluation/baselines_comparison.py:224-301
  Impact        : A stale or mismatched checkpoint aborts the remainder of the comparison routine, so parity analysis and the comparison plot are skipped even though the analytical baseline results were already computed.
  Severity      : HIGH
  Severity basis: This can silently turn a partial evaluation into a missing report under a common checkpoint-compatibility failure mode while the log message suggests only the neural tier was skipped.
  Status        : CONFIRMED

SMOKING GUN #6
  Experiment    : bc_train
  File          : experiments/training/pretrain_bc.py
  Function      : main
  Lines         : 60-68
  Observed      : BC pretraining writes its checkpoint pointer as `latest_domain_randomized_weights.txt`.
  Source        : experiments/training/pretrain_bc.py:60-68; src/gibbsq/utils/model_io.py:60-63 plus experiments/evaluation/baselines_comparison.py:173-185 and experiments/training/train_reinforce.py:616-620
  Impact        : Downstream evaluation code prioritizes `latest_domain_randomized_weights.txt` over `latest_reinforce_weights.txt`, so after the normal `bc_train` then `reinforce_train` sequence the evaluation stack can resolve the BC warm-start weights instead of the trained REINFORCE model.
  Severity      : CRITICAL
  Severity basis: This can invalidate multiple neural evaluation experiments by loading the wrong checkpoint family while still appearing to use the “latest” model.
  Status        : CONFIRMED

SMOKING GUN #7
  Experiment    : reinforce_train
  File          : experiments/training/train_reinforce.py
  Function      : collect_trajectory_ssa / ReinforceTrainer.execute.policy_loss_fn
  Lines         : 333-343 and 840-858
  Observed      : Trajectories are sampled from the raw softmax over `policy_net.numpy_forward(...)` logits, but the optimization loss recomputes `log_probs` from `logits / temp` and never uses the recorded sampled log-probabilities.
  Source        : experiments/training/train_reinforce.py:333-343,379-380,840-858
  Impact        : The score-function term is evaluated under a different policy from the one that generated the actions, so the update is not the on-policy REINFORCE gradient for the sampled trajectories.
  Severity      : CRITICAL
  Severity basis: This directly affects the correctness of the main training algorithm rather than only its diagnostics or reporting.
  Status        : CONFIRMED

SMOKING GUN #8
  Experiment    : stats
  File          : experiments/evaluation/n_gibbsq_evals/stats_bench.py
  Function      : StatsBenchmark.execute
  Lines         : 86-92
  Observed      : On neural model shape mismatch, the benchmark logs an error and returns from `execute` without raising.
  Source        : experiments/evaluation/n_gibbsq_evals/stats_bench.py:86-92,235-247; scripts/execution/experiment_runner.py:82-97
  Impact        : The module can exit with status 0 while producing no valid benchmark result, so the pipeline can treat a failed benchmark as successful.
  Severity      : HIGH
  Severity basis: This suppresses an evaluation failure in a publication-facing benchmark and can let an invalid run pass automation checks.
  Status        : CONFIRMED

SMOKING GUN #9
  Experiment    : generalize
  File          : experiments/evaluation/n_gibbsq_evals/gen_sweep.py
  Function      : GeneralizationSweeper.execute
  Lines         : 105-111
  Observed      : On neural model shape mismatch, the sweep logs an error and returns from `execute` without raising.
  Source        : experiments/evaluation/n_gibbsq_evals/gen_sweep.py:105-111,225-238; scripts/execution/experiment_runner.py:82-97
  Impact        : The generalization sweep can terminate with a success exit code while skipping the actual sweep, so automation can record a false pass and leave no trustworthy heatmap.
  Severity      : HIGH
  Severity basis: This converts a hard evaluation failure into a silent no-op in one of the final paper experiments.
  Status        : CONFIRMED

SMOKING GUN #10
  Experiment    : critical
  File          : experiments/evaluation/n_gibbsq_evals/critical_load.py
  Function      : CriticalLoadTest.execute
  Lines         : 75-81
  Observed      : On neural model shape mismatch, the critical-load experiment logs an error and returns from `execute` without raising.
  Source        : experiments/evaluation/n_gibbsq_evals/critical_load.py:75-81,206-219; scripts/execution/experiment_runner.py:82-97
  Impact        : The critical-load experiment can report success to the runner even when no boundary analysis was performed, leaving the final pipeline vulnerable to a false-green result.
  Severity      : HIGH
  Severity basis: This hides a terminal evaluation failure in a final-stage experiment whose output is used to assess stability near the boundary.
  Status        : CONFIRMED

## Total confirmed : 10
## Total critical  : 3
## Total high      : 7
## Total medium    : 0

Each entry:
SMOKING GUN #{n}
  Experiment    : [name]
  File          : [exact path]
  Function      : [exact name]
  Lines         : [exact range]
  Observed      : [direct observation]
  Source        : [file:line or reference]
  Impact        : [traced impact]
  Severity      : [CRITICAL | HIGH | MEDIUM]
  Severity basis: [severity reasoning]
  Status        : [CONFIRMED | NEEDS VERIFICATION]
