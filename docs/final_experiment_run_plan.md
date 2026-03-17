# Final Experiment Script Selection Plan

## 1) Source-of-truth criteria used for categorization

This classification is based on repository sources only:

1. **Research objective**: queueing-network stability under softmax routing and CTMC/SSA framing from the main research document. (`docs/gibbsq.md`).
2. **Phase IV corrective loop** explicitly defines the corrected stack as:
   - `reinforce_check`
   - `reinforce_train`
   - `dr_train`
   - `corrected_policy`
   and states deprecated DGA experiments are ignored (`scripts/verify_phase_iv.ps1`).
3. **Launcher deprecation labels** in `scripts/run_experiment.sh` and `scripts/run_experiment.ps1` are treated as explicit deprecation intent.
4. **Script-level deprecation headers** are treated as explicit deprecation intent.

## 2) Complete experiment inventory (audited)

### A) Core corrected pipeline (MOVE FORWARD)

| Alias / Script | Module | Category | Why |
|---|---|---|---|
| `reinforce_train` | `experiments/n_gibbsq/train_reinforce.py` | **FINAL TRAINING** | Explicit Phase IV Track 1 corrected REINFORCE training. |
| `dr_train` | `experiments/n_gibbsq/train_domain_randomized.py` | **FINAL TRAINING** | Explicit Phase IV Track 3 corrected domain-randomized REINFORCE training. |
| `reinforce_check` | `experiments/testing/reinforce_gradient_check.py` | **FINAL VALIDATION** | Explicit Phase IV Track 5 gradient estimator validation. |
| `corrected_policy` | `experiments/evaluation/corrected_policy_comparison.py` | **FINAL EVALUATION** | Explicit Phase IV Track 4 corrected policy benchmark. |

### B) Auxiliary SSA analyses (KEEP as secondary / optional)

| Alias / Script | Module | Category | Why |
|---|---|---|---|
| `ablation` | `experiments/n_gibbsq/ablation_ssa.py` | **KEEP (SSA AUX)** | New SSA-native ablation path; launcher alias now points here. |
| `critical` | `experiments/n_gibbsq/critical_load.py` | **KEEP (SSA AUX)** | Critical-load stress analysis on SSA evaluation path. |
| `generalize` | `experiments/n_gibbsq/gen_sweep.py` | **KEEP (SSA AUX)** | Generalization sweep with SSA evaluation path. |
| `stats` | `experiments/n_gibbsq/stats_bench.py` | **KEEP (SSA AUX)** | Statistical benchmark on SSA evaluation path. |
| `drift` | `experiments/verification/drift_verification.py` | **KEEP (THEORY-CHECK AUX)** | Drift verification experiment supporting theory checks. |
| `sweep` | `experiments/sweeps/stability_sweep.py` | **KEEP (THEORY-CHECK AUX)** | Stability sweep for alpha/rho behavior. |
| `stress` | `experiments/testing/stress_test.py` | **KEEP (ROBUSTNESS AUX)** | Stress diagnostics (scale/critical-load). |

### C) Legacy/deprecated stack (DROP from final run)

| Alias / Script | Module | Category | Why |
|---|---|---|---|
| `train` | `experiments/training/train_dga.py` | **DROP (DEPRECATED)** | Launcher marks deprecated DGA routing-agent training. |
| `n_train` | `experiments/n_gibbsq/train.py` | **DROP (DEPRECATED)** | Launcher marks deprecated neural curriculum training. |
| `policy` | `experiments/evaluation/policy_comparison.py` | **DROP (DEPRECATED)** | Launcher marks deprecated; replaced by corrected policy benchmark. |
| `parity` | `experiments/n_gibbsq/eval.py` | **DROP (DEPRECATED)** | Launcher marks deprecated legacy parity flow. |
| `jacobian` | `experiments/n_gibbsq/jacobian_check.py` | **DROP (DEPRECATED)** | Launcher marks deprecated Jacobian diagnostic flow. |
| `fidelity` | `experiments/n_gibbsq/grad_check.py` | **DROP (LEGACY DGA CHECK)** | Gradient fidelity check on DGA path, not Phase IV corrected stack. |
| (no alias) | `experiments/n_gibbsq/ablation.py` | **DROP (DEPRECATED)** | Script header marks legacy DGA-based ablation. |
| (no alias) | `experiments/testing/debug_scaling_test.py` | **DROP (DEPRECATED)** | Script header marks deprecated diagnostic. |
| (no alias) | `experiments/testing/debug_stress_runner.py` | **DROP (DEPRECATED)** | Script header marks deprecated diagnostic runner. |
| `bias` | `experiments/testing/verify_bias.py` | **DROP (LEGACY DGA-vs-SSA DIAGNOSTIC)** | DGA/SSA bias diagnostic, not in final Phase IV run chain. |
| (no alias) | `experiments/testing/check_configs.py` | **UTILITY ONLY** | Config-validation helper, not an experiment result script. |

## 3) Final run-set recommendation

For **final paper/research run**, execute this ordered set:

1. `reinforce_check` (gate)
2. `reinforce_train`
3. `dr_train`
4. `corrected_policy`
5. Optional add-ons for extended evidence: `ablation` (SSA), `stats`, `generalize`, `critical`, `drift`, `sweep`, `stress`

## 4) Triple-check checklist before final execution

1. Confirm all final aliases map to intended modules in both launchers.
2. Confirm no deprecated alias appears in the production run script.
3. Confirm final run config and output directories are pinned and reproducible.
4. Confirm each final stage emits artifacts/metrics before moving to next stage.

