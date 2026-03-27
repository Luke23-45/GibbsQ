# Master Audit Index

## Loop Status
Current pass     : 1
Experiments done : 12 / 12
New findings     : 1
Exit condition   : not met (new smoking gun found; rerun loop required)

## Experiment Status Table
| # | Name | Entry File | Status | Smoking Guns | Pass |
|---|---|---|---|---:|---:|
| 1 | check_configs | experiments/testing/check_configs.py | AUDITED | 0 | 1 |
| 2 | reinforce_check | experiments/testing/reinforce_gradient_check.py | AUDITED | 1 | 1 |
| 3 | drift | experiments/verification/drift_verification.py | AUDITED | 0 | 1 |
| 4 | sweep | experiments/sweeps/stability_sweep.py | AUDITED | 0 | 1 |
| 5 | stress | experiments/testing/stress_test.py | AUDITED | 0 | 1 |
| 6 | policy | experiments/evaluation/baselines_comparison.py | AUDITED | 0 | 1 |
| 7 | bc_train | experiments/training/pretrain_bc.py | AUDITED | 0 | 1 |
| 8 | reinforce_train | experiments/training/train_reinforce.py | AUDITED | 0 | 1 |
| 9 | stats | experiments/evaluation/n_gibbsq_evals/stats_bench.py | AUDITED | 0 | 1 |
| 10 | generalize | experiments/evaluation/n_gibbsq_evals/gen_sweep.py | AUDITED | 0 | 1 |
| 11 | ablation | experiments/evaluation/n_gibbsq_evals/ablation_ssa.py | AUDITED | 0 | 1 |
| 12 | critical | experiments/evaluation/n_gibbsq_evals/critical_load.py | AUDITED | 0 | 1 |

## Final Loop Log
| Pass | Experiments | New Findings | Exit Condition |
|---:|---:|---:|---|
| 1 | 12 | 1 | not met (new smoking gun; unresolved unknown remains) |

## Final Status
| # | Experiment | Smoking Guns | Status |
|---|---|---:|---|
| 1 | check_configs | 0 | AUDITED |
| 2 | reinforce_check | 1 | AUDITED |
| 3 | drift | 0 | AUDITED |
| 4 | sweep | 0 | AUDITED |
| 5 | stress | 0 | AUDITED |
| 6 | policy | 0 | AUDITED |
| 7 | bc_train | 0 | AUDITED |
| 8 | reinforce_train | 0 | AUDITED |
| 9 | stats | 0 | AUDITED |
| 10 | generalize | 0 | AUDITED |
| 11 | ablation | 0 | AUDITED |
| 12 | critical | 0 | AUDITED |

## Audit Complete
Passes run        : 1
Total files read  : 24
Sources consulted : 24
Smoking guns found: 1
Unresolved unknowns: 1
Exit condition met: NO
