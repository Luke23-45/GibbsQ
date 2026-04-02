PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ> python scripts\execution\reproduction_pipeline.py --config-name debug
==========================================================
  GibbsQ Research Paper: Final Execution Pipeline
==========================================================
  Progress Mode: auto

[Initiating Pipeline...]

pipeline:   0%|                                                                         | 0/12 [00:00<?, ?experiment/s]
[Pre-Flight] Running Configuration Sanity Checks...
1/12 check_configs:   0%|                                                               | 0/12 [00:00<?, ?experiment/s][handoff] Launching check_configs (1/12)
==========================================================
 Starting Experiment: check_configs
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=check_configs
==========================================================
check_configs: roots:   0%|                                                                  | 0/4 [00:00<?, ?config/s][OK] Config debug validated successfully.
[OK] Config small validated successfully.
[OK] Config default validated successfully.
[OK] Config final_experiment validated successfully.
check_configs: resolved paths:   0%|                                                          | 0/48 [00:00<?, ?path/s][OK] Resolved experiment path hyperqual (profile=debug, overrides=['++active_profile=debug', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=debug, overrides=['++active_profile=debug', '++active_experiment=reinforce_check']) validated successfully.
[OK] Resolved experiment path drift (profile=debug, overrides=['++active_profile=debug', '++active_experiment=drift']) validated successfully.
check_configs: resolved paths:   6%|███▏                                              | 3/48 [00:00<00:10,  4.27path/s][OK] Resolved experiment path sweep (profile=debug, overrides=['++active_profile=debug', '++active_experiment=sweep']) validated successfully.
[OK] Resolved experiment path stress (profile=debug, overrides=['++active_profile=debug', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Resolved experiment path policy (profile=debug, overrides=['++active_profile=debug', '++active_experiment=policy']) validated successfully.
check_configs: resolved paths:  12%|██████▎                                           | 6/48 [00:01<00:09,  4.52path/s][OK] Resolved experiment path bc_train (profile=debug, overrides=['++active_profile=debug', '++active_experiment=bc_train']) validated successfully.
[OK] Resolved experiment path reinforce_train (profile=debug, overrides=['++active_profile=debug', '++active_experiment=reinforce_train']) validated successfully.
[OK] Resolved experiment path stats (profile=debug, overrides=['++active_profile=debug', '++active_experiment=stats']) validated successfully.
check_configs: resolved paths:  19%|█████████▍                                        | 9/48 [00:01<00:08,  4.61path/s][OK] Resolved experiment path generalize (profile=debug, overrides=['++active_profile=debug', '++active_experiment=generalize']) validated successfully.
[OK] Resolved experiment path ablation (profile=debug, overrides=['++active_profile=debug', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=debug, overrides=['++active_profile=debug', '++active_experiment=critical']) validated successfully.
check_configs: resolved paths:  25%|████████████▎                                    | 12/48 [00:02<00:08,  4.49path/s][OK] Resolved experiment path hyperqual (profile=small, overrides=['++active_profile=small', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=small, overrides=['++active_profile=small', '++active_experiment=reinforce_check']) validated successfully.
[OK] Resolved experiment path drift (profile=small, overrides=['++active_profile=small', '++active_experiment=drift']) validated successfully.
check_configs: resolved paths:  31%|███████████████▎                                 | 15/48 [00:03<00:07,  4.46path/s][OK] Resolved experiment path sweep (profile=small, overrides=['++active_profile=small', '++active_experiment=sweep']) validated successfully.
[OK] Resolved experiment path stress (profile=small, overrides=['++active_profile=small', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Resolved experiment path policy (profile=small, overrides=['++active_profile=small', '++active_experiment=policy']) validated successfully.
check_configs: resolved paths:  38%|██████████████████▍                              | 18/48 [00:04<00:06,  4.45path/s][OK] Resolved experiment path bc_train (profile=small, overrides=['++active_profile=small', '++active_experiment=bc_train']) validated successfully.
[OK] Resolved experiment path reinforce_train (profile=small, overrides=['++active_profile=small', '++active_experiment=reinforce_train']) validated successfully.
[OK] Resolved experiment path stats (profile=small, overrides=['++active_profile=small', '++active_experiment=stats']) validated successfully.
check_configs: resolved paths:  44%|█████████████████████▍                           | 21/48 [00:04<00:06,  4.28path/s][OK] Resolved experiment path generalize (profile=small, overrides=['++active_profile=small', '++active_experiment=generalize']) validated successfully.
[OK] Resolved experiment path ablation (profile=small, overrides=['++active_profile=small', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=small, overrides=['++active_profile=small', '++active_experiment=critical']) validated successfully.
check_configs: resolved paths:  50%|████████████████████████▌                        | 24/48 [00:05<00:05,  4.36path/s][OK] Resolved experiment path hyperqual (profile=default, overrides=['++active_profile=default', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=default, overrides=['++active_profile=default', '++active_experiment=reinforce_check']) validated successfully.
[OK] Resolved experiment path drift (profile=default, overrides=['++active_profile=default', '++active_experiment=drift']) validated successfully.
check_configs: resolved paths:  56%|███████████████████████████▌                     | 27/48 [00:06<00:05,  4.07path/s][OK] Resolved experiment path sweep (profile=default, overrides=['++active_profile=default', '++active_experiment=sweep']) validated successfully.
[OK] Resolved experiment path stress (profile=default, overrides=['++active_profile=default', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Resolved experiment path policy (profile=default, overrides=['++active_profile=default', '++active_experiment=policy']) validated successfully.
check_configs: resolved paths:  62%|██████████████████████████████▋                  | 30/48 [00:07<00:04,  3.81path/s][OK] Resolved experiment path bc_train (profile=default, overrides=['++active_profile=default', '++active_experiment=bc_train']) validated successfully.
[OK] Resolved experiment path reinforce_train (profile=default, overrides=['++active_profile=default', '++active_experiment=reinforce_train']) validated successfully.
check_configs: resolved paths:  67%|████████████████████████████████▋                | 32/48 [00:07<00:04,  3.78path/s][OK] Resolved experiment path stats (profile=default, overrides=['++active_profile=default', '++active_experiment=stats']) validated successfully.
[OK] Resolved experiment path generalize (profile=default, overrides=['++active_profile=default', '++active_experiment=generalize']) validated successfully.
check_configs: resolved paths:  71%|██████████████████████████████████▋              | 34/48 [00:08<00:03,  3.74path/s][OK] Resolved experiment path ablation (profile=default, overrides=['++active_profile=default', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=default, overrides=['++active_profile=default', '++active_experiment=critical']) validated successfully.
check_configs: resolved paths:  75%|████████████████████████████████████▊            | 36/48 [00:08<00:03,  3.56path/s][OK] Resolved experiment path hyperqual (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=reinforce_check']) validated successfully.
[OK] Resolved experiment path drift (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=drift']) validated successfully.
check_configs: resolved paths:  81%|███████████████████████████████████████▊         | 39/48 [00:09<00:02,  3.73path/s][OK] Resolved experiment path sweep (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=sweep']) validated successfully.
[OK] Resolved experiment path stress (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Resolved experiment path policy (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=policy']) validated successfully.
check_configs: resolved paths:  88%|██████████████████████████████████████████▉      | 42/48 [00:10<00:01,  3.73path/s][OK] Resolved experiment path bc_train (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=bc_train']) validated successfully.
[OK] Resolved experiment path reinforce_train (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=reinforce_train']) validated successfully.
[OK] Resolved experiment path stats (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=stats']) validated successfully.
check_configs: resolved paths:  94%|█████████████████████████████████████████████▉   | 45/48 [00:11<00:00,  3.87path/s][OK] Resolved experiment path generalize (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=generalize']) validated successfully.
[OK] Resolved experiment path ablation (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=critical']) validated successfully.
check_configs: public paths:   0%|                                                            | 0/12 [00:00<?, ?path/s][OK] Public experiment path hyperqual (base=default, overrides=['++active_profile=default', '++active_experiment=hyperqual']) validated successfully.
[OK] Public experiment path reinforce_check (base=default, overrides=['++active_profile=default', '++active_experiment=reinforce_check']) validated successfully.
check_configs: public paths:  17%|████████▋                                           | 2/12 [00:00<00:03,  3.11path/s][OK] Public experiment path drift (base=default, overrides=['++active_profile=default', '++active_experiment=drift']) validated successfully.
[OK] Public experiment path sweep (base=default, overrides=['++active_profile=default', '++active_experiment=sweep']) validated successfully.
check_configs: public paths:  33%|█████████████████▎                                  | 4/12 [00:01<00:02,  3.26path/s][OK] Public experiment path stress (base=default, overrides=['++active_profile=default', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Public experiment path policy (base=default, overrides=['++active_profile=default', '++active_experiment=policy']) validated successfully.
check_configs: public paths:  50%|██████████████████████████                          | 6/12 [00:01<00:01,  3.35path/s][OK] Public experiment path bc_train (base=default, overrides=['++active_profile=default', '++active_experiment=bc_train']) validated successfully.
[OK] Public experiment path reinforce_train (base=default, overrides=['++active_profile=default', '++active_experiment=reinforce_train']) validated successfully.
check_configs: public paths:  67%|██████████████████████████████████▋                 | 8/12 [00:02<00:01,  3.27path/s][OK] Public experiment path stats (base=default, overrides=['++active_profile=default', '++active_experiment=stats']) validated successfully.
[OK] Public experiment path generalize (base=default, overrides=['++active_profile=default', '++active_experiment=generalize']) validated successfully.
check_configs: public paths:  83%|██████████████████████████████████████████▌        | 10/12 [00:02<00:00,  3.41path/s][OK] Public experiment path ablation (base=default, overrides=['++active_profile=default', '++active_experiment=ablation']) validated successfully.
[OK] Public experiment path critical (base=default, overrides=['++active_profile=default', '++active_experiment=critical']) validated successfully.

[SUCCESS] All configs passed validation.
1/12 check_configs:   8%|██▊                               | 1/12 [00:24<04:27, 24.35s/experiment, alias=check_configs]
[1/11] Running REINFORCE Gradient validation...
2/12 reinforce_check:   8%|██▋                             | 1/12 [00:24<04:27, 24.35s/experiment, alias=check_configs][handoff] Launching reinforce_check (2/12)
==========================================================
 Starting Experiment: reinforce_check
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=reinforce_check
==========================================================
[2026-03-31 22:23:11,196][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\reinforce_check\run_20260331_222311
[2026-03-31 22:23:11,197][__main__][INFO] - ============================================================
[2026-03-31 22:23:11,197][__main__][INFO] -   REINFORCE Gradient Estimator Validation
[2026-03-31 22:23:11,197][__main__][INFO] - ============================================================
[2026-03-31 22:23:11,197][__main__][INFO] - Validating REINFORCE gradients against the trainer-aligned first-action objective.
INFO:2026-03-31 22:23:11,269:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:23:11,269][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:23:14,051][__main__][INFO] - Statistical Scaling: D=50 parameters.
[2026-03-31 22:23:14,052][__main__][INFO] - Adjusted Z-Critical Threshold: 3.29
[2026-03-31 22:23:14,052][__main__][INFO] - Scaled n_samples from 5000 -> 5000 to maintain confidence bounds.
[2026-03-31 22:23:14,052][__main__][INFO] - Computing REINFORCE gradient estimate...
[2026-03-31 22:23:18,309][__main__][INFO] - Computing finite-difference gradient estimate...
reinforce_check: FD params:  12%|██████▏                                             | 6/50 [00:00<00:03, 11.81param/s][2026-03-31 22:23:19,847][__main__][INFO] -   [OK] Param  10/50  (idx  2385): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
                                                                                                                       [2026-03-31 22:23:20,068][__main__][INFO] -   [OK] Param  20/50  (idx 16253): RF=  0.001668 | FD=  0.001258 | diff=  0.000410 | z= 0.00 | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
                                                                                                                       [2026-03-31 22:23:20,228][__main__][INFO] -   [OK] Param  30/50  (idx  2928): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
reinforce_check: FD params:  66%|█████████████████████████████████▋                 | 33/50 [00:01<00:00, 35.91param/s][2026-03-31 22:23:20,383][__main__][INFO] -   [OK] Param  40/50  (idx  1282): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
                                                                                                                       [2026-03-31 22:23:20,534][__main__][INFO] -   [OK] Param  50/50  (idx  9341): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
reinforce_check: FD params: 100%|███████████████████████████████████████████████████| 50/50 [00:01<00:00, 39.11param/s]
[2026-03-31 22:23:20,540][__main__][INFO] - Relative error: 0.3261
[2026-03-31 22:23:20,541][__main__][INFO] - Cosine similarity: 1.0000
[2026-03-31 22:23:20,541][__main__][INFO] - Bias estimate (L2): 0.001190
[2026-03-31 22:23:20,541][__main__][INFO] - Relative bias: 0.3261
[2026-03-31 22:23:20,541][__main__][INFO] - Variance estimate: 0.000000
[2026-03-31 22:23:20,542][__main__][INFO] - Passed: True
[2026-03-31 22:23:20,542][__main__][INFO] - Results saved to outputs\debug\reinforce_check\run_20260331_222311\gradient_check_result.json
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\utils\chart_exporter.py:95: UserWarning: Glyph 10003 (\N{CHECK MARK}) missing from font(s) Times New Roman.
  fig.savefig(
[2026-03-31 22:23:23,047][__main__][INFO] - Gradient scatter plot saved to outputs\debug\reinforce_check\run_20260331_222311\gradient_scatter.png, outputs\debug\reinforce_check\run_20260331_222311\gradient_scatter.pdf
[2026-03-31 22:23:23,047][__main__][INFO] - GRADIENT CHECK PASSED - REINFORCE estimator is valid
2/12 reinforce_check:  17%|█████                         | 2/12 [00:45<03:41, 22.20s/experiment, alias=reinforce_check]
[2/11] Running Drift Verification (Phase 1a)...
3/12 drift:  17%|██████▋                                 | 2/12 [00:45<03:41, 22.20s/experiment, alias=reinforce_check][handoff] Launching drift (3/12)
==========================================================
 Starting Experiment: drift
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=drift
==========================================================
[2026-03-31 22:23:32,666][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-31 22:23:32,687:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:23:32,687][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:23:32,688][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-31 22:23:32,703][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\drift\run_20260331_222332
[2026-03-31 22:23:32,705][__main__][INFO] - System: N=2, lam=1.0, alpha=1.0, cap=2.5000
[2026-03-31 22:23:32,708][__main__][INFO] - Proof bounds: R=2.4431, eps=0.750000
[2026-03-31 22:23:32,708][__main__][INFO] - --- Grid Evaluation (q_max=50) ---
drift: grid:   0%|                                                                            | 0/3 [00:00<?, ?stage/s][2026-03-31 22:23:32,716][__main__][INFO] - States evaluated: 2,601
[2026-03-31 22:23:32,717][__main__][INFO] - Bound violations: 0
[2026-03-31 22:23:33,443][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-31 22:23:33,806][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-31 22:23:34,320][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-31 22:23:34,450][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-31 22:23:34,579][__main__][INFO] - Saved: outputs\debug\drift\run_20260331_222332\drift_heatmap.png, outputs\debug\drift\run_20260331_222332\drift_heatmap.pdf
[2026-03-31 22:23:34,754][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-31 22:23:34,880][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-31 22:23:35,391][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-31 22:23:35,471][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-31 22:23:35,668][__main__][INFO] - Saved: outputs\debug\drift\run_20260331_222332\drift_vs_norm.png, outputs\debug\drift\run_20260331_222332\drift_vs_norm.pdf
drift: grid: 100%|████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.01stage/s, N=2, mode=grid]
[2026-03-31 22:23:35,671][__main__][INFO] - Drift verification complete.
3/12 drift:  25%|████████████▌                                     | 3/12 [00:57<02:40, 17.78s/experiment, alias=drift]
[3/11] Running Stability Sweep (Phase 1b)...
4/12 sweep:  25%|████████████▌                                     | 3/12 [00:57<02:40, 17.78s/experiment, alias=drift][handoff] Launching sweep (4/12)
==========================================================
 Starting Experiment: sweep
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=sweep
==========================================================
[2026-03-31 22:23:45,177][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-31 22:23:45,200:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:23:45,200][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:23:45,201][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-31 22:23:45,216][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\sweep\run_20260331_222345
[2026-03-31 22:23:45,216][gibbsq.utils.logging][INFO] - [Logging] WandB offline mode.
wandb: Tracking run with wandb version 0.23.1
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in outputs\debug\sweep\run_20260331_222345\wandb\offline-run-20260331_222345-pfnnqf1z
[2026-03-31 22:23:49,142][gibbsq.utils.logging][INFO] - [Logging] WandB Run Linked: run_20260331_222345 (offline)
[2026-03-31 22:23:49,146][__main__][INFO] - System: N=2, cap=2.5000 | Backend: JAX
[2026-03-31 22:23:49,147][__main__][INFO] - Grid: 9 alpha x 5 rho x 10 reps
sweep:   0%|                                                                                  | 0/45 [00:00<?, ?cell/s][2026-03-31 22:23:49,155][__main__][INFO] -
------------------------------------------------------------
  rho = 0.80  (lam = 2.0000)
------------------------------------------------------------
                                                                                                                       [2026-03-31 22:23:50,738][__main__][INFO] -   alpha=  0.01 | E[Q_total]=    7.35 | NONSTATIONARY  (1/45)
sweep:   2%|▉                                        | 1/45 [00:01<01:09,  1.58s/cell, rho=0.80, alpha=0.01, done=1/45][2026-03-31 22:23:50,935][__main__][INFO] -   alpha=  0.10 | E[Q_total]=    6.58 | NONSTATIONARY  (2/45)
                                                                                                                       [2026-03-31 22:23:51,141][__main__][INFO] -   alpha=  0.50 | E[Q_total]=    5.31 | NONSTATIONARY  (3/45)
                                                                                                                       [2026-03-31 22:23:51,350][__main__][INFO] -   alpha=  1.00 | E[Q_total]=    5.10 | NONSTATIONARY  (4/45)
sweep:   9%|███▋                                     | 4/45 [00:02<00:19,  2.15cell/s, rho=0.80, alpha=1.00, done=4/45][2026-03-31 22:23:51,548][__main__][INFO] -   alpha=  2.00 | E[Q_total]=    4.91 | OK  (5/45)
                                                                                                                       [2026-03-31 22:23:51,741][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    4.63 | OK  (6/45)
                                                                                                                       [2026-03-31 22:23:52,011][__main__][INFO] -   alpha= 10.00 | E[Q_total]=    4.75 | NONSTATIONARY  (7/45)
sweep:  16%|██████▏                                 | 7/45 [00:02<00:12,  3.00cell/s, rho=0.80, alpha=10.00, done=7/45][2026-03-31 22:23:52,311][__main__][INFO] -   alpha= 50.00 | E[Q_total]=    5.18 | NONSTATIONARY  (8/45)
                                                                                                                       [2026-03-31 22:23:52,611][__main__][INFO] -   alpha=100.00 | E[Q_total]=    5.04 | OK  (9/45)
sweep:  20%|███████▊                               | 9/45 [00:03<00:11,  3.10cell/s, rho=0.80, alpha=100.00, done=9/45][2026-03-31 22:23:52,613][__main__][INFO] -
------------------------------------------------------------
  rho = 0.85  (lam = 2.1250)
------------------------------------------------------------
                                                                                                                       [2026-03-31 22:23:54,118][__main__][INFO] -   alpha=  0.01 | E[Q_total]=    9.52 | NONSTATIONARY  (10/45)
                                                                                                                       [2026-03-31 22:23:54,306][__main__][INFO] -   alpha=  0.10 | E[Q_total]=    8.30 | OK  (11/45)
sweep:  24%|█████████▌                             | 11/45 [00:05<00:16,  2.03cell/s, rho=0.85, alpha=0.10, done=11/45][2026-03-31 22:23:54,531][__main__][INFO] -   alpha=  0.50 | E[Q_total]=    7.09 | NONSTATIONARY  (12/45)
                                                                                                                       [2026-03-31 22:23:54,737][__main__][INFO] -   alpha=  1.00 | E[Q_total]=    7.25 | OK  (13/45)
                                                                                                                       [2026-03-31 22:23:54,929][__main__][INFO] -   alpha=  2.00 | E[Q_total]=    6.88 | NONSTATIONARY  (14/45)
sweep:  31%|████████████▏                          | 14/45 [00:05<00:11,  2.66cell/s, rho=0.85, alpha=2.00, done=14/45][2026-03-31 22:23:55,138][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    6.01 | NONSTATIONARY  (15/45)
                                                                                                                       [2026-03-31 22:23:55,337][__main__][INFO] -   alpha= 10.00 | E[Q_total]=    6.42 | OK  (16/45)
                                                                                                                       [2026-03-31 22:23:55,548][__main__][INFO] -   alpha= 50.00 | E[Q_total]=    8.13 | OK  (17/45)
sweep:  38%|██████████████▎                       | 17/45 [00:06<00:08,  3.19cell/s, rho=0.85, alpha=50.00, done=17/45][2026-03-31 22:23:55,748][__main__][INFO] -   alpha=100.00 | E[Q_total]=    6.68 | OK  (18/45)
[2026-03-31 22:23:55,749][__main__][INFO] -
------------------------------------------------------------
  rho = 0.90  (lam = 2.2500)
------------------------------------------------------------
                                                                                                                       [2026-03-31 22:23:57,167][__main__][INFO] -   alpha=  0.01 | E[Q_total]=   17.44 | NONSTATIONARY  (19/45)
sweep:  42%|████████████████▍                      | 19/45 [00:08<00:11,  2.26cell/s, rho=0.90, alpha=0.01, done=19/45][2026-03-31 22:23:57,581][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   13.90 | OK  (20/45)
                                                                                                                       [2026-03-31 22:23:57,928][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   11.72 | OK  (21/45)
sweep:  47%|██████████████████▏                    | 21/45 [00:08<00:10,  2.35cell/s, rho=0.90, alpha=0.50, done=21/45][2026-03-31 22:23:58,189][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   10.16 | OK  (22/45)
                                                                                                                       [2026-03-31 22:23:58,452][__main__][INFO] -   alpha=  2.00 | E[Q_total]=    9.90 | OK  (23/45)
sweep:  51%|███████████████████▉                   | 23/45 [00:09<00:08,  2.63cell/s, rho=0.90, alpha=2.00, done=23/45][2026-03-31 22:23:58,723][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    9.41 | OK  (24/45)
                                                                                                                       [2026-03-31 22:23:58,952][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   10.07 | OK  (25/45)
sweep:  56%|█████████████████████                 | 25/45 [00:09<00:06,  2.92cell/s, rho=0.90, alpha=10.00, done=25/45][2026-03-31 22:23:59,147][__main__][INFO] -   alpha= 50.00 | E[Q_total]=    9.78 | NONSTATIONARY  (26/45)
                                                                                                                       [2026-03-31 22:23:59,353][__main__][INFO] -   alpha=100.00 | E[Q_total]=   10.50 | OK  (27/45)
[2026-03-31 22:23:59,354][__main__][INFO] -
------------------------------------------------------------
  rho = 0.95  (lam = 2.3750)
------------------------------------------------------------
                                                                                                                       [2026-03-31 22:24:00,639][__main__][INFO] -   alpha=  0.01 | E[Q_total]=   26.52 | NONSTATIONARY  (28/45)
sweep:  62%|████████████████████████▎              | 28/45 [00:11<00:07,  2.35cell/s, rho=0.95, alpha=0.01, done=28/45][2026-03-31 22:24:00,857][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   20.37 | NONSTATIONARY  (29/45)
                                                                                                                       [2026-03-31 22:24:01,055][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   19.48 | NONSTATIONARY  (30/45)
                                                                                                                       [2026-03-31 22:24:01,275][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   19.63 | NONSTATIONARY  (31/45)
sweep:  69%|██████████████████████████▊            | 31/45 [00:12<00:04,  2.85cell/s, rho=0.95, alpha=1.00, done=31/45][2026-03-31 22:24:01,492][__main__][INFO] -   alpha=  2.00 | E[Q_total]=   15.14 | NONSTATIONARY  (32/45)
                                                                                                                       [2026-03-31 22:24:01,713][__main__][INFO] -   alpha=  5.00 | E[Q_total]=   17.48 | NONSTATIONARY  (33/45)
                                                                                                                       [2026-03-31 22:24:01,932][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   13.56 | NONSTATIONARY  (34/45)
sweep:  76%|████████████████████████████▋         | 34/45 [00:12<00:03,  3.26cell/s, rho=0.95, alpha=10.00, done=34/45][2026-03-31 22:24:02,162][__main__][INFO] -   alpha= 50.00 | E[Q_total]=   14.22 | NONSTATIONARY  (35/45)
                                                                                                                       [2026-03-31 22:24:02,372][__main__][INFO] -   alpha=100.00 | E[Q_total]=   16.60 | NONSTATIONARY  (36/45)
[2026-03-31 22:24:02,373][__main__][INFO] -
------------------------------------------------------------
  rho = 0.98  (lam = 2.4500)
------------------------------------------------------------
                                                                                                                       [2026-03-31 22:24:04,147][__main__][INFO] -   alpha=  0.01 | E[Q_total]=   54.15 | NONSTATIONARY  (37/45)
sweep:  82%|████████████████████████████████       | 37/45 [00:14<00:03,  2.24cell/s, rho=0.98, alpha=0.01, done=37/45][2026-03-31 22:24:04,405][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   23.58 | NONSTATIONARY  (38/45)
                                                                                                                       [2026-03-31 22:24:04,664][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   21.20 | NONSTATIONARY  (39/45)
sweep:  87%|█████████████████████████████████▊     | 39/45 [00:15<00:02,  2.49cell/s, rho=0.98, alpha=0.50, done=39/45][2026-03-31 22:24:04,924][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   34.63 | NONSTATIONARY  (40/45)
                                                                                                                       [2026-03-31 22:24:05,141][__main__][INFO] -   alpha=  2.00 | E[Q_total]=   25.32 | NONSTATIONARY  (41/45)
                                                                                                                       [2026-03-31 22:24:05,345][__main__][INFO] -   alpha=  5.00 | E[Q_total]=   30.64 | NONSTATIONARY  (42/45)
sweep:  93%|████████████████████████████████████▍  | 42/45 [00:16<00:01,  2.91cell/s, rho=0.98, alpha=5.00, done=42/45][2026-03-31 22:24:05,556][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   25.00 | NONSTATIONARY  (43/45)
                                                                                                                       [2026-03-31 22:24:05,762][__main__][INFO] -   alpha= 50.00 | E[Q_total]=   38.74 | NONSTATIONARY  (44/45)
                                                                                                                       [2026-03-31 22:24:05,971][__main__][INFO] -   alpha=100.00 | E[Q_total]=   25.68 | NONSTATIONARY  (45/45)
sweep: 100%|█████████████████████████████████████| 45/45 [00:16<00:00,  2.68cell/s, rho=0.98, alpha=100.00, done=45/45]
[2026-03-31 22:24:06,378][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:06,743][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:07,467][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:07,611][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:07,843][__main__][INFO] -
Saved plot: outputs\debug\sweep\run_20260331_222345\alpha_sweep.png, outputs\debug\sweep\run_20260331_222345\alpha_sweep.pdf
[2026-03-31 22:24:08,014][__main__][INFO] -
Summary: 30/45 configurations non-stationary.
wandb:
wandb: Run history:
wandb:                   alpha ▁▁▁▁▁▁▂▄▁▁▁▁▁▁▂▄▁▁▁▁▁▁▂▄▁▁▁▁▁▁▂▄▁▁▁▁▁▁▂█
wandb:                     lam ▁▁▁▁▁▁▁▁▃▃▃▃▃▃▃▃▅▅▅▅▅▅▅▅▇▇▇▇▇▇▇▇████████
wandb:            mean_q_total ▁▁▁▁▁▁▁▁▂▂▁▁▁▁▁▁▃▂▂▂▂▂▂▂▄▃▃▃▂▃▂▂█▄▃▅▄▅▄▄
wandb:        num_replications ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                     rho ▁▁▁▁▁▁▁▁▃▃▃▃▃▃▃▃▅▅▅▅▅▅▅▅▇▇▇▇▇▇▇▇████████
wandb:       stationarity_rate ▄▇▇▇██▇▇▇█▇█▇▇██▇██████▅▄▅▅▅▅▇▇▅▁▄▅▂▄▅▅▄
wandb:  stationarity_threshold ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: stationary_replications ▄▇▇▇██▇▇▇█▇█▇▇██▇██████▅▄▅▅▅▅▇▇▅▁▄▅▂▄▅▅▄
wandb:
wandb: Run summary:
wandb:                   alpha 100
wandb:                 backend JAX
wandb:           is_stationary False
wandb:                     lam 2.45
wandb:            mean_q_total 25.68414
wandb:        num_replications 10
wandb:                     rho 0.98
wandb:       stationarity_rate 0.7
wandb:  stationarity_threshold 1
wandb: stationary_replications 7
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync outputs\debug\sweep\run_20260331_222345\wandb\offline-run-20260331_222345-pfnnqf1z
wandb: Find logs at: outputs\debug\sweep\run_20260331_222345\wandb\offline-run-20260331_222345-pfnnqf1z\logs
4/12 sweep:  33%|████████████████▋                                 | 4/12 [01:30<03:09, 23.63s/experiment, alias=sweep]
[4/11] Running Scaling Stress Tests (Phase 1c)...
5/12 stress:  33%|████████████████▎                                | 4/12 [01:30<03:09, 23.63s/experiment, alias=sweep][handoff] Launching stress (5/12)
==========================================================
 Starting Experiment: stress
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=stress ++jax.enabled=True
==========================================================
[2026-03-31 22:24:17,237][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-31 22:24:17,258:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:24:17,258][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:24:17,259][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-31 22:24:17,271][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\stress\run_20260331_222417
[2026-03-31 22:24:17,273][__main__][INFO] - ============================================================
[2026-03-31 22:24:17,278][__main__][INFO] -   GibbsQ Stress Test (JAX Accelerator Active)
[2026-03-31 22:24:17,278][__main__][INFO] - ============================================================
stress:   0%|                                                                                 | 0/3 [00:00<?, ?stage/s][2026-03-31 22:24:17,283][__main__][INFO] -
[TEST 1] Massive-N Scaling Analysis
                                                                                                                       [2026-03-31 22:24:17,376][__main__][INFO] -   Simulating N=4 experts (rho=0.8)...                  | 0/2 [00:00<?, ?N/s]
                                                                                                                       [2026-03-31 22:24:19,178][__main__][INFO] -     -> Average Gini Imbalance: 0.0244
                                                                                                                       [2026-03-31 22:24:19,246][__main__][INFO] -   Simulating N=8 experts (rho=0.8)...     | 1/2 [00:01<00:01,  1.90s/N, N=4]
                                                                                                                       [2026-03-31 22:24:21,194][__main__][INFO] -     -> Average Gini Imbalance: 0.0288
stress:  33%|████████████████████████▎                                                | 1/3 [00:03<00:07,  3.91s/stage][2026-03-31 22:24:21,196][__main__][INFO] -
[TEST 2] Critical Load Analysis (rho up to 0.9)
                                                                                                                       [2026-03-31 22:24:21,234][__main__][INFO] -   Simulating rho=0.900 (T=2000.0)...                 | 0/1 [00:00<?, ?rho/s]
[2026-03-31 22:24:25,509][__main__][INFO] -     -> Gelman-Rubin R-hat across replicas (post MSER-5 burn-in): 1.0057
                                                                                                                       [2026-03-31 22:24:25,558][__main__][INFO] -     -> Avg E[Q_total]: 21.11 | Stationarity: 9/10
stress:  67%|████████████████████████████████████████████████▋                        | 2/3 [00:08<00:04,  4.18s/stage][2026-03-31 22:24:25,560][__main__][INFO] -
[TEST 3] Extreme Heterogeneity Resilience (100x Speed Gap)
[2026-03-31 22:24:25,573][__main__][INFO] -   Simulating heterogenous setup: mu=[10.   0.1  0.1  0.1]
                                                                                                                       [2026-03-31 22:24:27,495][__main__][INFO] -     -> Mean Queue per Expert: [1.07353308 0.         0.         0.        ]
[2026-03-31 22:24:27,496][__main__][INFO] -     -> Gini: 0.7500
stress: 100%|█████████████████████████████████████████████████████████████████████████| 3/3 [00:10<00:00,  3.40s/stage]
[2026-03-31 22:24:28,687][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:28,806][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:28,981][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:29,038][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:29,233][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:29,306][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:30,480][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:30,529][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:30,664][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:30,734][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:24:30,924][__main__][INFO] - Stress dashboard saved to outputs\debug\stress\run_20260331_222417\stress_dashboard.png, outputs\debug\stress\run_20260331_222417\stress_dashboard.pdf
[2026-03-31 22:24:30,925][__main__][INFO] -
Stress test complete.
5/12 stress:  42%|████████████████████                            | 5/12 [01:52<02:43, 23.31s/experiment, alias=stress]
[5/11] Running Platinum BC Pretraining (Phase 2a)...
6/12 bc_train:  42%|███████████████████▏                          | 5/12 [01:52<02:43, 23.31s/experiment, alias=stress][handoff] Launching bc_train (6/12)
==========================================================
 Starting Experiment: bc_train
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=bc_train
==========================================================
[2026-03-31 22:24:40,821][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-31 22:24:40,844:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:24:40,844][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:24:40,844][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-31 22:24:40,857][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\bc_train\run_20260331_222440
[2026-03-31 22:24:42,265][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-31 22:24:43,692][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
bc_train:   0%|▎                                                                     | 1/201 [00:00<02:16,  1.46step/s][2026-03-31 22:24:44,381][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6862 | Acc: 22.50%
[2026-03-31 22:24:44,623][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5651 | Acc: 93.94%
[2026-03-31 22:24:44,874][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5583 | Acc: 99.39%
bc_train: 100%|██████████████████████████████████████████| 201/201 [00:01<00:00, 170.61step/s, loss=0.5583, acc=99.39%]
[2026-03-31 22:24:44,880][__main__][INFO] -
[DONE] Platinum BC Weights saved to outputs\debug\bc_train\run_20260331_222440\n_gibbsq_platinum_bc_weights.eqx
[2026-03-31 22:24:44,881][__main__][INFO] - [Metadata] BC warm-start compatibility metadata saved to outputs\debug\bc_train\run_20260331_222440\n_gibbsq_platinum_bc_weights.eqx.bc_metadata.json
[2026-03-31 22:24:44,884][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_bc_weights.txt at outputs\debug\latest_bc_weights.txt
6/12 bc_train:  50%|██████████████████████                      | 6/12 [02:06<02:00, 20.13s/experiment, alias=bc_train]
[6/11] Running REINFORCE SSA Training (Phase 2b)...
7/12 reinforce_train:  50%|██████████████████▌                  | 6/12 [02:06<02:00, 20.13s/experiment, alias=bc_train][handoff] Launching reinforce_train (7/12)
==========================================================
 Starting Experiment: reinforce_train
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=reinforce_train
==========================================================
[2026-03-31 22:24:54,425][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-31 22:24:54,446:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:24:54,446][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:24:54,446][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-31 22:24:54,459][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\reinforce_train\run_20260331_222454
reinforce: setup:   0%|                                                                       | 0/3 [00:00<?, ?stage/s][2026-03-31 22:24:57,331][__main__][INFO] -   JSQ Mean Queue (Target): 1.0803
[2026-03-31 22:24:57,336][__main__][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-03-31 22:24:57,353][__main__][INFO] - Reusing BC warm-start actor weights from C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\bc_train\run_20260331_222440\n_gibbsq_platinum_bc_weights.eqx
[2026-03-31 22:24:57,354][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-31 22:24:58,785][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
                                                                                                                       [2026-03-31 22:24:59,274][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 1046275.3125 0/201 [00:00<?, ?step/s]
                                                                                                                       [2026-03-31 22:24:59,528][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 770842.1250step/s, loss=1044463.6875]
[2026-03-31 22:24:59,786][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 626511.8750
bc_value: 100%|█████████████████████████████████████████████████| 201/201 [00:00<00:00, 201.57step/s, loss=626511.8750]
reinforce: setup:  67%|██████████████████████████████████████████                     | 2/3 [00:02<00:01,  1.38s/stage][2026-03-31 22:24:59,793][__main__][INFO] - ============================================================
[2026-03-31 22:24:59,794][__main__][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-03-31 22:24:59,795][__main__][INFO] - ============================================================
[2026-03-31 22:24:59,795][__main__][INFO] -   Epochs: 5, Batch size: 4
[2026-03-31 22:24:59,795][__main__][INFO] -   Simulation time: 1000.0
[2026-03-31 22:24:59,796][__main__][INFO] - ------------------------------------------------------------
reinforce_train:   0%|                                                                        | 0/5 [00:00<?, ?epoch/s][2026-03-31 22:25:15,825][__main__][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: 0.0343 | mean_logp: -0.5753 | corr: -0.0126
[2026-03-31 22:25:15,826][__main__][INFO] -     [Grad Check] P-Grad Norm: 0.0527 | V-Grad Norm: 10186.9658
reinforce_train:  20%|███████▊                               | 1/5 [01:09<04:37, 69.43s/epoch, queue=2.522, pi=-204.3%]
[2026-03-31 22:26:13,369][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\debug\latest_reinforce_weights.txt
[2026-03-31 22:26:13,370][__main__][INFO] - -------------------------------------------------------
[2026-03-31 22:26:13,370][__main__][INFO] - -------------------------------------------------------
[2026-03-31 22:26:13,371][__main__][INFO] - Running Final Deterministic Evaluation (N=3)...
[2026-03-31 22:26:15,230][__main__][INFO] - Stage profile written to outputs\debug\reinforce_train\run_20260331_222454\reinforce_stage_profile.json
[2026-03-31 22:26:15,232][__main__][INFO] - Deterministic Policy Score: 93.87% ± 17.32%
[2026-03-31 22:26:15,232][__main__][INFO] - JSQ Target: 100.0% | Random Floor: 0.0% (Performance Index Scale)
[2026-03-31 22:26:15,232][__main__][INFO] - -------------------------------------------------------
[2026-03-31 22:26:15,233][__main__][INFO] - Training Complete! Final Loss: 0.0066
[2026-03-31 22:26:15,233][__main__][INFO] - Final Base-Regime Index Proxy: -204.32
[2026-03-31 22:26:15,233][__main__][INFO] - Policy weights: outputs\debug\reinforce_train\run_20260331_222454\n_gibbsq_reinforce_weights.eqx
[2026-03-31 22:26:15,234][__main__][INFO] - Value weights: outputs\debug\reinforce_train\run_20260331_222454\value_network_weights.eqx
7/12 reinforce_train:  58%|█████████████████▌            | 7/12 [03:39<03:39, 43.82s/experiment, alias=reinforce_train]
[7/11] Running Corrected Policy Evaluation Benchmark (Phase 3a)...
8/12 policy:  58%|██████████████████████▊                | 7/12 [03:39<03:39, 43.82s/experiment, alias=reinforce_train][handoff] Launching policy (8/12)
==========================================================
 Starting Experiment: policy
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=policy
==========================================================
[2026-03-31 22:26:27,351][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-31 22:26:27,372:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:26:27,372][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:26:27,373][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-31 22:26:27,387][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\policy\run_20260331_222627
[2026-03-31 22:26:27,388][gibbsq.utils.logging][INFO] - [Logging] WandB offline mode.
wandb: Tracking run with wandb version 0.23.1
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in outputs\debug\policy\run_20260331_222627\wandb\offline-run-20260331_222627-ri9spek7
[2026-03-31 22:26:30,550][gibbsq.utils.logging][INFO] - [Logging] WandB Run Linked: run_20260331_222627 (offline)
[2026-03-31 22:26:30,552][__main__][INFO] - ============================================================
[2026-03-31 22:26:30,553][__main__][INFO] -   Corrected Policy Comparison
[2026-03-31 22:26:30,554][__main__][INFO] - ============================================================
[2026-03-31 22:26:30,555][__main__][INFO] - System: N=2, lambda=1.0000, Lambda=2.5000, rho=0.4000
[2026-03-31 22:26:30,556][__main__][INFO] - ------------------------------------------------------------
policy: tiers:   0%|                                                                         | 0/7 [00:00<?, ?policy/s][2026-03-31 22:26:30,561][__main__][INFO] - Evaluating Tier 2: JSQ (Min Queue)...
                                                                                                                       [2026-03-31 22:26:30,566][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 22:26:30,689][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-03-31 22:26:30,691][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-31 22:26:30,812][gibbsq.engines.numpy_engine][INFO] -   -> 993 arrivals, 993 departures, final Q_total = 0
[2026-03-31 22:26:30,813][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-31 22:26:30,930][gibbsq.engines.numpy_engine][INFO] -   -> 1,000 arrivals, 1,000 departures, final Q_total = 0
[2026-03-31 22:26:30,931][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-31 22:26:31,059][gibbsq.engines.numpy_engine][INFO] -   -> 1,051 arrivals, 1,050 departures, final Q_total = 1
[2026-03-31 22:26:31,060][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-03-31 22:26:31,183][gibbsq.engines.numpy_engine][INFO] -   -> 1,039 arrivals, 1,039 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:26:31,185][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:00<00:00,  8.09rep/s]
[2026-03-31 22:26:31,301][gibbsq.engines.numpy_engine][INFO] -   -> 915 arrivals, 915 departures, final Q_total = 0
[2026-03-31 22:26:31,303][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-31 22:26:31,426][gibbsq.engines.numpy_engine][INFO] -   -> 986 arrivals, 986 departures, final Q_total = 0
[2026-03-31 22:26:31,427][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-31 22:26:31,554][gibbsq.engines.numpy_engine][INFO] -   -> 1,026 arrivals, 1,025 departures, final Q_total = 1
[2026-03-31 22:26:31,555][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-03-31 22:26:31,683][gibbsq.engines.numpy_engine][INFO] -   -> 1,033 arrivals, 1,030 departures, final Q_total = 3
[2026-03-31 22:26:31,684][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-31 22:26:31,814][gibbsq.engines.numpy_engine][INFO] -   -> 1,058 arrivals, 1,058 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:26:31,821][__main__][INFO] -   E[Q_total] = 1.0603 ± 0.0313
policy: tiers:  14%|█████████▎                                                       | 1/7 [00:01<00:07,  1.27s/policy][2026-03-31 22:26:31,829][__main__][INFO] - Evaluating Tier 2: JSSQ (Min Sojourn)...
                                                                                                                       [2026-03-31 22:26:31,832][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 22:26:31,962][gibbsq.engines.numpy_engine][INFO] -   -> 1,012 arrivals, 1,011 departures, final Q_total = 1
[2026-03-31 22:26:31,963][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-31 22:26:32,097][gibbsq.engines.numpy_engine][INFO] -   -> 1,008 arrivals, 1,007 departures, final Q_total = 1
[2026-03-31 22:26:32,098][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-31 22:26:32,220][gibbsq.engines.numpy_engine][INFO] -   -> 981 arrivals, 975 departures, final Q_total = 6
[2026-03-31 22:26:32,222][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-31 22:26:32,352][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,050 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:26:32,355][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:00<00:00,  7.69rep/s]
[2026-03-31 22:26:32,482][gibbsq.engines.numpy_engine][INFO] -   -> 1,036 arrivals, 1,036 departures, final Q_total = 0
[2026-03-31 22:26:32,483][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-03-31 22:26:32,608][gibbsq.engines.numpy_engine][INFO] -   -> 922 arrivals, 922 departures, final Q_total = 0
[2026-03-31 22:26:32,610][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-31 22:26:32,761][gibbsq.engines.numpy_engine][INFO] -   -> 989 arrivals, 989 departures, final Q_total = 0
[2026-03-31 22:26:32,762][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-31 22:26:32,898][gibbsq.engines.numpy_engine][INFO] -   -> 1,023 arrivals, 1,022 departures, final Q_total = 1
                                                                                                                       [2026-03-31 22:26:32,902][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:01<00:00,  7.47rep/s]
[2026-03-31 22:26:33,033][gibbsq.engines.numpy_engine][INFO] -   -> 1,046 arrivals, 1,046 departures, final Q_total = 0
[2026-03-31 22:26:33,034][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-31 22:26:33,165][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,050 departures, final Q_total = 4
                                                                                                                       [2026-03-31 22:26:33,172][__main__][INFO] -   E[Q_total] = 0.9949 ± 0.0294
policy: tiers:  29%|██████████████████▌                                              | 2/7 [00:02<00:06,  1.31s/policy][2026-03-31 22:26:33,176][__main__][INFO] - Evaluating Tier 3: UAS (alpha=1.0)...
                                                                                                                       [2026-03-31 22:26:33,179][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 22:26:33,317][gibbsq.engines.numpy_engine][INFO] -   -> 1,004 arrivals, 1,001 departures, final Q_total = 3
[2026-03-31 22:26:33,318][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-31 22:26:33,451][gibbsq.engines.numpy_engine][INFO] -   -> 997 arrivals, 996 departures, final Q_total = 1
[2026-03-31 22:26:33,452][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-31 22:26:33,584][gibbsq.engines.numpy_engine][INFO] -   -> 983 arrivals, 983 departures, final Q_total = 0
[2026-03-31 22:26:33,586][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-31 22:26:33,723][gibbsq.engines.numpy_engine][INFO] -   -> 1,044 arrivals, 1,044 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:26:33,726][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:00<00:00,  7.35rep/s]
[2026-03-31 22:26:33,875][gibbsq.engines.numpy_engine][INFO] -   -> 1,031 arrivals, 1,029 departures, final Q_total = 2
[2026-03-31 22:26:33,876][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-03-31 22:26:34,003][gibbsq.engines.numpy_engine][INFO] -   -> 909 arrivals, 909 departures, final Q_total = 0
[2026-03-31 22:26:34,004][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-31 22:26:34,142][gibbsq.engines.numpy_engine][INFO] -   -> 991 arrivals, 991 departures, final Q_total = 0
[2026-03-31 22:26:34,143][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-31 22:26:34,277][gibbsq.engines.numpy_engine][INFO] -   -> 1,015 arrivals, 1,015 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:26:34,281][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:01<00:00,  7.26rep/s]
[2026-03-31 22:26:34,448][gibbsq.engines.numpy_engine][INFO] -   -> 1,041 arrivals, 1,040 departures, final Q_total = 1
[2026-03-31 22:26:34,451][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-31 22:26:34,722][gibbsq.engines.numpy_engine][INFO] -   -> 1,058 arrivals, 1,057 departures, final Q_total = 1
                                                                                                                       [2026-03-31 22:26:34,727][__main__][INFO] -   E[Q_total] = 1.1298 ± 0.0372
policy: tiers:  43%|███████████████████████████▊                                     | 3/7 [00:04<00:05,  1.42s/policy][2026-03-31 22:26:34,734][__main__][INFO] - Evaluating Tier 3: UAS (alpha=10.0)...
                                                                                                                       [2026-03-31 22:26:34,738][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 22:26:34,930][gibbsq.engines.numpy_engine][INFO] -   -> 1,012 arrivals, 1,011 departures, final Q_total = 1
[2026-03-31 22:26:34,932][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-31 22:26:35,086][gibbsq.engines.numpy_engine][INFO] -   -> 1,007 arrivals, 1,007 departures, final Q_total = 0
[2026-03-31 22:26:35,087][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-31 22:26:35,220][gibbsq.engines.numpy_engine][INFO] -   -> 983 arrivals, 983 departures, final Q_total = 0
[2026-03-31 22:26:35,221][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-31 22:26:35,363][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,049 departures, final Q_total = 1
                                                                                                                       [2026-03-31 22:26:35,365][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:00<00:00,  6.40rep/s]
[2026-03-31 22:26:35,501][gibbsq.engines.numpy_engine][INFO] -   -> 1,036 arrivals, 1,036 departures, final Q_total = 0
[2026-03-31 22:26:35,502][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-03-31 22:26:35,630][gibbsq.engines.numpy_engine][INFO] -   -> 921 arrivals, 920 departures, final Q_total = 1
[2026-03-31 22:26:35,632][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-31 22:26:35,764][gibbsq.engines.numpy_engine][INFO] -   -> 988 arrivals, 988 departures, final Q_total = 0
[2026-03-31 22:26:35,766][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-31 22:26:35,900][gibbsq.engines.numpy_engine][INFO] -   -> 1,020 arrivals, 1,020 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:26:35,905][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:01<00:00,  6.97rep/s]
[2026-03-31 22:26:36,050][gibbsq.engines.numpy_engine][INFO] -   -> 1,045 arrivals, 1,044 departures, final Q_total = 1
[2026-03-31 22:26:36,051][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-31 22:26:36,197][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,051 departures, final Q_total = 3
                                                                                                                       [2026-03-31 22:26:36,203][__main__][INFO] -   E[Q_total] = 1.0100 ± 0.0304
policy: tiers:  57%|█████████████████████████████████████▏                           | 4/7 [00:05<00:04,  1.44s/policy][2026-03-31 22:26:36,204][__main__][INFO] - Evaluating Tier 3: UAS (alpha=5.0)...
                                                                                                                       [2026-03-31 22:26:36,208][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 22:26:36,336][gibbsq.engines.numpy_engine][INFO] -   -> 1,007 arrivals, 1,007 departures, final Q_total = 0
[2026-03-31 22:26:36,338][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-31 22:26:36,481][gibbsq.engines.numpy_engine][INFO] -   -> 1,006 arrivals, 1,004 departures, final Q_total = 2
[2026-03-31 22:26:36,482][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-31 22:26:36,621][gibbsq.engines.numpy_engine][INFO] -   -> 971 arrivals, 971 departures, final Q_total = 0
[2026-03-31 22:26:36,622][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-31 22:26:36,758][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,049 departures, final Q_total = 1
                                                                                                                       [2026-03-31 22:26:36,762][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:00<00:00,  7.26rep/s]
[2026-03-31 22:26:36,904][gibbsq.engines.numpy_engine][INFO] -   -> 1,032 arrivals, 1,030 departures, final Q_total = 2
[2026-03-31 22:26:36,905][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-03-31 22:26:37,037][gibbsq.engines.numpy_engine][INFO] -   -> 914 arrivals, 914 departures, final Q_total = 0
[2026-03-31 22:26:37,039][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-31 22:26:37,175][gibbsq.engines.numpy_engine][INFO] -   -> 986 arrivals, 985 departures, final Q_total = 1
[2026-03-31 22:26:37,177][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-31 22:26:37,316][gibbsq.engines.numpy_engine][INFO] -   -> 1,015 arrivals, 1,015 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:26:37,319][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:01<00:00,  7.21rep/s]
[2026-03-31 22:26:37,458][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,053 departures, final Q_total = 1
[2026-03-31 22:26:37,460][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-31 22:26:37,602][gibbsq.engines.numpy_engine][INFO] -   -> 1,049 arrivals, 1,047 departures, final Q_total = 2
                                                                                                                       [2026-03-31 22:26:37,608][__main__][INFO] -   E[Q_total] = 1.0250 ± 0.0335
policy: tiers:  71%|██████████████████████████████████████████████▍                  | 5/7 [00:07<00:02,  1.43s/policy][2026-03-31 22:26:37,611][__main__][INFO] - Evaluating Tier 4: Proportional (mu/Lambda)...
                                                                                                                       [2026-03-31 22:26:37,615][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 22:26:37,717][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,008 departures, final Q_total = 2
[2026-03-31 22:26:37,718][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-31 22:26:37,825][gibbsq.engines.numpy_engine][INFO] -   -> 1,017 arrivals, 1,016 departures, final Q_total = 1
[2026-03-31 22:26:37,826][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-31 22:26:37,926][gibbsq.engines.numpy_engine][INFO] -   -> 974 arrivals, 973 departures, final Q_total = 1
[2026-03-31 22:26:37,928][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-31 22:26:38,042][gibbsq.engines.numpy_engine][INFO] -   -> 1,040 arrivals, 1,040 departures, final Q_total = 0
[2026-03-31 22:26:38,045][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-03-31 22:26:38,146][gibbsq.engines.numpy_engine][INFO] -   -> 1,025 arrivals, 1,024 departures, final Q_total = 1
                                                                                                                       [2026-03-31 22:26:38,150][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:00<00:00,  9.39rep/s]
[2026-03-31 22:26:38,241][gibbsq.engines.numpy_engine][INFO] -   -> 902 arrivals, 894 departures, final Q_total = 8
[2026-03-31 22:26:38,243][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-31 22:26:38,344][gibbsq.engines.numpy_engine][INFO] -   -> 989 arrivals, 988 departures, final Q_total = 1
[2026-03-31 22:26:38,345][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-31 22:26:38,450][gibbsq.engines.numpy_engine][INFO] -   -> 1,027 arrivals, 1,027 departures, final Q_total = 0
[2026-03-31 22:26:38,451][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-03-31 22:26:38,559][gibbsq.engines.numpy_engine][INFO] -   -> 1,025 arrivals, 1,023 departures, final Q_total = 2
[2026-03-31 22:26:38,560][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-31 22:26:38,674][gibbsq.engines.numpy_engine][INFO] -   -> 1,068 arrivals, 1,065 departures, final Q_total = 3
                                                                                                                       [2026-03-31 22:26:38,680][__main__][INFO] -   E[Q_total] = 1.3704 ± 0.0569
policy: tiers:  86%|███████████████████████████████████████████████████████▋         | 6/7 [00:08<00:01,  1.31s/policy][2026-03-31 22:26:38,683][__main__][INFO] - Evaluating Tier 4: Uniform (1/N)...
                                                                                                                       [2026-03-31 22:26:38,687][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 22:26:38,797][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-03-31 22:26:38,799][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-31 22:26:38,912][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-03-31 22:26:38,913][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-31 22:26:39,028][gibbsq.engines.numpy_engine][INFO] -   -> 991 arrivals, 990 departures, final Q_total = 1
[2026-03-31 22:26:39,029][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-31 22:26:39,141][gibbsq.engines.numpy_engine][INFO] -   -> 1,040 arrivals, 1,040 departures, final Q_total = 0
[2026-03-31 22:26:39,142][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-03-31 22:26:39,254][gibbsq.engines.numpy_engine][INFO] -   -> 1,027 arrivals, 1,026 departures, final Q_total = 1
                                                                                                                       [2026-03-31 22:26:39,257][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:00<00:00,  8.79rep/s]
[2026-03-31 22:26:39,366][gibbsq.engines.numpy_engine][INFO] -   -> 911 arrivals, 911 departures, final Q_total = 0
[2026-03-31 22:26:39,367][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-31 22:26:39,476][gibbsq.engines.numpy_engine][INFO] -   -> 993 arrivals, 991 departures, final Q_total = 2
[2026-03-31 22:26:39,477][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-31 22:26:39,586][gibbsq.engines.numpy_engine][INFO] -   -> 1,029 arrivals, 1,027 departures, final Q_total = 2
[2026-03-31 22:26:39,587][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-03-31 22:26:39,778][gibbsq.engines.numpy_engine][INFO] -   -> 1,020 arrivals, 1,016 departures, final Q_total = 4
[2026-03-31 22:26:39,780][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-31 22:26:39,976][gibbsq.engines.numpy_engine][INFO] -   -> 1,058 arrivals, 1,058 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:26:39,986][__main__][INFO] -   E[Q_total] = 1.5965 ± 0.0857
policy: tiers: 100%|█████████████████████████████████████████████████████████████████| 7/7 [00:09<00:00,  1.35s/policy]
[2026-03-31 22:26:39,994][__main__][INFO] -
Evaluating Tier 5: N-GibbsQ (REINFORCE trained)...
[2026-03-31 22:26:39,995][__main__][INFO] - Using neural weights from C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\reinforce_train\run_20260331_222454\n_gibbsq_reinforce_weights.eqx
[2026-03-31 22:26:41,514][__main__][INFO] - Evaluating N-GibbsQ (deterministic)...
policy eval (DeterministicNeuralPolicy):   0%|                                                 | 0/10 [00:00<?, ?rep/s][2026-03-31 22:26:41,517][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)
[2026-03-31 22:26:42,205][gibbsq.engines.numpy_engine][INFO] -   -> 1,013 arrivals, 1,012 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  10%|████                                     | 1/10 [00:00<00:06,  1.45rep/s][2026-03-31 22:26:42,207][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-31 22:26:42,876][gibbsq.engines.numpy_engine][INFO] -   -> 1,015 arrivals, 1,015 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  20%|████████▏                                | 2/10 [00:01<00:05,  1.47rep/s][2026-03-31 22:26:42,879][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-31 22:26:43,524][gibbsq.engines.numpy_engine][INFO] -   -> 961 arrivals, 961 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  30%|████████████▎                            | 3/10 [00:02<00:04,  1.50rep/s][2026-03-31 22:26:43,527][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-31 22:26:44,221][gibbsq.engines.numpy_engine][INFO] -   -> 1,043 arrivals, 1,042 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  40%|████████████████▍                        | 4/10 [00:02<00:04,  1.48rep/s][2026-03-31 22:26:44,223][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-03-31 22:26:44,901][gibbsq.engines.numpy_engine][INFO] -   -> 1,016 arrivals, 1,016 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  50%|████████████████████▌                    | 5/10 [00:03<00:03,  1.47rep/s][2026-03-31 22:26:44,904][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-03-31 22:26:45,790][gibbsq.engines.numpy_engine][INFO] -   -> 908 arrivals, 907 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  60%|████████████████████████▌                | 6/10 [00:04<00:02,  1.33rep/s][2026-03-31 22:26:45,792][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-31 22:26:46,471][gibbsq.engines.numpy_engine][INFO] -   -> 983 arrivals, 982 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  70%|████████████████████████████▋            | 7/10 [00:04<00:02,  1.37rep/s][2026-03-31 22:26:46,474][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-31 22:26:47,157][gibbsq.engines.numpy_engine][INFO] -   -> 1,023 arrivals, 1,022 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  80%|████████████████████████████████▊        | 8/10 [00:05<00:01,  1.40rep/s][2026-03-31 22:26:47,159][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-03-31 22:26:47,877][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,048 departures, final Q_total = 2
policy eval (DeterministicNeuralPolicy):  90%|████████████████████████████████████▉    | 9/10 [00:06<00:00,  1.40rep/s][2026-03-31 22:26:47,879][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-31 22:26:48,622][gibbsq.engines.numpy_engine][INFO] -   -> 1,074 arrivals, 1,074 departures, final Q_total = 0
[2026-03-31 22:26:48,630][__main__][INFO] -   E[Q_total] = 1.1137 ± 0.0397
[2026-03-31 22:26:48,631][__main__][INFO] -
============================================================
[2026-03-31 22:26:48,631][__main__][INFO] -   Parity Analysis (Corrected Criteria)
[2026-03-31 22:26:48,632][__main__][INFO] - ============================================================
[2026-03-31 22:26:48,633][__main__][INFO] - N-GibbsQ (Platinum/Greedy): E[Q] = 1.1137
[2026-03-31 22:26:48,634][__main__][INFO] - Reference thresholds:
[2026-03-31 22:26:48,635][__main__][INFO] -   JSSQ (Tier 2): E[Q] = 0.9949
[2026-03-31 22:26:48,636][__main__][INFO] -   UAS (Tier 3): E[Q] = 1.1298
[2026-03-31 22:26:48,637][__main__][INFO] -   Proportional (Tier 4): E[Q] = 1.3704
[2026-03-31 22:26:48,637][__main__][INFO] - Reference statistical bounds (95% CI):
[2026-03-31 22:26:48,638][__main__][INFO] -   JSSQ (Tier 2): E[Q] = 0.9949 ± 0.0294
[2026-03-31 22:26:48,639][__main__][INFO] -   UAS (Tier 3): E[Q] = 1.1298 ± 0.0372
[2026-03-31 22:26:48,640][__main__][INFO] -   Proportional (Tier 4): E[Q] = 1.3704 ± 0.0569
[2026-03-31 22:26:48,641][__main__][INFO] - PARITY RESULT: SILVER [OK] (Statistically matches empirical UAS baseline)
[2026-03-31 22:26:50,067][__main__][INFO] - Comparison plot saved to outputs\debug\policy\run_20260331_222627\corrected_policy_comparison.png, outputs\debug\policy\run_20260331_222627\corrected_policy_comparison.pdf
wandb: You can sync this run to the cloud by running:
wandb: wandb sync outputs\debug\policy\run_20260331_222627\wandb\offline-run-20260331_222627-ri9spek7
wandb: Find logs at: outputs\debug\policy\run_20260331_222627\wandb\offline-run-20260331_222627-ri9spek7\logs
8/12 policy:  67%|████████████████████████████████                | 8/12 [04:12<02:40, 40.24s/experiment, alias=policy]
[8/11] Running Statistical Verification Analysis (Phase 3b)...
9/12 stats:  67%|████████████████████████████████▋                | 8/12 [04:12<02:40, 40.24s/experiment, alias=policy][handoff] Launching stats (9/12)
==========================================================
 Starting Experiment: stats
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=stats
==========================================================
[2026-03-31 22:26:59,713][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-31 22:26:59,735:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:26:59,735][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:26:59,735][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-31 22:26:59,748][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\stats\run_20260331_222659
[2026-03-31 22:26:59,749][__main__][INFO] - ============================================================
[2026-03-31 22:26:59,752][__main__][INFO] -   Phase VII: Statistical Summary
[2026-03-31 22:26:59,753][__main__][INFO] - ============================================================
[2026-03-31 22:26:59,953][__main__][INFO] - Initiating statistical comparison (n=10 seeds).
[2026-03-31 22:26:59,990][__main__][INFO] - Environment: N=2, rho=0.40
[2026-03-31 22:27:01,238][__main__][INFO] - Loaded trained model from C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\reinforce_train\run_20260331_222454\n_gibbsq_reinforce_weights.eqx
stats:   0%|                                                                                  | 0/2 [00:00<?, ?stage/s][2026-03-31 22:27:01,244][__main__][INFO] - Running 10 GibbsQ SSA simulations with policy='uas'...
stats:  50%|█████████████████████████████████████                                     | 1/2 [00:01<00:01,  1.84s/stage][2026-03-31 22:27:03,087][__main__][INFO] - Running 10 Neural SSA simulations...
[2026-03-31 22:27:03,088][__main__][INFO] - Neural evaluation mode: deterministic
stats: 100%|██████████████████████████████████████████████████████████████████████████| 2/2 [00:08<00:00,  4.47s/stage]
[2026-03-31 22:27:10,205][__main__][INFO] -
============================================================
[2026-03-31 22:27:10,205][__main__][INFO] -   STATISTICAL SUMMARY
[2026-03-31 22:27:10,206][__main__][INFO] - ============================================================
[2026-03-31 22:27:10,206][__main__][INFO] - GibbsQ E[Q]:   1.0834 ± 0.0911
[2026-03-31 22:27:10,206][__main__][INFO] - N-GibbsQ E[Q]:   1.1137 ± 0.1254
[2026-03-31 22:27:10,206][__main__][INFO] - Rel. Improve:  -2.80%
[2026-03-31 22:27:10,206][__main__][INFO] - ----------------------------------------
[2026-03-31 22:27:10,207][__main__][INFO] - P-Value:       5.44e-01 (NOT SIGNIFICANT)
[2026-03-31 22:27:10,207][__main__][INFO] - Effect Size:   0.28 (Cohen's d)
[2026-03-31 22:27:10,207][__main__][INFO] - 95% CI (Diff): [-0.0726, 0.1333]
[2026-03-31 22:27:10,208][__main__][INFO] - ============================================================
[2026-03-31 22:27:10,812][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:27:10,889][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:27:10,974][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:27:11,639][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:27:11,698][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
9/12 stats:  75%|█████████████████████████████████████▌            | 9/12 [04:33<01:43, 34.47s/experiment, alias=stats]
[9/11] Running Generalization Stress Heatmaps...
10/12 generalize:  75%|█████████████████████████████████           | 9/12 [04:33<01:43, 34.47s/experiment, alias=stats][handoff] Launching generalize (10/12)
==========================================================
 Starting Experiment: generalize
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=generalize
==========================================================
[2026-03-31 22:27:21,849][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-31 22:27:21,871:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:27:21,871][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:27:21,872][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-31 22:27:21,887][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\generalize\run_20260331_222721
[2026-03-31 22:27:21,888][__main__][INFO] - ============================================================
[2026-03-31 22:27:21,891][__main__][INFO] -   Phase VIII: Generalization & Stress Heatmap
[2026-03-31 22:27:21,891][__main__][INFO] - ============================================================
[2026-03-31 22:27:22,077][__main__][INFO] - Initiating Generalization Sweep (Scales=[0.5, 2.0], rho=[0.5, 0.85])
[2026-03-31 22:27:23,392][__main__][INFO] - Evaluating N-GibbsQ improvement ratio (GibbsQ / Neural) on 5x5 Grid...
generalize:   0%|                                                                              | 0/4 [00:00<?, ?cell/s][2026-03-31 22:27:29,630][__main__][INFO] -    Scale=  0.5x | rho=0.50 | Improvement=1.07x
generalize:  25%|███████████▊                                   | 1/4 [00:06<00:18,  6.24s/cell, scale=0.50x, rho=0.50][2026-03-31 22:27:38,110][__main__][INFO] -    Scale=  0.5x | rho=0.85 | Improvement=1.03x
generalize:  50%|███████████████████████▌                       | 2/4 [00:14<00:15,  7.55s/cell, scale=0.50x, rho=0.85][2026-03-31 22:27:56,163][__main__][INFO] -    Scale=  2.0x | rho=0.50 | Improvement=1.07x
generalize:  75%|███████████████████████████████████▎           | 3/4 [00:32<00:12, 12.35s/cell, scale=2.00x, rho=0.50][2026-03-31 22:28:26,393][__main__][INFO] -    Scale=  2.0x | rho=0.85 | Improvement=0.88x
generalize: 100%|███████████████████████████████████████████████| 4/4 [01:02<00:00, 15.75s/cell, scale=2.00x, rho=0.85]
[2026-03-31 22:28:28,509][__main__][INFO] - Generalization analysis complete. Heatmap saved to outputs\debug\generalize\run_20260331_222721\generalization_heatmap.png, outputs\debug\generalize\run_20260331_222721\generalization_heatmap.pdf
10/12 generalize:  83%|███████████████████████████████▋      | 10/12 [05:50<01:34, 47.50s/experiment, alias=generalize]
[10/11] Running Critical Load Boundary Analysis...
11/12 critical:  83%|█████████████████████████████████▎      | 10/12 [05:50<01:34, 47.50s/experiment, alias=generalize][handoff] Launching critical (11/12)
==========================================================
 Starting Experiment: critical
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=critical
==========================================================
[2026-03-31 22:28:38,470][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-31 22:28:38,492:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:28:38,492][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:28:38,493][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-31 22:28:38,507][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\critical\run_20260331_222838
[2026-03-31 22:28:38,508][__main__][INFO] - ============================================================
[2026-03-31 22:28:38,511][__main__][INFO] -   Phase VIII: The Critical Stability Boundary
[2026-03-31 22:28:38,512][__main__][INFO] - ============================================================
[2026-03-31 22:28:39,958][__main__][INFO] - System Capacity: 2.50
[2026-03-31 22:28:39,959][__main__][INFO] - Targeting Load Boundary: [0.95]
critical:   0%|                                                                                 | 0/1 [00:00<?, ?rho/s][2026-03-31 22:28:39,978][__main__][INFO] - Evaluating Boundary rho=0.950 (Arrival=2.375)...
                                                                                                                       [2026-03-31 22:29:47,762][__main__][INFO] -    => N-GibbsQ E[Q]: 18.43 | GibbsQ E[Q]: 20.73
critical: 100%|██████████████████████████████████████████████████████████████| 1/1 [01:07<00:00, 67.80s/rho, rho=0.950]
[2026-03-31 22:29:48,266][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:29:48,382][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:29:48,527][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:29:49,287][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:29:49,391][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 22:29:49,548][__main__][INFO] - Critical load test complete. Curve saved to outputs\debug\critical\run_20260331_222838\critical_load_curve.png, outputs\debug\critical\run_20260331_222838\critical_load_curve.pdf
11/12 critical:  92%|██████████████████████████████████████▌   | 11/12 [07:11<00:57, 57.75s/experiment, alias=critical]
[11/11] Running SSA Component Ablation...
12/12 ablation:  92%|██████████████████████████████████████▌   | 11/12 [07:11<00:57, 57.75s/experiment, alias=critical][handoff] Launching ablation (12/12)
==========================================================
 Starting Experiment: ablation
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=ablation
==========================================================
[2026-03-31 22:30:00,129][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-31 22:30:00,151:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:30:00,151][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-31 22:30:00,151][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-31 22:30:00,163][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\ablation\run_20260331_223000
[2026-03-31 22:30:00,164][__main__][INFO] - ============================================================
[2026-03-31 22:30:00,166][__main__][INFO] -   SSA-Based Ablation Study
[2026-03-31 22:30:00,166][__main__][INFO] - ============================================================
ablation:   0%|                                                                             | 0/4 [00:00<?, ?variant/s][2026-03-31 22:30:00,173][__main__][INFO] - ------------------------------------------------------------
[2026-03-31 22:30:00,173][__main__][INFO] - Training variant: Full Model
[2026-03-31 22:30:00,173][__main__][INFO] -   preprocessing=log1p, init_type=zero_final
                                                                                                                       [2026-03-31 22:30:04,457][experiments.training.train_reinforce][INFO] -   JSQ Mean Queue (Target): 1.0554,  1.50s/stage]
[2026-03-31 22:30:04,458][experiments.training.train_reinforce][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-03-31 22:30:04,459][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-31 22:30:07,980][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
                                                                                                                       [2026-03-31 22:30:08,804][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6875 | Acc: 24.19%
bc_train:   0%|▎                                                                     | 1/201 [00:00<02:44,  1.22step/s][2026-03-31 22:30:09,411][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5707 | Acc: 93.94%
bc_train:  42%|██████████████████▍                         | 84/201 [00:01<00:01, 77.62step/s, loss=0.5724, acc=90.67%][2026-03-31 22:30:10,034][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5645 | Acc: 99.83%
bc_train: 100%|███████████████████████████████████████████| 201/201 [00:02<00:00, 97.84step/s, loss=0.5645, acc=99.83%]
[2026-03-31 22:30:10,035][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-31 22:30:13,591][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
                                                                                                                       [2026-03-31 22:30:14,080][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 1184189.0000
bc_value:   0%|                                                                              | 0/201 [00:00<?, ?step/s][2026-03-31 22:30:14,696][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 934033.5625
bc_value:  42%|█████████████████████▏                            | 85/201 [00:01<00:01, 98.49step/s, loss=1015358.4375][2026-03-31 22:30:15,304][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 827274.7500
bc_value: 100%|█████████████████████████████████████████████████| 201/201 [00:01<00:00, 117.44step/s, loss=827274.7500]
                                                                                                                       [2026-03-31 22:30:15,310][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-31 22:30:15,310][experiments.training.train_reinforce][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-03-31 22:30:15,311][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-31 22:30:15,311][experiments.training.train_reinforce][INFO] -   Epochs: 30, Batch size: 16
[2026-03-31 22:30:15,311][experiments.training.train_reinforce][INFO] -   Simulation time: 1000.0
[2026-03-31 22:30:15,311][experiments.training.train_reinforce][INFO] - ------------------------------------------------------------
                                                                                                                       [2026-03-31 22:31:01,710][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: 0.0366 | mean_logp: -0.5773 | corr: -0.0371
[2026-03-31 22:31:01,711][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.0571 | V-Grad Norm: 135078.9844
                                                                                                                       [2026-03-31 22:38:04,655][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 10 model to policy_net_epoch_010.eqx
                                                                                                                       [2026-03-31 22:38:57,173][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: 0.0556 | mean_logp: -0.5941 | corr: -0.0149
[2026-03-31 22:38:57,174][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.0261 | V-Grad Norm: 87709.7812
                                                                                                                       [2026-03-31 22:46:30,936][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 20 model to policy_net_epoch_020.eqx
                                                                                                                       [2026-03-31 22:47:20,400][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: 0.0586 | mean_logp: -0.5932 | corr: -0.0187
[2026-03-31 22:47:20,401][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.1283 | V-Grad Norm: 6220.5605
reinforce_train:   3%|█                                | 1/30 [24:24<11:47:41, 1464.20s/epoch, queue=4.169, pi=-533.8%]
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:613: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_c.legend(loc="lower right", fontsize=7)
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:632: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_d.legend(loc="upper right", fontsize=7)
[2026-03-31 22:54:42,586][__main__][INFO] - Saved variant artifacts in outputs\debug\ablation\run_20260331_223000\variant_1_full_model
                                                                                                                       [2026-03-31 22:54:42,644][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 22:54:43,547][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:54:43,549][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:08,  1.11rep/s]
[2026-03-31 22:54:44,418][gibbsq.engines.numpy_engine][INFO] -   -> 1,004 arrivals, 1,003 departures, final Q_total = 1
                                                                                                                       [2026-03-31 22:54:44,420][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:07,  1.13rep/s]
[2026-03-31 22:54:45,283][gibbsq.engines.numpy_engine][INFO] -   -> 979 arrivals, 973 departures, final Q_total = 6
                                                                                                                       [2026-03-31 22:54:45,284][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:02<00:06,  1.14rep/s]
[2026-03-31 22:54:46,596][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,051 departures, final Q_total = 3
                                                                                                                       [2026-03-31 22:54:46,597][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:03<00:06,  1.05s/rep]
[2026-03-31 22:54:47,517][gibbsq.engines.numpy_engine][INFO] -   -> 1,036 arrivals, 1,036 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:54:47,519][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:04<00:05,  1.00s/rep]
[2026-03-31 22:54:48,322][gibbsq.engines.numpy_engine][INFO] -   -> 922 arrivals, 922 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:54:48,324][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:05<00:03,  1.07rep/s]
[2026-03-31 22:54:49,201][gibbsq.engines.numpy_engine][INFO] -   -> 991 arrivals, 991 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:54:49,204][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:06<00:02,  1.09rep/s]
[2026-03-31 22:54:50,091][gibbsq.engines.numpy_engine][INFO] -   -> 1,021 arrivals, 1,021 departures, final Q_total = 0
                                                                                                                       [2026-03-31 22:54:50,093][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:07<00:01,  1.10rep/s]
[2026-03-31 22:54:51,016][gibbsq.engines.numpy_engine][INFO] -   -> 1,049 arrivals, 1,048 departures, final Q_total = 1
                                                                                                                       [2026-03-31 22:54:51,017][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:08<00:00,  1.10rep/s]
[2026-03-31 22:54:51,947][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,052 departures, final Q_total = 2
                                                                                                                       [2026-03-31 22:54:51,954][__main__][INFO] -   SSA E[Q_total] = 1.0074 +/- 0.0327
ablation:  25%|███████████▎                                 | 1/4 [24:51<1:14:35, 1491.78s/variant, variant=Full Model][2026-03-31 22:54:51,956][__main__][INFO] - ------------------------------------------------------------
[2026-03-31 22:54:51,956][__main__][INFO] - Training variant: Ablated: No Log-Norm
[2026-03-31 22:54:51,957][__main__][INFO] -   preprocessing=none, init_type=zero_final
                                                                                                                       [2026-03-31 22:54:53,252][experiments.training.train_reinforce][INFO] -   JSQ Mean Queue (Target): 1.0554,  1.28s/stage]
[2026-03-31 22:54:53,255][experiments.training.train_reinforce][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-03-31 22:54:53,257][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-31 22:54:56,835][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
                                                                                                                       [2026-03-31 22:54:57,581][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6800 | Acc: 75.81%
bc_train:   0%|▎                                                                     | 1/201 [00:00<02:28,  1.35step/s][2026-03-31 22:55:00,031][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5613 | Acc: 100.00%
bc_train:  43%|██████████████████▍                        | 86/201 [00:02<00:03, 36.02step/s, loss=0.5613, acc=100.00%][2026-03-31 22:55:02,474][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5612 | Acc: 100.00%
bc_train: 100%|██████████████████████████████████████████| 201/201 [00:05<00:00, 35.66step/s, loss=0.5612, acc=100.00%]
[2026-03-31 22:55:02,476][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-31 22:55:05,949][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
                                                                                                                       [2026-03-31 22:55:06,467][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 1184234.7500
bc_value:   0%|▎                                                                     | 1/201 [00:00<01:43,  1.94step/s][2026-03-31 22:55:08,541][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 588076.2500
bc_value:  49%|█████████████████████████                          | 99/201 [00:02<00:02, 43.96step/s, loss=612513.6875][2026-03-31 22:55:10,608][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 352263.9375
bc_value: 100%|██████████████████████████████████████████████████| 201/201 [00:04<00:00, 43.15step/s, loss=352263.9375]
                                                                                                                       [2026-03-31 22:55:10,615][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-31 22:55:10,615][experiments.training.train_reinforce][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-03-31 22:55:10,616][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-31 22:55:10,616][experiments.training.train_reinforce][INFO] -   Epochs: 30, Batch size: 16
[2026-03-31 22:55:10,616][experiments.training.train_reinforce][INFO] -   Simulation time: 1000.0
[2026-03-31 22:55:10,616][experiments.training.train_reinforce][INFO] - ------------------------------------------------------------
                                                                                                                       [2026-03-31 22:56:01,243][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: 0.0262 | mean_logp: -0.5656 | corr: -0.0171
[2026-03-31 22:56:01,243][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.0108 | V-Grad Norm: 242319.7344
                                                                                                                       [2026-03-31 23:03:56,083][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 10 model to policy_net_epoch_010.eqx
                                                                                                                       [2026-03-31 23:05:01,030][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: 0.0132 | mean_logp: -0.5675 | corr: -0.0169
[2026-03-31 23:05:01,035][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.0296 | V-Grad Norm: 23797.3691
                                                                                                                       [2026-03-31 23:13:09,670][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 20 model to policy_net_epoch_020.eqx
                                                                                                                       [2026-03-31 23:14:01,291][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: 0.0189 | mean_logp: -0.5790 | corr: -0.0283
[2026-03-31 23:14:01,292][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.0276 | V-Grad Norm: 40111.3320
reinforce_train:   3%|█                                | 1/30 [26:39<12:53:16, 1599.88s/epoch, queue=4.143, pi=-528.5%]
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:613: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_c.legend(loc="lower right", fontsize=7)
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:632: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_d.legend(loc="upper right", fontsize=7)
[2026-03-31 23:21:54,341][__main__][INFO] - Saved variant artifacts in outputs\debug\ablation\run_20260331_223000\variant_2_ablated_no_log-norm
                                                                                                                       [2026-03-31 23:21:54,376][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 23:21:55,712][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:21:55,713][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:01<00:12,  1.34s/rep]
[2026-03-31 23:21:56,942][gibbsq.engines.numpy_engine][INFO] -   -> 1,005 arrivals, 1,003 departures, final Q_total = 2
                                                                                                                       [2026-03-31 23:21:56,944][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:02<00:10,  1.27s/rep]
[2026-03-31 23:21:58,586][gibbsq.engines.numpy_engine][INFO] -   -> 979 arrivals, 973 departures, final Q_total = 6
                                                                                                                       [2026-03-31 23:21:58,588][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:04<00:10,  1.44s/rep]
[2026-03-31 23:21:59,711][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,051 departures, final Q_total = 3
                                                                                                                       [2026-03-31 23:21:59,713][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:05<00:07,  1.32s/rep]
[2026-03-31 23:22:00,833][gibbsq.engines.numpy_engine][INFO] -   -> 1,036 arrivals, 1,036 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:22:00,835][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:06<00:06,  1.25s/rep]
[2026-03-31 23:22:01,823][gibbsq.engines.numpy_engine][INFO] -   -> 923 arrivals, 923 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:22:01,825][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:07<00:04,  1.16s/rep]
[2026-03-31 23:22:02,915][gibbsq.engines.numpy_engine][INFO] -   -> 991 arrivals, 991 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:22:02,917][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:08<00:03,  1.14s/rep]
[2026-03-31 23:22:04,390][gibbsq.engines.numpy_engine][INFO] -   -> 1,021 arrivals, 1,021 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:22:04,391][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:10<00:02,  1.24s/rep]
[2026-03-31 23:22:05,526][gibbsq.engines.numpy_engine][INFO] -   -> 1,049 arrivals, 1,048 departures, final Q_total = 1
                                                                                                                       [2026-03-31 23:22:05,528][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:11<00:01,  1.21s/rep]
[2026-03-31 23:22:06,743][gibbsq.engines.numpy_engine][INFO] -   -> 1,055 arrivals, 1,054 departures, final Q_total = 1
                                                                                                                       [2026-03-31 23:22:06,745][__main__][INFO] -   SSA E[Q_total] = 1.0099 +/- 0.0334
ablation:  50%|██████████████████▌                  | 2/4 [52:06<52:31, 1575.91s/variant, variant=Ablated: No Log-Norm][2026-03-31 23:22:06,748][__main__][INFO] - ------------------------------------------------------------
[2026-03-31 23:22:06,749][__main__][INFO] - Training variant: Ablated: No Zero-Init
[2026-03-31 23:22:06,750][__main__][INFO] -   preprocessing=log1p, init_type=standard
                                                                                                                       [2026-03-31 23:22:08,347][experiments.training.train_reinforce][INFO] -   JSQ Mean Queue (Target): 1.0554,  1.58s/stage]
[2026-03-31 23:22:08,347][experiments.training.train_reinforce][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-03-31 23:22:08,349][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-31 23:22:12,796][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
                                                                                                                       [2026-03-31 23:22:13,722][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6800 | Acc: 55.12%
bc_train:   0%|▎                                                                     | 1/201 [00:00<03:04,  1.08step/s][2026-03-31 23:22:16,274][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5693 | Acc: 94.86%
bc_train:  50%|█████████████████████▌                     | 101/201 [00:03<00:02, 34.54step/s, loss=0.5694, acc=94.86%][2026-03-31 23:22:19,614][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5638 | Acc: 99.12%
bc_train: 100%|███████████████████████████████████████████| 201/201 [00:06<00:00, 29.49step/s, loss=0.5638, acc=99.12%]
[2026-03-31 23:22:19,616][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-31 23:22:24,504][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
                                                                                                                       [2026-03-31 23:22:25,159][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 1183506.6250
bc_value:   0%|▎                                                                     | 1/201 [00:00<02:10,  1.53step/s][2026-03-31 23:22:27,482][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 930773.5000
bc_value:  43%|█████████████████████▋                            | 87/201 [00:02<00:03, 37.67step/s, loss=1005088.8750][2026-03-31 23:22:29,509][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 820051.3125
bc_value: 100%|██████████████████████████████████████████████████| 201/201 [00:05<00:00, 40.16step/s, loss=820051.3125]
                                                                                                                       [2026-03-31 23:22:29,520][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-31 23:22:29,521][experiments.training.train_reinforce][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-03-31 23:22:29,521][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-31 23:22:29,522][experiments.training.train_reinforce][INFO] -   Epochs: 30, Batch size: 16
[2026-03-31 23:22:29,522][experiments.training.train_reinforce][INFO] -   Simulation time: 1000.0
[2026-03-31 23:22:29,522][experiments.training.train_reinforce][INFO] - ------------------------------------------------------------
                                                                                                                       [2026-03-31 23:23:45,054][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: 0.0379 | mean_logp: -0.5746 | corr: -0.0337
[2026-03-31 23:23:45,055][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.0469 | V-Grad Norm: 133306.4531
                                                                                                                       [2026-03-31 23:35:12,525][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 10 model to policy_net_epoch_010.eqx
                                                                                                                       [2026-03-31 23:36:30,810][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: 0.0582 | mean_logp: -0.5967 | corr: -0.0088
[2026-03-31 23:36:30,810][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.0196 | V-Grad Norm: 88171.8359
                                                                                                                       [2026-03-31 23:44:55,508][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 20 model to policy_net_epoch_020.eqx
                                                                                                                       [2026-03-31 23:45:43,685][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: 0.0587 | mean_logp: -0.5922 | corr: -0.0158
[2026-03-31 23:45:43,685][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.1901 | V-Grad Norm: 7645.7026
reinforce_train:   3%|█                                | 1/30 [30:35<14:47:07, 1835.45s/epoch, queue=4.160, pi=-532.1%]
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:613: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_c.legend(loc="lower right", fontsize=7)
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:632: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_d.legend(loc="upper right", fontsize=7)
[2026-03-31 23:53:07,523][__main__][INFO] - Saved variant artifacts in outputs\debug\ablation\run_20260331_223000\variant_3_ablated_no_zero-init
                                                                                                                       [2026-03-31 23:53:07,557][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 23:53:08,866][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:53:08,868][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:01<00:11,  1.31s/rep]
[2026-03-31 23:53:10,195][gibbsq.engines.numpy_engine][INFO] -   -> 1,004 arrivals, 1,003 departures, final Q_total = 1
                                                                                                                       [2026-03-31 23:53:10,197][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:02<00:10,  1.32s/rep]
[2026-03-31 23:53:11,290][gibbsq.engines.numpy_engine][INFO] -   -> 979 arrivals, 973 departures, final Q_total = 6
                                                                                                                       [2026-03-31 23:53:11,292][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:03<00:08,  1.22s/rep]
[2026-03-31 23:53:12,323][gibbsq.engines.numpy_engine][INFO] -   -> 1,053 arrivals, 1,051 departures, final Q_total = 2
                                                                                                                       [2026-03-31 23:53:12,325][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:04<00:06,  1.14s/rep]
[2026-03-31 23:53:13,364][gibbsq.engines.numpy_engine][INFO] -   -> 1,036 arrivals, 1,036 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:53:13,366][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:05<00:05,  1.11s/rep]
[2026-03-31 23:53:14,308][gibbsq.engines.numpy_engine][INFO] -   -> 923 arrivals, 923 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:53:14,310][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:06<00:04,  1.05s/rep]
[2026-03-31 23:53:15,516][gibbsq.engines.numpy_engine][INFO] -   -> 991 arrivals, 991 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:53:15,518][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:07<00:03,  1.10s/rep]
[2026-03-31 23:53:16,686][gibbsq.engines.numpy_engine][INFO] -   -> 1,021 arrivals, 1,021 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:53:16,688][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:09<00:02,  1.12s/rep]
[2026-03-31 23:53:17,739][gibbsq.engines.numpy_engine][INFO] -   -> 1,049 arrivals, 1,048 departures, final Q_total = 1
                                                                                                                       [2026-03-31 23:53:17,741][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:10<00:01,  1.10s/rep]
[2026-03-31 23:53:18,782][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,052 departures, final Q_total = 2
                                                                                                                       [2026-03-31 23:53:18,786][__main__][INFO] -   SSA E[Q_total] = 1.0059 +/- 0.0333
ablation:  75%|█████████████████████████▌        | 3/4 [1:23:18<28:31, 1711.13s/variant, variant=Ablated: No Zero-Init][2026-03-31 23:53:18,788][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-03-31 23:53:18,910][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-03-31 23:53:18,910][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-31 23:53:19,032][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-03-31 23:53:19,033][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-31 23:53:19,182][gibbsq.engines.numpy_engine][INFO] -   -> 991 arrivals, 990 departures, final Q_total = 1
[2026-03-31 23:53:19,183][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-31 23:53:19,309][gibbsq.engines.numpy_engine][INFO] -   -> 1,040 arrivals, 1,040 departures, final Q_total = 0
                                                                                                                       [2026-03-31 23:53:19,311][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:00<00:00,  7.68rep/s]
[2026-03-31 23:53:19,450][gibbsq.engines.numpy_engine][INFO] -   -> 1,027 arrivals, 1,026 departures, final Q_total = 1
[2026-03-31 23:53:19,451][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-03-31 23:53:19,603][gibbsq.engines.numpy_engine][INFO] -   -> 911 arrivals, 911 departures, final Q_total = 0
[2026-03-31 23:53:19,604][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-31 23:53:19,767][gibbsq.engines.numpy_engine][INFO] -   -> 993 arrivals, 991 departures, final Q_total = 2
[2026-03-31 23:53:19,767][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-31 23:53:19,949][gibbsq.engines.numpy_engine][INFO] -   -> 1,029 arrivals, 1,027 departures, final Q_total = 2
                                                                                                                       [2026-03-31 23:53:19,951][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:01<00:00,  6.76rep/s]
[2026-03-31 23:53:20,121][gibbsq.engines.numpy_engine][INFO] -   -> 1,020 arrivals, 1,016 departures, final Q_total = 4
[2026-03-31 23:53:20,122][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-31 23:53:20,269][gibbsq.engines.numpy_engine][INFO] -   -> 1,058 arrivals, 1,058 departures, final Q_total = 0
ablation: 100%|██████████████████████████████████| 4/4 [1:23:20<00:00, 1250.03s/variant, variant=Ablated: No Zero-Init]
[2026-03-31 23:53:20,850][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 23:53:21,009][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 23:53:21,213][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 23:53:22,029][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-31 23:53:22,155][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
12/12 ablation: 100%|███████████████████████████████████████| 12/12 [1:30:55<00:00, 454.63s/experiment, alias=ablation]

==========================================================
  Pipeline fully complete.
  Review '/outputs/' for your plots and logs.
==========================================================
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ>


