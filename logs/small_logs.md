PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ> python scripts/execution/reproduction_pipeline.py --config-name debug
==========================================================
  GibbsQ Research Paper: Final Execution Pipeline
==========================================================
  Progress Mode: auto
  Pipeline Started At: 2026-04-02 01:48:23+05:45

[Initiating Pipeline...]

pipeline:   0%|                                                                         | 0/12 [00:00<?, ?experiment/s]
[Pre-Flight] Running Configuration Sanity Checks...
1/12 check_configs:   0%|                                                               | 0/12 [00:00<?, ?experiment/s][handoff] Launching check_configs (1/12)
  -> [check_configs] Started at 2026-04-02 01:48:23+05:45
==========================================================
 Starting Experiment: check_configs
 Start Time: 2026-04-02 01:48:27+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=check_configs
==========================================================
check_configs: roots:   0%|                                                                  | 0/4 [00:00<?, ?config/s][OK] Config debug validated successfully.
[OK] Config small validated successfully.
[OK] Config default validated successfully.
check_configs: roots:  75%|███████████████████████████████████████████▌              | 3/4 [00:00<00:00,  5.15config/s][OK] Config final_experiment validated successfully.
check_configs: resolved paths:   0%|                                                          | 0/48 [00:00<?, ?path/s][OK] Resolved experiment path hyperqual (profile=debug, overrides=['++active_profile=debug', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=debug, overrides=['++active_profile=debug', '++active_experiment=reinforce_check']) validated successfully.
check_configs: resolved paths:   4%|██                                                | 2/48 [00:00<00:14,  3.07path/s][OK] Resolved experiment path drift (profile=debug, overrides=['++active_profile=debug', '++active_experiment=drift']) validated successfully.
[OK] Resolved experiment path sweep (profile=debug, overrides=['++active_profile=debug', '++active_experiment=sweep']) validated successfully.
check_configs: resolved paths:   8%|████▏                                             | 4/48 [00:01<00:14,  3.01path/s][OK] Resolved experiment path stress (profile=debug, overrides=['++active_profile=debug', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Resolved experiment path policy (profile=debug, overrides=['++active_profile=debug', '++active_experiment=policy']) validated successfully.
check_configs: resolved paths:  12%|██████▎                                           | 6/48 [00:02<00:17,  2.46path/s][OK] Resolved experiment path bc_train (profile=debug, overrides=['++active_profile=debug', '++active_experiment=bc_train']) validated successfully.
[OK] Resolved experiment path reinforce_train (profile=debug, overrides=['++active_profile=debug', '++active_experiment=reinforce_train']) validated successfully.
check_configs: resolved paths:  17%|████████▎                                         | 8/48 [00:03<00:16,  2.38path/s][OK] Resolved experiment path stats (profile=debug, overrides=['++active_profile=debug', '++active_experiment=stats']) validated successfully.
[OK] Resolved experiment path generalize (profile=debug, overrides=['++active_profile=debug', '++active_experiment=generalize']) validated successfully.
check_configs: resolved paths:  21%|██████████▏                                      | 10/48 [00:03<00:13,  2.77path/s][OK] Resolved experiment path ablation (profile=debug, overrides=['++active_profile=debug', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=debug, overrides=['++active_profile=debug', '++active_experiment=critical']) validated successfully.
check_configs: resolved paths:  25%|████████████▎                                    | 12/48 [00:04<00:11,  3.08path/s][OK] Resolved experiment path hyperqual (profile=small, overrides=['++active_profile=small', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=small, overrides=['++active_profile=small', '++active_experiment=reinforce_check']) validated successfully.
[OK] Resolved experiment path drift (profile=small, overrides=['++active_profile=small', '++active_experiment=drift']) validated successfully.
check_configs: resolved paths:  31%|███████████████▎                                 | 15/48 [00:04<00:09,  3.41path/s][OK] Resolved experiment path sweep (profile=small, overrides=['++active_profile=small', '++active_experiment=sweep']) validated successfully.
[OK] Resolved experiment path stress (profile=small, overrides=['++active_profile=small', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
check_configs: resolved paths:  35%|█████████████████▎                               | 17/48 [00:05<00:08,  3.56path/s][OK] Resolved experiment path policy (profile=small, overrides=['++active_profile=small', '++active_experiment=policy']) validated successfully.
[OK] Resolved experiment path bc_train (profile=small, overrides=['++active_profile=small', '++active_experiment=bc_train']) validated successfully.
check_configs: resolved paths:  40%|███████████████████▍                             | 19/48 [00:06<00:08,  3.52path/s][OK] Resolved experiment path reinforce_train (profile=small, overrides=['++active_profile=small', '++active_experiment=reinforce_train']) validated successfully.
[OK] Resolved experiment path stats (profile=small, overrides=['++active_profile=small', '++active_experiment=stats']) validated successfully.
[OK] Resolved experiment path generalize (profile=small, overrides=['++active_profile=small', '++active_experiment=generalize']) validated successfully.
check_configs: resolved paths:  46%|██████████████████████▍                          | 22/48 [00:06<00:07,  3.68path/s][OK] Resolved experiment path ablation (profile=small, overrides=['++active_profile=small', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=small, overrides=['++active_profile=small', '++active_experiment=critical']) validated successfully.
check_configs: resolved paths:  50%|████████████████████████▌                        | 24/48 [00:07<00:06,  3.73path/s][OK] Resolved experiment path hyperqual (profile=default, overrides=['++active_profile=default', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=default, overrides=['++active_profile=default', '++active_experiment=reinforce_check']) validated successfully.
check_configs: resolved paths:  54%|██████████████████████████▌                      | 26/48 [00:08<00:07,  2.93path/s][OK] Resolved experiment path drift (profile=default, overrides=['++active_profile=default', '++active_experiment=drift']) validated successfully.
[OK] Resolved experiment path sweep (profile=default, overrides=['++active_profile=default', '++active_experiment=sweep']) validated successfully.
check_configs: resolved paths:  58%|████████████████████████████▌                    | 28/48 [00:09<00:06,  2.90path/s][OK] Resolved experiment path stress (profile=default, overrides=['++active_profile=default', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Resolved experiment path policy (profile=default, overrides=['++active_profile=default', '++active_experiment=policy']) validated successfully.
check_configs: resolved paths:  62%|██████████████████████████████▋                  | 30/48 [00:09<00:06,  2.91path/s][OK] Resolved experiment path bc_train (profile=default, overrides=['++active_profile=default', '++active_experiment=bc_train']) validated successfully.
[OK] Resolved experiment path reinforce_train (profile=default, overrides=['++active_profile=default', '++active_experiment=reinforce_train']) validated successfully.
check_configs: resolved paths:  67%|████████████████████████████████▋                | 32/48 [00:10<00:05,  2.74path/s][OK] Resolved experiment path stats (profile=default, overrides=['++active_profile=default', '++active_experiment=stats']) validated successfully.
[OK] Resolved experiment path generalize (profile=default, overrides=['++active_profile=default', '++active_experiment=generalize']) validated successfully.
check_configs: resolved paths:  71%|██████████████████████████████████▋              | 34/48 [00:11<00:05,  2.73path/s][OK] Resolved experiment path ablation (profile=default, overrides=['++active_profile=default', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=default, overrides=['++active_profile=default', '++active_experiment=critical']) validated successfully.
check_configs: resolved paths:  75%|████████████████████████████████████▊            | 36/48 [00:12<00:04,  2.67path/s][OK] Resolved experiment path hyperqual (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=reinforce_check']) validated successfully.
check_configs: resolved paths:  79%|██████████████████████████████████████▊          | 38/48 [00:12<00:03,  2.61path/s][OK] Resolved experiment path drift (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=drift']) validated successfully.
[OK] Resolved experiment path sweep (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=sweep']) validated successfully.
check_configs: resolved paths:  83%|████████████████████████████████████████▊        | 40/48 [00:13<00:02,  2.80path/s][OK] Resolved experiment path stress (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Resolved experiment path policy (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=policy']) validated successfully.
check_configs: resolved paths:  88%|██████████████████████████████████████████▉      | 42/48 [00:14<00:01,  3.01path/s][OK] Resolved experiment path bc_train (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=bc_train']) validated successfully.
[OK] Resolved experiment path reinforce_train (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=reinforce_train']) validated successfully.
check_configs: resolved paths:  92%|████████████████████████████████████████████▉    | 44/48 [00:14<00:01,  3.19path/s][OK] Resolved experiment path stats (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=stats']) validated successfully.
[OK] Resolved experiment path generalize (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=generalize']) validated successfully.
check_configs: resolved paths:  96%|██████████████████████████████████████████████▉  | 46/48 [00:15<00:00,  3.17path/s][OK] Resolved experiment path ablation (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=critical']) validated successfully.
check_configs: public paths:   0%|                                                            | 0/12 [00:00<?, ?path/s][OK] Public experiment path hyperqual (base=default, overrides=['++active_profile=default', '++active_experiment=hyperqual']) validated successfully.
[OK] Public experiment path reinforce_check (base=default, overrides=['++active_profile=default', '++active_experiment=reinforce_check']) validated successfully.
check_configs: public paths:  17%|████████▋                                           | 2/12 [00:00<00:03,  3.13path/s][OK] Public experiment path drift (base=default, overrides=['++active_profile=default', '++active_experiment=drift']) validated successfully.
[OK] Public experiment path sweep (base=default, overrides=['++active_profile=default', '++active_experiment=sweep']) validated successfully.
check_configs: public paths:  33%|█████████████████▎                                  | 4/12 [00:01<00:02,  2.94path/s][OK] Public experiment path stress (base=default, overrides=['++active_profile=default', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Public experiment path policy (base=default, overrides=['++active_profile=default', '++active_experiment=policy']) validated successfully.
check_configs: public paths:  50%|██████████████████████████                          | 6/12 [00:02<00:02,  2.78path/s][OK] Public experiment path bc_train (base=default, overrides=['++active_profile=default', '++active_experiment=bc_train']) validated successfully.
[OK] Public experiment path reinforce_train (base=default, overrides=['++active_profile=default', '++active_experiment=reinforce_train']) validated successfully.
check_configs: public paths:  67%|██████████████████████████████████▋                 | 8/12 [00:03<00:01,  2.40path/s][OK] Public experiment path stats (base=default, overrides=['++active_profile=default', '++active_experiment=stats']) validated successfully.
[OK] Public experiment path generalize (base=default, overrides=['++active_profile=default', '++active_experiment=generalize']) validated successfully.
check_configs: public paths:  83%|██████████████████████████████████████████▌        | 10/12 [00:03<00:00,  2.43path/s][OK] Public experiment path ablation (base=default, overrides=['++active_profile=default', '++active_experiment=ablation']) validated successfully.
[OK] Public experiment path critical (base=default, overrides=['++active_profile=default', '++active_experiment=critical']) validated successfully.

[SUCCESS] All configs passed validation.

[Experiment 'check_configs' Finished]
  Status: completed
  End Time: 2026-04-02 01:48:53+05:45
  Elapsed Duration: 25.867s
  -> [check_configs] Ended at 2026-04-02 01:48:54+05:45
  -> [check_configs] Status: completed
  -> [check_configs] Elapsed: 30.346s
1/12 check_configs:   8%|██▊                               | 1/12 [00:30<05:33, 30.35s/experiment, alias=check_configs]
[1/11] Running REINFORCE Gradient validation...
2/12 reinforce_check:   8%|██▋                             | 1/12 [00:30<05:33, 30.35s/experiment, alias=check_configs][handoff] Launching reinforce_check (2/12)
  -> [reinforce_check] Started at 2026-04-02 01:48:54+05:45
==========================================================
 Starting Experiment: reinforce_check
 Start Time: 2026-04-02 01:48:58+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=reinforce_check
==========================================================
[2026-04-02 01:49:02,774][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\reinforce_check\run_20260402_014902
[2026-04-02 01:49:02,775][__main__][INFO] - ============================================================
[2026-04-02 01:49:02,775][__main__][INFO] -   REINFORCE Gradient Estimator Validation
[2026-04-02 01:49:02,776][__main__][INFO] - ============================================================
[2026-04-02 01:49:02,776][__main__][INFO] - Validating REINFORCE gradients against the trainer-aligned first-action objective.
INFO:2026-04-02 01:49:02,849:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:49:02,849][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:49:05,554][__main__][INFO] - Statistical Scaling: D=50 parameters.
[2026-04-02 01:49:05,555][__main__][INFO] - Adjusted Z-Critical Threshold: 3.29
[2026-04-02 01:49:05,555][__main__][INFO] - Scaled n_samples from 5000 -> 5000 to maintain confidence bounds.
[2026-04-02 01:49:05,555][__main__][INFO] - Computing REINFORCE gradient estimate...
[2026-04-02 01:49:11,311][__main__][INFO] - Computing finite-difference gradient estimate...
reinforce_check: FD params:   2%|█                                                   | 1/50 [00:00<00:35,  1.39param/s][2026-04-02 01:49:13,286][__main__][INFO] -   [OK] Param  10/50  (idx  2385): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
                                                                                                                       [2026-04-02 01:49:13,523][__main__][INFO] -   [OK] Param  20/50  (idx 16253): RF=  0.001668 | FD=  0.001258 | diff=  0.000410 | z= 0.00 | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
reinforce_check: FD params:  48%|████████████████████████▍                          | 24/50 [00:01<00:01, 23.22param/s][2026-04-02 01:49:13,808][__main__][INFO] -   [OK] Param  30/50  (idx  2928): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
                                                                                                                       [2026-04-02 01:49:14,023][__main__][INFO] -   [OK] Param  40/50  (idx  1282): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
reinforce_check: FD params:  90%|█████████████████████████████████████████████▉     | 45/50 [00:01<00:00, 30.63param/s][2026-04-02 01:49:14,239][__main__][INFO] -   [OK] Param  50/50  (idx  9341): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
reinforce_check: FD params: 100%|███████████████████████████████████████████████████| 50/50 [00:01<00:00, 26.90param/s]
[2026-04-02 01:49:14,247][__main__][INFO] - Relative error: 0.3261
[2026-04-02 01:49:14,247][__main__][INFO] - Cosine similarity: 1.0000
[2026-04-02 01:49:14,248][__main__][INFO] - Bias estimate (L2): 0.001190
[2026-04-02 01:49:14,248][__main__][INFO] - Relative bias: 0.3261
[2026-04-02 01:49:14,248][__main__][INFO] - Variance estimate: 0.000000
[2026-04-02 01:49:14,249][__main__][INFO] - Passed: True
[2026-04-02 01:49:14,250][__main__][INFO] - Results saved to outputs\debug\reinforce_check\run_20260402_014902\gradient_check_result.json
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\utils\chart_exporter.py:95: UserWarning: Glyph 10003 (\N{CHECK MARK}) missing from font(s) Times New Roman.
  fig.savefig(
[2026-04-02 01:49:17,527][__main__][INFO] - Gradient scatter plot saved to outputs\debug\reinforce_check\run_20260402_014902\gradient_scatter.png, outputs\debug\reinforce_check\run_20260402_014902\gradient_scatter.pdf
[2026-04-02 01:49:17,527][__main__][INFO] - GRADIENT CHECK PASSED - REINFORCE estimator is valid

[Experiment 'reinforce_check' Finished]
  Status: completed
  End Time: 2026-04-02 01:49:18+05:45
  Elapsed Duration: 19.723s
  -> [reinforce_check] Ended at 2026-04-02 01:49:18+05:45
  -> [reinforce_check] Status: completed
  -> [reinforce_check] Elapsed: 24.341s
2/12 reinforce_check:  17%|█████                         | 2/12 [00:54<04:28, 26.82s/experiment, alias=reinforce_check]
[2/11] Running Drift Verification (Phase 1a)...
3/12 drift:  17%|██████▋                                 | 2/12 [00:54<04:28, 26.82s/experiment, alias=reinforce_check][handoff] Launching drift (3/12)
  -> [drift] Started at 2026-04-02 01:49:18+05:45
==========================================================
 Starting Experiment: drift
 Start Time: 2026-04-02 01:49:23+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=drift
==========================================================
[2026-04-02 01:49:29,901][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\drift\run_20260402_014929
[2026-04-02 01:49:29,902][__main__][INFO] - System: N=2, lam=1.0, alpha=1.0, cap=2.5000
[2026-04-02 01:49:29,903][__main__][INFO] - Proof bounds: R=2.4431, eps=0.750000
[2026-04-02 01:49:29,903][__main__][INFO] - --- Grid Evaluation (q_max=50) ---
drift: grid:   0%|                                                                            | 0/3 [00:00<?, ?stage/s][2026-04-02 01:49:29,910][__main__][INFO] - States evaluated: 2,601
[2026-04-02 01:49:29,911][__main__][INFO] - Bound violations: 0
[2026-04-02 01:49:30,732][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:49:31,189][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:49:31,816][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:49:31,998][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:49:32,153][__main__][INFO] - Saved: outputs\debug\drift\run_20260402_014929\drift_heatmap.png, outputs\debug\drift\run_20260402_014929\drift_heatmap.pdf
[2026-04-02 01:49:32,386][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:49:32,566][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:49:33,917][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:49:34,003][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:49:34,206][__main__][INFO] - Saved: outputs\debug\drift\run_20260402_014929\drift_vs_norm.png, outputs\debug\drift\run_20260402_014929\drift_vs_norm.pdf
drift: grid: 100%|████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.43s/stage, N=2, mode=grid]
[2026-04-02 01:49:34,211][__main__][INFO] - Drift verification complete.

[Experiment 'drift' Finished]
  Status: completed
  End Time: 2026-04-02 01:49:34+05:45
  Elapsed Duration: 10.666s
  -> [drift] Ended at 2026-04-02 01:49:35+05:45
  -> [drift] Status: completed
  -> [drift] Elapsed: 16.752s
3/12 drift:  25%|████████████▌                                     | 3/12 [01:11<03:20, 22.28s/experiment, alias=drift]
[3/11] Running Stability Sweep (Phase 1b)...
4/12 sweep:  25%|████████████▌                                     | 3/12 [01:12<03:20, 22.28s/experiment, alias=drift][handoff] Launching sweep (4/12)
  -> [sweep] Started at 2026-04-02 01:49:35+05:45
==========================================================
 Starting Experiment: sweep
 Start Time: 2026-04-02 01:49:41+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=sweep
==========================================================
[2026-04-02 01:49:46,674][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-04-02 01:49:46,694:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:49:46,694][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:49:46,695][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-04-02 01:49:46,707][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\sweep\run_20260402_014946
[2026-04-02 01:49:46,708][gibbsq.utils.logging][INFO] - [Logging] WandB offline mode.
wandb: Tracking run with wandb version 0.23.1
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in outputs\debug\sweep\run_20260402_014946\wandb\offline-run-20260402_014947-g1v5ykv4
[2026-04-02 01:49:50,172][gibbsq.utils.logging][INFO] - [Logging] WandB Run Linked: run_20260402_014946 (offline)
[2026-04-02 01:49:50,176][__main__][INFO] - System: N=2, cap=2.5000 | Backend: JAX
[2026-04-02 01:49:50,177][__main__][INFO] - Grid: 9 alpha x 5 rho x 10 reps
sweep:   0%|                                                                                  | 0/45 [00:00<?, ?cell/s][2026-04-02 01:49:50,182][__main__][INFO] -
------------------------------------------------------------
  rho = 0.80  (lam = 2.0000)
------------------------------------------------------------
                                                                                                                       [2026-04-02 01:49:51,730][__main__][INFO] -   alpha=  0.01 | E[Q_total]=    7.35 | NONSTATIONARY  (1/45)
sweep:   2%|▉                                        | 1/45 [00:01<01:08,  1.55s/cell, rho=0.80, alpha=0.01, done=1/45][2026-04-02 01:49:51,915][__main__][INFO] -   alpha=  0.10 | E[Q_total]=    6.58 | NONSTATIONARY  (2/45)
                                                                                                                       [2026-04-02 01:49:52,126][__main__][INFO] -   alpha=  0.50 | E[Q_total]=    5.31 | NONSTATIONARY  (3/45)
                                                                                                                       [2026-04-02 01:49:52,302][__main__][INFO] -   alpha=  1.00 | E[Q_total]=    5.10 | NONSTATIONARY  (4/45)
sweep:   9%|███▋                                     | 4/45 [00:02<00:18,  2.23cell/s, rho=0.80, alpha=1.00, done=4/45][2026-04-02 01:49:52,476][__main__][INFO] -   alpha=  2.00 | E[Q_total]=    4.91 | OK  (5/45)
                                                                                                                       [2026-04-02 01:49:52,675][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    4.63 | OK  (6/45)
                                                                                                                       [2026-04-02 01:49:52,850][__main__][INFO] -   alpha= 10.00 | E[Q_total]=    4.75 | NONSTATIONARY  (7/45)
sweep:  16%|██████▏                                 | 7/45 [00:02<00:11,  3.27cell/s, rho=0.80, alpha=10.00, done=7/45][2026-04-02 01:49:53,033][__main__][INFO] -   alpha= 50.00 | E[Q_total]=    5.18 | NONSTATIONARY  (8/45)
                                                                                                                       [2026-04-02 01:49:53,216][__main__][INFO] -   alpha=100.00 | E[Q_total]=    5.04 | OK  (9/45)
[2026-04-02 01:49:53,216][__main__][INFO] -
------------------------------------------------------------
  rho = 0.85  (lam = 2.1250)
------------------------------------------------------------
                                                                                                                       [2026-04-02 01:49:54,729][__main__][INFO] -   alpha=  0.01 | E[Q_total]=    9.52 | NONSTATIONARY  (10/45)
sweep:  22%|████████▋                              | 10/45 [00:04<00:15,  2.25cell/s, rho=0.85, alpha=0.01, done=10/45][2026-04-02 01:49:54,948][__main__][INFO] -   alpha=  0.10 | E[Q_total]=    8.30 | OK  (11/45)
                                                                                                                       [2026-04-02 01:49:55,206][__main__][INFO] -   alpha=  0.50 | E[Q_total]=    7.09 | NONSTATIONARY  (12/45)
                                                                                                                       [2026-04-02 01:49:55,447][__main__][INFO] -   alpha=  1.00 | E[Q_total]=    7.25 | OK  (13/45)
sweep:  29%|███████████▎                           | 13/45 [00:05<00:11,  2.73cell/s, rho=0.85, alpha=1.00, done=13/45][2026-04-02 01:49:55,636][__main__][INFO] -   alpha=  2.00 | E[Q_total]=    6.88 | NONSTATIONARY  (14/45)
                                                                                                                       [2026-04-02 01:49:55,824][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    6.01 | NONSTATIONARY  (15/45)
                                                                                                                       [2026-04-02 01:49:56,021][__main__][INFO] -   alpha= 10.00 | E[Q_total]=    6.42 | OK  (16/45)
sweep:  36%|█████████████▌                        | 16/45 [00:05<00:08,  3.29cell/s, rho=0.85, alpha=10.00, done=16/45][2026-04-02 01:49:56,223][__main__][INFO] -   alpha= 50.00 | E[Q_total]=    8.13 | OK  (17/45)
                                                                                                                       [2026-04-02 01:49:56,426][__main__][INFO] -   alpha=100.00 | E[Q_total]=    6.68 | OK  (18/45)
[2026-04-02 01:49:56,426][__main__][INFO] -
------------------------------------------------------------
  rho = 0.90  (lam = 2.2500)
------------------------------------------------------------
                                                                                                                       [2026-04-02 01:49:57,662][__main__][INFO] -   alpha=  0.01 | E[Q_total]=   17.44 | NONSTATIONARY  (19/45)
sweep:  42%|████████████████▍                      | 19/45 [00:07<00:10,  2.59cell/s, rho=0.90, alpha=0.01, done=19/45][2026-04-02 01:49:57,890][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   13.90 | OK  (20/45)
                                                                                                                       [2026-04-02 01:49:58,074][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   11.72 | OK  (21/45)
                                                                                                                       [2026-04-02 01:49:58,299][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   10.16 | OK  (22/45)
sweep:  49%|███████████████████                    | 22/45 [00:08<00:07,  3.03cell/s, rho=0.90, alpha=1.00, done=22/45][2026-04-02 01:49:58,532][__main__][INFO] -   alpha=  2.00 | E[Q_total]=    9.90 | OK  (23/45)
                                                                                                                       [2026-04-02 01:49:58,779][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    9.41 | OK  (24/45)
                                                                                                                       [2026-04-02 01:49:59,071][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   10.07 | OK  (25/45)
sweep:  56%|█████████████████████                 | 25/45 [00:08<00:06,  3.26cell/s, rho=0.90, alpha=10.00, done=25/45][2026-04-02 01:49:59,295][__main__][INFO] -   alpha= 50.00 | E[Q_total]=    9.78 | NONSTATIONARY  (26/45)
                                                                                                                       [2026-04-02 01:49:59,535][__main__][INFO] -   alpha=100.00 | E[Q_total]=   10.50 | OK  (27/45)
[2026-04-02 01:49:59,537][__main__][INFO] -
------------------------------------------------------------
  rho = 0.95  (lam = 2.3750)
------------------------------------------------------------
                                                                                                                       [2026-04-02 01:50:01,152][__main__][INFO] -   alpha=  0.01 | E[Q_total]=   26.52 | NONSTATIONARY  (28/45)
sweep:  62%|████████████████████████▎              | 28/45 [00:10<00:07,  2.34cell/s, rho=0.95, alpha=0.01, done=28/45][2026-04-02 01:50:01,367][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   20.37 | NONSTATIONARY  (29/45)
                                                                                                                       [2026-04-02 01:50:01,577][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   19.48 | NONSTATIONARY  (30/45)
                                                                                                                       [2026-04-02 01:50:01,814][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   19.63 | NONSTATIONARY  (31/45)
sweep:  69%|██████████████████████████▊            | 31/45 [00:11<00:05,  2.75cell/s, rho=0.95, alpha=1.00, done=31/45][2026-04-02 01:50:02,028][__main__][INFO] -   alpha=  2.00 | E[Q_total]=   15.14 | NONSTATIONARY  (32/45)
                                                                                                                       [2026-04-02 01:50:02,240][__main__][INFO] -   alpha=  5.00 | E[Q_total]=   17.48 | NONSTATIONARY  (33/45)
                                                                                                                       [2026-04-02 01:50:02,459][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   13.56 | NONSTATIONARY  (34/45)
sweep:  76%|████████████████████████████▋         | 34/45 [00:12<00:03,  3.14cell/s, rho=0.95, alpha=10.00, done=34/45][2026-04-02 01:50:02,663][__main__][INFO] -   alpha= 50.00 | E[Q_total]=   14.22 | NONSTATIONARY  (35/45)
                                                                                                                       [2026-04-02 01:50:02,896][__main__][INFO] -   alpha=100.00 | E[Q_total]=   16.60 | NONSTATIONARY  (36/45)
[2026-04-02 01:50:02,896][__main__][INFO] -
------------------------------------------------------------
  rho = 0.98  (lam = 2.4500)
------------------------------------------------------------
                                                                                                                       [2026-04-02 01:50:04,206][__main__][INFO] -   alpha=  0.01 | E[Q_total]=   54.15 | NONSTATIONARY  (37/45)
sweep:  82%|████████████████████████████████       | 37/45 [00:14<00:03,  2.51cell/s, rho=0.98, alpha=0.01, done=37/45][2026-04-02 01:50:04,439][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   23.58 | NONSTATIONARY  (38/45)
                                                                                                                       [2026-04-02 01:50:04,701][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   21.20 | NONSTATIONARY  (39/45)
                                                                                                                       [2026-04-02 01:50:04,949][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   34.63 | NONSTATIONARY  (40/45)
sweep:  89%|██████████████████████████████████▋    | 40/45 [00:14<00:01,  2.83cell/s, rho=0.98, alpha=1.00, done=40/45][2026-04-02 01:50:05,219][__main__][INFO] -   alpha=  2.00 | E[Q_total]=   25.32 | NONSTATIONARY  (41/45)
                                                                                                                       [2026-04-02 01:50:05,472][__main__][INFO] -   alpha=  5.00 | E[Q_total]=   30.64 | NONSTATIONARY  (42/45)
sweep:  93%|████████████████████████████████████▍  | 42/45 [00:15<00:00,  3.01cell/s, rho=0.98, alpha=5.00, done=42/45][2026-04-02 01:50:05,738][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   25.00 | NONSTATIONARY  (43/45)
                                                                                                                       [2026-04-02 01:50:05,949][__main__][INFO] -   alpha= 50.00 | E[Q_total]=   38.74 | NONSTATIONARY  (44/45)
                                                                                                                       [2026-04-02 01:50:06,147][__main__][INFO] -   alpha=100.00 | E[Q_total]=   25.68 | NONSTATIONARY  (45/45)
sweep: 100%|█████████████████████████████████████| 45/45 [00:15<00:00,  2.82cell/s, rho=0.98, alpha=100.00, done=45/45]
[2026-04-02 01:50:06,534][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:06,890][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:07,596][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:07,732][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:07,957][__main__][INFO] -
Saved plot: outputs\debug\sweep\run_20260402_014946\alpha_sweep.png, outputs\debug\sweep\run_20260402_014946\alpha_sweep.pdf
[2026-04-02 01:50:08,074][__main__][INFO] -
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
wandb: wandb sync outputs\debug\sweep\run_20260402_014946\wandb\offline-run-20260402_014947-g1v5ykv4
wandb: Find logs at: outputs\debug\sweep\run_20260402_014946\wandb\offline-run-20260402_014947-g1v5ykv4\logs

[Experiment 'sweep' Finished]
  Status: completed
  End Time: 2026-04-02 01:50:08+05:45
  Elapsed Duration: 26.723s
  -> [sweep] Ended at 2026-04-02 01:50:08+05:45
  -> [sweep] Status: completed
  -> [sweep] Elapsed: 32.974s
4/12 sweep:  33%|████████████████▋                                 | 4/12 [01:45<03:33, 26.75s/experiment, alias=sweep]
[4/11] Running Scaling Stress Tests (Phase 1c)...
5/12 stress:  33%|████████████████▎                                | 4/12 [01:45<03:33, 26.75s/experiment, alias=sweep][handoff] Launching stress (5/12)
  -> [stress] Started at 2026-04-02 01:50:08+05:45
==========================================================
 Starting Experiment: stress
 Start Time: 2026-04-02 01:50:12+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=stress ++jax.enabled=True
==========================================================
[2026-04-02 01:50:16,653][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-04-02 01:50:16,674:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:50:16,674][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:50:16,674][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-04-02 01:50:16,687][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\stress\run_20260402_015016
[2026-04-02 01:50:16,687][__main__][INFO] - ============================================================
[2026-04-02 01:50:16,692][__main__][INFO] -   GibbsQ Stress Test (JAX Accelerator Active)
[2026-04-02 01:50:16,692][__main__][INFO] - ============================================================
stress:   0%|                                                                                 | 0/3 [00:00<?, ?stage/s][2026-04-02 01:50:16,696][__main__][INFO] -
[TEST 1] Massive-N Scaling Analysis
                                                                                                                       [2026-04-02 01:50:16,782][__main__][INFO] -   Simulating N=4 experts (rho=0.8)...                  | 0/2 [00:00<?, ?N/s]
                                                                                                                       [2026-04-02 01:50:18,367][__main__][INFO] -     -> Average Gini Imbalance: 0.0244
                                                                                                                       [2026-04-02 01:50:18,435][__main__][INFO] -   Simulating N=8 experts (rho=0.8)...     | 1/2 [00:01<00:01,  1.67s/N, N=4]
                                                                                                                       [2026-04-02 01:50:20,148][__main__][INFO] -     -> Average Gini Imbalance: 0.0288
stress:  33%|████████████████████████▎                                                | 1/3 [00:03<00:06,  3.45s/stage][2026-04-02 01:50:20,151][__main__][INFO] -
[TEST 2] Critical Load Analysis (rho up to 0.9)
                                                                                                                       [2026-04-02 01:50:20,188][__main__][INFO] -   Simulating rho=0.900 (T=2000.0)...                 | 0/1 [00:00<?, ?rho/s]
[2026-04-02 01:50:24,107][__main__][INFO] -     -> Gelman-Rubin R-hat across replicas (post MSER-5 burn-in): 1.0057
                                                                                                                       [2026-04-02 01:50:24,151][__main__][INFO] -     -> Avg E[Q_total]: 21.11 | Stationarity: 9/10
stress:  67%|████████████████████████████████████████████████▋                        | 2/3 [00:07<00:03,  3.78s/stage][2026-04-02 01:50:24,154][__main__][INFO] -
[TEST 3] Extreme Heterogeneity Resilience (100x Speed Gap)
[2026-04-02 01:50:24,170][__main__][INFO] -   Simulating heterogenous setup: mu=[10.   0.1  0.1  0.1]
                                                                                                                       [2026-04-02 01:50:25,822][__main__][INFO] -     -> Mean Queue per Expert: [1.07353308 0.         0.         0.        ]
[2026-04-02 01:50:25,822][__main__][INFO] -     -> Gini: 0.7500
stress: 100%|█████████████████████████████████████████████████████████████████████████| 3/3 [00:09<00:00,  3.04s/stage]
[2026-04-02 01:50:26,728][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:26,849][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:27,029][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:27,081][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:27,268][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:27,338][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:28,577][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:28,622][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:28,749][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:28,814][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:50:29,013][__main__][INFO] - Stress dashboard saved to outputs\debug\stress\run_20260402_015016\stress_dashboard.png, outputs\debug\stress\run_20260402_015016\stress_dashboard.pdf
[2026-04-02 01:50:29,014][__main__][INFO] -
Stress test complete.

[Experiment 'stress' Finished]
  Status: completed
  End Time: 2026-04-02 01:50:29+05:45
  Elapsed Duration: 17.003s
  -> [stress] Ended at 2026-04-02 01:50:29+05:45
  -> [stress] Status: completed
  -> [stress] Elapsed: 20.901s
5/12 stress:  42%|████████████████████                            | 5/12 [02:06<02:52, 24.64s/experiment, alias=stress]
[5/11] Running Platinum BC Pretraining (Phase 2a)...
6/12 bc_train:  42%|███████████████████▏                          | 5/12 [02:06<02:52, 24.64s/experiment, alias=stress][handoff] Launching bc_train (6/12)
  -> [bc_train] Started at 2026-04-02 01:50:29+05:45
==========================================================
 Starting Experiment: bc_train
 Start Time: 2026-04-02 01:50:33+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=bc_train
==========================================================
[2026-04-02 01:50:37,976][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\bc_train\run_20260402_015037
INFO:2026-04-02 01:50:37,997:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:50:37,997][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:50:39,307][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-04-02 01:50:40,714][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
bc_train:   0%|▎                                                                     | 1/201 [00:00<02:11,  1.52step/s][2026-04-02 01:50:41,377][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6862 | Acc: 22.50%
[2026-04-02 01:50:41,620][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5651 | Acc: 93.94%
[2026-04-02 01:50:41,865][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5583 | Acc: 99.39%
bc_train: 100%|██████████████████████████████████████████| 201/201 [00:01<00:00, 175.41step/s, loss=0.5583, acc=99.39%]
[2026-04-02 01:50:41,872][__main__][INFO] -
[DONE] Platinum BC Weights saved to outputs\debug\bc_train\run_20260402_015037\n_gibbsq_platinum_bc_weights.eqx
[2026-04-02 01:50:41,872][__main__][INFO] - [Metadata] BC warm-start compatibility metadata saved to outputs\debug\bc_train\run_20260402_015037\n_gibbsq_platinum_bc_weights.eqx.bc_metadata.json
[2026-04-02 01:50:41,874][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_bc_weights.txt at outputs\debug\latest_bc_weights.txt

[Experiment 'bc_train' Finished]
  Status: completed
  End Time: 2026-04-02 01:50:42+05:45
  Elapsed Duration: 8.742s
  -> [bc_train] Ended at 2026-04-02 01:50:42+05:45
  -> [bc_train] Status: completed
  -> [bc_train] Elapsed: 12.725s
6/12 bc_train:  50%|██████████████████████                      | 6/12 [02:18<02:03, 20.59s/experiment, alias=bc_train]
[6/11] Running REINFORCE SSA Training (Phase 2b)...
7/12 reinforce_train:  50%|██████████████████▌                  | 6/12 [02:18<02:03, 20.59s/experiment, alias=bc_train][handoff] Launching reinforce_train (7/12)
  -> [reinforce_train] Started at 2026-04-02 01:50:42+05:45
==========================================================
 Starting Experiment: reinforce_train
 Start Time: 2026-04-02 01:50:46+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=reinforce_train
==========================================================
[2026-04-02 01:50:50,700][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\reinforce_train\run_20260402_015050
INFO:2026-04-02 01:50:50,728:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:50:50,728][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
reinforce: setup:   0%|                                                                       | 0/3 [00:00<?, ?stage/s][2026-04-02 01:50:53,372][__main__][INFO] -   JSQ Mean Queue (Target): 1.0803
[2026-04-02 01:50:53,373][__main__][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-04-02 01:50:53,390][__main__][INFO] - Reusing BC warm-start actor weights from C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\bc_train\run_20260402_015037\n_gibbsq_platinum_bc_weights.eqx
[2026-04-02 01:50:53,395][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-04-02 01:50:54,757][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
                                                                                                                       [2026-04-02 01:50:55,210][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 1046275.3125 0/201 [00:00<?, ?step/s]
                                                                                                                       [2026-04-02 01:50:55,452][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 770842.1250step/s, loss=1038977.1875]
[2026-04-02 01:50:55,690][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 626511.8750
bc_value: 100%|█████████████████████████████████████████████████| 201/201 [00:00<00:00, 215.66step/s, loss=626511.8750]
reinforce: setup:  67%|██████████████████████████████████████████                     | 2/3 [00:02<00:01,  1.31s/stage][2026-04-02 01:50:55,696][__main__][INFO] - ============================================================
[2026-04-02 01:50:55,696][__main__][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-04-02 01:50:55,696][__main__][INFO] - ============================================================
[2026-04-02 01:50:55,696][__main__][INFO] -   Epochs: 5, Batch size: 4
[2026-04-02 01:50:55,697][__main__][INFO] -   Simulation time: 1000.0
[2026-04-02 01:50:55,697][__main__][INFO] - ------------------------------------------------------------
reinforce_train:   0%|                                                                        | 0/5 [00:00<?, ?epoch/s][2026-04-02 01:51:11,128][__main__][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: 0.0343 | mean_logp: -0.5753 | corr: -0.0126
[2026-04-02 01:51:11,128][__main__][INFO] -     [Grad Check] P-Grad Norm: 0.0527 | V-Grad Norm: 10186.9658
reinforce_train:  20%|███████▊                               | 1/5 [01:07<04:29, 67.32s/epoch, queue=2.522, pi=-204.3%]
[2026-04-02 01:52:07,026][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\debug\latest_reinforce_weights.txt
[2026-04-02 01:52:07,027][__main__][INFO] - -------------------------------------------------------
[2026-04-02 01:52:07,027][__main__][INFO] - -------------------------------------------------------
[2026-04-02 01:52:07,027][__main__][INFO] - Running Final Deterministic Evaluation (N=3)...
[2026-04-02 01:52:08,728][__main__][INFO] - Stage profile written to outputs\debug\reinforce_train\run_20260402_015050\reinforce_stage_profile.json
[2026-04-02 01:52:08,728][__main__][INFO] - Deterministic Policy Score: 93.87% ± 17.32%
[2026-04-02 01:52:08,728][__main__][INFO] - JSQ Target: 100.0% | Random Floor: 0.0% (Performance Index Scale)
[2026-04-02 01:52:08,729][__main__][INFO] - -------------------------------------------------------
[2026-04-02 01:52:08,729][__main__][INFO] - Training Complete! Final Loss: 0.0066
[2026-04-02 01:52:08,729][__main__][INFO] - Final Base-Regime Index Proxy: -204.32
[2026-04-02 01:52:08,730][__main__][INFO] - Policy weights: outputs\debug\reinforce_train\run_20260402_015050\n_gibbsq_reinforce_weights.eqx
[2026-04-02 01:52:08,730][__main__][INFO] - Value weights: outputs\debug\reinforce_train\run_20260402_015050\value_network_weights.eqx

[Experiment 'reinforce_train' Finished]
  Status: completed
  End Time: 2026-04-02 01:52:11+05:45
  Elapsed Duration: 84.905s
  -> [reinforce_train] Ended at 2026-04-02 01:52:11+05:45
  -> [reinforce_train] Status: completed
  -> [reinforce_train] Elapsed: 88.824s
7/12 reinforce_train:  58%|█████████████████▌            | 7/12 [03:47<03:34, 42.90s/experiment, alias=reinforce_train]
[7/11] Running Corrected Policy Evaluation Benchmark (Phase 3a)...
8/12 policy:  58%|██████████████████████▊                | 7/12 [03:47<03:34, 42.90s/experiment, alias=reinforce_train][handoff] Launching policy (8/12)
  -> [policy] Started at 2026-04-02 01:52:11+05:45
==========================================================
 Starting Experiment: policy
 Start Time: 2026-04-02 01:52:14+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=policy
==========================================================
[2026-04-02 01:52:19,650][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\policy\run_20260402_015219
[2026-04-02 01:52:19,651][gibbsq.utils.logging][INFO] - [Logging] WandB offline mode.
wandb: Tracking run with wandb version 0.23.1
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in outputs\debug\policy\run_20260402_015219\wandb\offline-run-20260402_015219-eaqszz8v
[2026-04-02 01:52:22,712][gibbsq.utils.logging][INFO] - [Logging] WandB Run Linked: run_20260402_015219 (offline)
[2026-04-02 01:52:22,713][__main__][INFO] - ============================================================
[2026-04-02 01:52:22,714][__main__][INFO] -   Corrected Policy Comparison
[2026-04-02 01:52:22,715][__main__][INFO] - ============================================================
[2026-04-02 01:52:22,717][__main__][INFO] - System: N=2, lambda=1.0000, Lambda=2.5000, rho=0.4000
[2026-04-02 01:52:22,718][__main__][INFO] - ------------------------------------------------------------
policy: tiers:   0%|                                                                         | 0/7 [00:00<?, ?policy/s][2026-04-02 01:52:22,722][__main__][INFO] - Evaluating Tier 2: JSQ (Min Queue)...
                                                                                                                       [2026-04-02 01:52:22,725][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 01:52:22,839][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-04-02 01:52:22,840][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-04-02 01:52:22,956][gibbsq.engines.numpy_engine][INFO] -   -> 993 arrivals, 993 departures, final Q_total = 0
[2026-04-02 01:52:22,956][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-04-02 01:52:23,072][gibbsq.engines.numpy_engine][INFO] -   -> 1,000 arrivals, 1,000 departures, final Q_total = 0
[2026-04-02 01:52:23,074][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-04-02 01:52:23,204][gibbsq.engines.numpy_engine][INFO] -   -> 1,051 arrivals, 1,050 departures, final Q_total = 1
[2026-04-02 01:52:23,205][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-04-02 01:52:23,331][gibbsq.engines.numpy_engine][INFO] -   -> 1,039 arrivals, 1,039 departures, final Q_total = 0
                                                                                                                       [2026-04-02 01:52:23,334][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:00<00:00,  8.23rep/s]
[2026-04-02 01:52:23,440][gibbsq.engines.numpy_engine][INFO] -   -> 915 arrivals, 915 departures, final Q_total = 0
[2026-04-02 01:52:23,442][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 01:52:23,562][gibbsq.engines.numpy_engine][INFO] -   -> 986 arrivals, 986 departures, final Q_total = 0
[2026-04-02 01:52:23,563][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-04-02 01:52:23,690][gibbsq.engines.numpy_engine][INFO] -   -> 1,026 arrivals, 1,025 departures, final Q_total = 1
[2026-04-02 01:52:23,691][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-04-02 01:52:23,808][gibbsq.engines.numpy_engine][INFO] -   -> 1,033 arrivals, 1,030 departures, final Q_total = 3
[2026-04-02 01:52:23,809][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-04-02 01:52:23,936][gibbsq.engines.numpy_engine][INFO] -   -> 1,058 arrivals, 1,058 departures, final Q_total = 0
                                                                                                                       [2026-04-02 01:52:23,941][__main__][INFO] -   E[Q_total] = 1.0603 ± 0.0313
policy: tiers:  14%|█████████▎                                                       | 1/7 [00:01<00:07,  1.22s/policy][2026-04-02 01:52:23,945][__main__][INFO] - Evaluating Tier 2: JSSQ (Min Sojourn)...
                                                                                                                       [2026-04-02 01:52:23,949][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 01:52:24,084][gibbsq.engines.numpy_engine][INFO] -   -> 1,012 arrivals, 1,011 departures, final Q_total = 1
[2026-04-02 01:52:24,086][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-04-02 01:52:24,205][gibbsq.engines.numpy_engine][INFO] -   -> 1,008 arrivals, 1,007 departures, final Q_total = 1
[2026-04-02 01:52:24,207][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-04-02 01:52:24,335][gibbsq.engines.numpy_engine][INFO] -   -> 981 arrivals, 975 departures, final Q_total = 6
[2026-04-02 01:52:24,337][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-04-02 01:52:24,464][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,050 departures, final Q_total = 0
                                                                                                                       [2026-04-02 01:52:24,467][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:00<00:00,  7.75rep/s]
[2026-04-02 01:52:24,599][gibbsq.engines.numpy_engine][INFO] -   -> 1,036 arrivals, 1,036 departures, final Q_total = 0
[2026-04-02 01:52:24,600][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-04-02 01:52:24,723][gibbsq.engines.numpy_engine][INFO] -   -> 922 arrivals, 922 departures, final Q_total = 0
[2026-04-02 01:52:24,725][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 01:52:24,854][gibbsq.engines.numpy_engine][INFO] -   -> 989 arrivals, 989 departures, final Q_total = 0
[2026-04-02 01:52:24,855][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-04-02 01:52:24,977][gibbsq.engines.numpy_engine][INFO] -   -> 1,023 arrivals, 1,022 departures, final Q_total = 1
                                                                                                                       [2026-04-02 01:52:24,980][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:01<00:00,  7.78rep/s]
[2026-04-02 01:52:25,116][gibbsq.engines.numpy_engine][INFO] -   -> 1,046 arrivals, 1,046 departures, final Q_total = 0
[2026-04-02 01:52:25,117][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-04-02 01:52:25,239][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,050 departures, final Q_total = 4
                                                                                                                       [2026-04-02 01:52:25,246][__main__][INFO] -   E[Q_total] = 0.9949 ± 0.0294
policy: tiers:  29%|██████████████████▌                                              | 2/7 [00:02<00:06,  1.27s/policy][2026-04-02 01:52:25,249][__main__][INFO] - Evaluating Tier 3: UAS (alpha=1.0)...
                                                                                                                       [2026-04-02 01:52:25,252][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 01:52:25,376][gibbsq.engines.numpy_engine][INFO] -   -> 1,004 arrivals, 1,001 departures, final Q_total = 3
[2026-04-02 01:52:25,378][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-04-02 01:52:25,507][gibbsq.engines.numpy_engine][INFO] -   -> 997 arrivals, 996 departures, final Q_total = 1
[2026-04-02 01:52:25,508][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-04-02 01:52:25,633][gibbsq.engines.numpy_engine][INFO] -   -> 983 arrivals, 983 departures, final Q_total = 0
[2026-04-02 01:52:25,634][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-04-02 01:52:25,782][gibbsq.engines.numpy_engine][INFO] -   -> 1,044 arrivals, 1,044 departures, final Q_total = 0
                                                                                                                       [2026-04-02 01:52:25,786][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:00<00:00,  7.53rep/s]
[2026-04-02 01:52:25,922][gibbsq.engines.numpy_engine][INFO] -   -> 1,031 arrivals, 1,029 departures, final Q_total = 2
[2026-04-02 01:52:25,924][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-04-02 01:52:26,046][gibbsq.engines.numpy_engine][INFO] -   -> 909 arrivals, 909 departures, final Q_total = 0
[2026-04-02 01:52:26,048][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 01:52:26,187][gibbsq.engines.numpy_engine][INFO] -   -> 991 arrivals, 991 departures, final Q_total = 0
[2026-04-02 01:52:26,188][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-04-02 01:52:26,323][gibbsq.engines.numpy_engine][INFO] -   -> 1,015 arrivals, 1,015 departures, final Q_total = 0
                                                                                                                       [2026-04-02 01:52:26,327][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:01<00:00,  7.45rep/s]
[2026-04-02 01:52:26,457][gibbsq.engines.numpy_engine][INFO] -   -> 1,041 arrivals, 1,040 departures, final Q_total = 1
[2026-04-02 01:52:26,459][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-04-02 01:52:26,601][gibbsq.engines.numpy_engine][INFO] -   -> 1,058 arrivals, 1,057 departures, final Q_total = 1
                                                                                                                       [2026-04-02 01:52:26,606][__main__][INFO] -   E[Q_total] = 1.1298 ± 0.0372
policy: tiers:  43%|███████████████████████████▊                                     | 3/7 [00:03<00:05,  1.31s/policy][2026-04-02 01:52:26,612][__main__][INFO] - Evaluating Tier 3: UAS (alpha=10.0)...
                                                                                                                       [2026-04-02 01:52:26,615][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 01:52:26,747][gibbsq.engines.numpy_engine][INFO] -   -> 1,012 arrivals, 1,011 departures, final Q_total = 1
[2026-04-02 01:52:26,749][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-04-02 01:52:26,890][gibbsq.engines.numpy_engine][INFO] -   -> 1,007 arrivals, 1,007 departures, final Q_total = 0
[2026-04-02 01:52:26,892][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-04-02 01:52:27,022][gibbsq.engines.numpy_engine][INFO] -   -> 983 arrivals, 983 departures, final Q_total = 0
[2026-04-02 01:52:27,024][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-04-02 01:52:27,164][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,049 departures, final Q_total = 1
                                                                                                                       [2026-04-02 01:52:27,167][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:00<00:00,  7.28rep/s]
[2026-04-02 01:52:27,306][gibbsq.engines.numpy_engine][INFO] -   -> 1,036 arrivals, 1,036 departures, final Q_total = 0
[2026-04-02 01:52:27,308][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-04-02 01:52:27,433][gibbsq.engines.numpy_engine][INFO] -   -> 921 arrivals, 920 departures, final Q_total = 1
[2026-04-02 01:52:27,434][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 01:52:27,567][gibbsq.engines.numpy_engine][INFO] -   -> 988 arrivals, 988 departures, final Q_total = 0
[2026-04-02 01:52:27,570][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-04-02 01:52:27,704][gibbsq.engines.numpy_engine][INFO] -   -> 1,020 arrivals, 1,020 departures, final Q_total = 0
                                                                                                                       [2026-04-02 01:52:27,707][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:01<00:00,  7.35rep/s]
[2026-04-02 01:52:27,847][gibbsq.engines.numpy_engine][INFO] -   -> 1,045 arrivals, 1,044 departures, final Q_total = 1
[2026-04-02 01:52:27,849][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-04-02 01:52:27,995][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,051 departures, final Q_total = 3
                                                                                                                       [2026-04-02 01:52:28,001][__main__][INFO] -   E[Q_total] = 1.0100 ± 0.0304
policy: tiers:  57%|█████████████████████████████████████▏                           | 4/7 [00:05<00:04,  1.34s/policy][2026-04-02 01:52:28,004][__main__][INFO] - Evaluating Tier 3: UAS (alpha=5.0)...
                                                                                                                       [2026-04-02 01:52:28,007][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 01:52:28,135][gibbsq.engines.numpy_engine][INFO] -   -> 1,007 arrivals, 1,007 departures, final Q_total = 0
[2026-04-02 01:52:28,137][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-04-02 01:52:28,263][gibbsq.engines.numpy_engine][INFO] -   -> 1,006 arrivals, 1,004 departures, final Q_total = 2
[2026-04-02 01:52:28,265][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-04-02 01:52:28,387][gibbsq.engines.numpy_engine][INFO] -   -> 971 arrivals, 971 departures, final Q_total = 0
[2026-04-02 01:52:28,389][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-04-02 01:52:28,531][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,049 departures, final Q_total = 1
                                                                                                                       [2026-04-02 01:52:28,535][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:00<00:00,  7.59rep/s]
[2026-04-02 01:52:28,668][gibbsq.engines.numpy_engine][INFO] -   -> 1,032 arrivals, 1,030 departures, final Q_total = 2
[2026-04-02 01:52:28,669][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-04-02 01:52:28,797][gibbsq.engines.numpy_engine][INFO] -   -> 914 arrivals, 914 departures, final Q_total = 0
[2026-04-02 01:52:28,798][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 01:52:28,921][gibbsq.engines.numpy_engine][INFO] -   -> 986 arrivals, 985 departures, final Q_total = 1
[2026-04-02 01:52:28,922][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-04-02 01:52:29,060][gibbsq.engines.numpy_engine][INFO] -   -> 1,015 arrivals, 1,015 departures, final Q_total = 0
                                                                                                                       [2026-04-02 01:52:29,063][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:01<00:00,  7.58rep/s]
[2026-04-02 01:52:29,213][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,053 departures, final Q_total = 1
[2026-04-02 01:52:29,216][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-04-02 01:52:29,358][gibbsq.engines.numpy_engine][INFO] -   -> 1,049 arrivals, 1,047 departures, final Q_total = 2
                                                                                                                       [2026-04-02 01:52:29,365][__main__][INFO] -   E[Q_total] = 1.0250 ± 0.0335
policy: tiers:  71%|██████████████████████████████████████████████▍                  | 5/7 [00:06<00:02,  1.35s/policy][2026-04-02 01:52:29,367][__main__][INFO] - Evaluating Tier 4: Proportional (mu/Lambda)...
                                                                                                                       [2026-04-02 01:52:29,370][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 01:52:29,476][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,008 departures, final Q_total = 2
[2026-04-02 01:52:29,477][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-04-02 01:52:29,582][gibbsq.engines.numpy_engine][INFO] -   -> 1,017 arrivals, 1,016 departures, final Q_total = 1
[2026-04-02 01:52:29,583][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-04-02 01:52:29,684][gibbsq.engines.numpy_engine][INFO] -   -> 974 arrivals, 973 departures, final Q_total = 1
[2026-04-02 01:52:29,685][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-04-02 01:52:29,794][gibbsq.engines.numpy_engine][INFO] -   -> 1,040 arrivals, 1,040 departures, final Q_total = 0
[2026-04-02 01:52:29,796][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-04-02 01:52:29,906][gibbsq.engines.numpy_engine][INFO] -   -> 1,025 arrivals, 1,024 departures, final Q_total = 1
                                                                                                                       [2026-04-02 01:52:29,909][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:00<00:00,  9.30rep/s]
[2026-04-02 01:52:30,004][gibbsq.engines.numpy_engine][INFO] -   -> 902 arrivals, 894 departures, final Q_total = 8
[2026-04-02 01:52:30,005][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 01:52:30,106][gibbsq.engines.numpy_engine][INFO] -   -> 989 arrivals, 988 departures, final Q_total = 1
[2026-04-02 01:52:30,107][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-04-02 01:52:30,232][gibbsq.engines.numpy_engine][INFO] -   -> 1,027 arrivals, 1,027 departures, final Q_total = 0
[2026-04-02 01:52:30,233][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-04-02 01:52:30,379][gibbsq.engines.numpy_engine][INFO] -   -> 1,025 arrivals, 1,023 departures, final Q_total = 2
[2026-04-02 01:52:30,381][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-04-02 01:52:30,521][gibbsq.engines.numpy_engine][INFO] -   -> 1,068 arrivals, 1,065 departures, final Q_total = 3
                                                                                                                       [2026-04-02 01:52:30,530][__main__][INFO] -   E[Q_total] = 1.3704 ± 0.0569
policy: tiers:  86%|███████████████████████████████████████████████████████▋         | 6/7 [00:07<00:01,  1.29s/policy][2026-04-02 01:52:30,535][__main__][INFO] - Evaluating Tier 4: Uniform (1/N)...
                                                                                                                       [2026-04-02 01:52:30,539][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 01:52:30,687][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-04-02 01:52:30,689][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-04-02 01:52:30,837][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-04-02 01:52:30,839][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-04-02 01:52:30,985][gibbsq.engines.numpy_engine][INFO] -   -> 991 arrivals, 990 departures, final Q_total = 1
[2026-04-02 01:52:30,987][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-04-02 01:52:31,133][gibbsq.engines.numpy_engine][INFO] -   -> 1,040 arrivals, 1,040 departures, final Q_total = 0
                                                                                                                       [2026-04-02 01:52:31,137][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:00<00:00,  6.71rep/s]
[2026-04-02 01:52:31,291][gibbsq.engines.numpy_engine][INFO] -   -> 1,027 arrivals, 1,026 departures, final Q_total = 1
[2026-04-02 01:52:31,293][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-04-02 01:52:31,379][gibbsq.engines.numpy_engine][INFO] -   -> 911 arrivals, 911 departures, final Q_total = 0
[2026-04-02 01:52:31,381][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 01:52:31,488][gibbsq.engines.numpy_engine][INFO] -   -> 993 arrivals, 991 departures, final Q_total = 2
[2026-04-02 01:52:31,489][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-04-02 01:52:31,592][gibbsq.engines.numpy_engine][INFO] -   -> 1,029 arrivals, 1,027 departures, final Q_total = 2
[2026-04-02 01:52:31,593][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-04-02 01:52:31,695][gibbsq.engines.numpy_engine][INFO] -   -> 1,020 arrivals, 1,016 departures, final Q_total = 4
                                                                                                                       [2026-04-02 01:52:31,698][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:01<00:00,  7.96rep/s]
[2026-04-02 01:52:31,812][gibbsq.engines.numpy_engine][INFO] -   -> 1,058 arrivals, 1,058 departures, final Q_total = 0
                                                                                                                       [2026-04-02 01:52:31,818][__main__][INFO] -   E[Q_total] = 1.5965 ± 0.0857
policy: tiers: 100%|█████████████████████████████████████████████████████████████████| 7/7 [00:09<00:00,  1.30s/policy]
[2026-04-02 01:52:31,822][__main__][INFO] -
Evaluating Tier 5: N-GibbsQ (REINFORCE trained)...
[2026-04-02 01:52:31,824][__main__][INFO] - Using neural weights from C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\reinforce_train\run_20260402_015050\n_gibbsq_reinforce_weights.eqx
INFO:2026-04-02 01:52:31,845:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:52:31,845][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:52:33,229][__main__][INFO] - Evaluating N-GibbsQ (deterministic)...
policy eval (DeterministicNeuralPolicy):   0%|                                                 | 0/10 [00:00<?, ?rep/s][2026-04-02 01:52:33,233][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)
[2026-04-02 01:52:33,895][gibbsq.engines.numpy_engine][INFO] -   -> 1,013 arrivals, 1,012 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  10%|████                                     | 1/10 [00:00<00:05,  1.51rep/s][2026-04-02 01:52:33,897][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-04-02 01:52:34,540][gibbsq.engines.numpy_engine][INFO] -   -> 1,015 arrivals, 1,015 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  20%|████████▏                                | 2/10 [00:01<00:05,  1.53rep/s][2026-04-02 01:52:34,542][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-04-02 01:52:35,162][gibbsq.engines.numpy_engine][INFO] -   -> 961 arrivals, 961 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  30%|████████████▎                            | 3/10 [00:01<00:04,  1.56rep/s][2026-04-02 01:52:35,164][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-04-02 01:52:35,829][gibbsq.engines.numpy_engine][INFO] -   -> 1,043 arrivals, 1,042 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  40%|████████████████▍                        | 4/10 [00:02<00:03,  1.54rep/s][2026-04-02 01:52:35,832][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-04-02 01:52:36,511][gibbsq.engines.numpy_engine][INFO] -   -> 1,016 arrivals, 1,016 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  50%|████████████████████▌                    | 5/10 [00:03<00:03,  1.51rep/s][2026-04-02 01:52:36,512][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-04-02 01:52:37,115][gibbsq.engines.numpy_engine][INFO] -   -> 908 arrivals, 907 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  60%|████████████████████████▌                | 6/10 [00:03<00:02,  1.56rep/s][2026-04-02 01:52:37,117][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 01:52:37,772][gibbsq.engines.numpy_engine][INFO] -   -> 983 arrivals, 982 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  70%|████████████████████████████▋            | 7/10 [00:04<00:01,  1.55rep/s][2026-04-02 01:52:37,774][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-04-02 01:52:38,430][gibbsq.engines.numpy_engine][INFO] -   -> 1,023 arrivals, 1,022 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  80%|████████████████████████████████▊        | 8/10 [00:05<00:01,  1.54rep/s][2026-04-02 01:52:38,432][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-04-02 01:52:39,116][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,048 departures, final Q_total = 2
policy eval (DeterministicNeuralPolicy):  90%|████████████████████████████████████▉    | 9/10 [00:05<00:00,  1.51rep/s][2026-04-02 01:52:39,118][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-04-02 01:52:39,924][gibbsq.engines.numpy_engine][INFO] -   -> 1,074 arrivals, 1,074 departures, final Q_total = 0
[2026-04-02 01:52:39,931][__main__][INFO] -   E[Q_total] = 1.1137 ± 0.0397
[2026-04-02 01:52:39,932][__main__][INFO] -
============================================================
[2026-04-02 01:52:39,933][__main__][INFO] -   Parity Analysis (Corrected Criteria)
[2026-04-02 01:52:39,934][__main__][INFO] - ============================================================
[2026-04-02 01:52:39,936][__main__][INFO] - N-GibbsQ (Platinum/Greedy): E[Q] = 1.1137
[2026-04-02 01:52:39,936][__main__][INFO] - Reference thresholds:
[2026-04-02 01:52:39,937][__main__][INFO] -   JSSQ (Tier 2): E[Q] = 0.9949
[2026-04-02 01:52:39,938][__main__][INFO] -   UAS (Tier 3): E[Q] = 1.1298
[2026-04-02 01:52:39,939][__main__][INFO] -   Proportional (Tier 4): E[Q] = 1.3704
[2026-04-02 01:52:39,940][__main__][INFO] - Reference statistical bounds (95% CI):
[2026-04-02 01:52:39,940][__main__][INFO] -   JSSQ (Tier 2): E[Q] = 0.9949 ± 0.0294
[2026-04-02 01:52:39,942][__main__][INFO] -   UAS (Tier 3): E[Q] = 1.1298 ± 0.0372
[2026-04-02 01:52:39,943][__main__][INFO] -   Proportional (Tier 4): E[Q] = 1.3704 ± 0.0569
[2026-04-02 01:52:39,944][__main__][INFO] - PARITY RESULT: SILVER [OK] (Statistically matches empirical UAS baseline)
[2026-04-02 01:52:41,515][__main__][INFO] - Comparison plot saved to outputs\debug\policy\run_20260402_015219\corrected_policy_comparison.png, outputs\debug\policy\run_20260402_015219\corrected_policy_comparison.pdf
wandb: You can sync this run to the cloud by running:
wandb: wandb sync outputs\debug\policy\run_20260402_015219\wandb\offline-run-20260402_015219-eaqszz8v
wandb: Find logs at: outputs\debug\policy\run_20260402_015219\wandb\offline-run-20260402_015219-eaqszz8v\logs

[Experiment 'policy' Finished]
  Status: completed
  End Time: 2026-04-02 01:52:42+05:45
  Elapsed Duration: 27.013s
  -> [policy] Ended at 2026-04-02 01:52:42+05:45
  -> [policy] Status: completed
  -> [policy] Elapsed: 30.935s
8/12 policy:  67%|████████████████████████████████                | 8/12 [04:18<02:36, 39.09s/experiment, alias=policy]
[8/11] Running Statistical Verification Analysis (Phase 3b)...
9/12 stats:  67%|████████████████████████████████▋                | 8/12 [04:18<02:36, 39.09s/experiment, alias=policy][handoff] Launching stats (9/12)
  -> [stats] Started at 2026-04-02 01:52:42+05:45
==========================================================
 Starting Experiment: stats
 Start Time: 2026-04-02 01:52:45+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=stats
==========================================================
[2026-04-02 01:52:50,528][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-04-02 01:52:50,548:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:52:50,548][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:52:50,549][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-04-02 01:52:50,560][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\stats\run_20260402_015250
[2026-04-02 01:52:50,561][__main__][INFO] - ============================================================
[2026-04-02 01:52:50,564][__main__][INFO] -   Phase VII: Statistical Summary
[2026-04-02 01:52:50,565][__main__][INFO] - ============================================================
[2026-04-02 01:52:50,744][__main__][INFO] - Initiating statistical comparison (n=10 seeds).
[2026-04-02 01:52:50,777][__main__][INFO] - Environment: N=2, rho=0.40
[2026-04-02 01:52:51,969][__main__][INFO] - Loaded trained model from C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\reinforce_train\run_20260402_015050\n_gibbsq_reinforce_weights.eqx
stats:   0%|                                                                                  | 0/2 [00:00<?, ?stage/s][2026-04-02 01:52:51,977][__main__][INFO] - Running 10 GibbsQ SSA simulations with policy='uas'...
stats:  50%|█████████████████████████████████████                                     | 1/2 [00:01<00:01,  1.38s/stage][2026-04-02 01:52:53,363][__main__][INFO] - Running 10 Neural SSA simulations...
[2026-04-02 01:52:53,363][__main__][INFO] - Neural evaluation mode: deterministic
stats: 100%|██████████████████████████████████████████████████████████████████████████| 2/2 [00:08<00:00,  4.06s/stage]
[2026-04-02 01:53:00,104][__main__][INFO] -
============================================================
[2026-04-02 01:53:00,105][__main__][INFO] -   STATISTICAL SUMMARY
[2026-04-02 01:53:00,105][__main__][INFO] - ============================================================
[2026-04-02 01:53:00,105][__main__][INFO] - GibbsQ E[Q]:   1.0834 ± 0.0911
[2026-04-02 01:53:00,106][__main__][INFO] - N-GibbsQ E[Q]:   1.1137 ± 0.1254
[2026-04-02 01:53:00,106][__main__][INFO] - Rel. Improve:  -2.80%
[2026-04-02 01:53:00,106][__main__][INFO] - ----------------------------------------
[2026-04-02 01:53:00,106][__main__][INFO] - P-Value:       5.44e-01 (NOT SIGNIFICANT)
[2026-04-02 01:53:00,107][__main__][INFO] - Effect Size:   0.28 (Cohen's d)
[2026-04-02 01:53:00,107][__main__][INFO] - 95% CI (Diff): [-0.0726, 0.1333]
[2026-04-02 01:53:00,107][__main__][INFO] - ============================================================
[2026-04-02 01:53:00,673][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:53:00,747][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:53:00,828][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:53:01,477][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:53:01,534][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode

[Experiment 'stats' Finished]
  Status: completed
  End Time: 2026-04-02 01:53:02+05:45
  Elapsed Duration: 16.266s
  -> [stats] Ended at 2026-04-02 01:53:02+05:45
  -> [stats] Status: completed
  -> [stats] Elapsed: 20.158s
9/12 stats:  75%|█████████████████████████████████████▌            | 9/12 [04:38<01:39, 33.17s/experiment, alias=stats]
[9/11] Running Generalization Stress Heatmaps...
10/12 generalize:  75%|█████████████████████████████████           | 9/12 [04:38<01:39, 33.17s/experiment, alias=stats][handoff] Launching generalize (10/12)
  -> [generalize] Started at 2026-04-02 01:53:02+05:45
==========================================================
 Starting Experiment: generalize
 Start Time: 2026-04-02 01:53:06+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=generalize
==========================================================
[2026-04-02 01:53:10,684][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-04-02 01:53:10,705:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:53:10,705][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:53:10,705][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-04-02 01:53:10,718][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\generalize\run_20260402_015310
[2026-04-02 01:53:10,718][__main__][INFO] - ============================================================
[2026-04-02 01:53:10,719][__main__][INFO] -   Phase VIII: Generalization & Stress Heatmap
[2026-04-02 01:53:10,719][__main__][INFO] - ============================================================
[2026-04-02 01:53:10,892][__main__][INFO] - Initiating Generalization Sweep (Scales=[0.5, 2.0], rho=[0.5, 0.85])
[2026-04-02 01:53:12,189][__main__][INFO] - Evaluating N-GibbsQ improvement ratio (GibbsQ / Neural) on 5x5 Grid...
generalize:   0%|                                                                              | 0/4 [00:00<?, ?cell/s][2026-04-02 01:53:18,197][__main__][INFO] -    Scale=  0.5x | rho=0.50 | Improvement=1.07x
generalize:  25%|███████████▊                                   | 1/4 [00:06<00:18,  6.01s/cell, scale=0.50x, rho=0.50][2026-04-02 01:53:26,506][__main__][INFO] -    Scale=  0.5x | rho=0.85 | Improvement=1.03x
generalize:  50%|███████████████████████▌                       | 2/4 [00:14<00:14,  7.36s/cell, scale=0.50x, rho=0.85][2026-04-02 01:53:44,361][__main__][INFO] -    Scale=  2.0x | rho=0.50 | Improvement=1.07x
generalize:  75%|███████████████████████████████████▎           | 3/4 [00:32<00:12, 12.15s/cell, scale=2.00x, rho=0.50][2026-04-02 01:54:14,948][__main__][INFO] -    Scale=  2.0x | rho=0.85 | Improvement=0.88x
generalize: 100%|███████████████████████████████████████████████| 4/4 [01:02<00:00, 15.69s/cell, scale=2.00x, rho=0.85]
[2026-04-02 01:54:17,096][__main__][INFO] - Generalization analysis complete. Heatmap saved to outputs\debug\generalize\run_20260402_015310\generalization_heatmap.png, outputs\debug\generalize\run_20260402_015310\generalization_heatmap.pdf

[Experiment 'generalize' Finished]
  Status: completed
  End Time: 2026-04-02 01:54:17+05:45
  Elapsed Duration: 71.575s
  -> [generalize] Ended at 2026-04-02 01:54:17+05:45
  -> [generalize] Status: completed
  -> [generalize] Elapsed: 75.449s
10/12 generalize:  83%|███████████████████████████████▋      | 10/12 [05:54<01:32, 46.22s/experiment, alias=generalize]
[10/11] Running Critical Load Boundary Analysis...
11/12 critical:  83%|█████████████████████████████████▎      | 10/12 [05:54<01:32, 46.22s/experiment, alias=generalize][handoff] Launching critical (11/12)
  -> [critical] Started at 2026-04-02 01:54:17+05:45
==========================================================
 Starting Experiment: critical
 Start Time: 2026-04-02 01:54:21+05:45
 Config Profile: debug
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=critical
==========================================================
[2026-04-02 01:54:26,072][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-04-02 01:54:26,093:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:54:26,093][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:54:26,094][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-04-02 01:54:26,104][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\debug\critical\run_20260402_015426
[2026-04-02 01:54:26,105][__main__][INFO] - ============================================================
[2026-04-02 01:54:26,108][__main__][INFO] -   Phase VIII: The Critical Stability Boundary
[2026-04-02 01:54:26,109][__main__][INFO] - ============================================================
[2026-04-02 01:54:27,424][__main__][INFO] - System Capacity: 2.50
[2026-04-02 01:54:27,425][__main__][INFO] - Targeting Load Boundary: [0.95]
critical:   0%|                                                                                 | 0/1 [00:00<?, ?rho/s][2026-04-02 01:54:27,445][__main__][INFO] - Evaluating Boundary rho=0.950 (Arrival=2.375)...
                                                                                                                       [2026-04-02 01:56:01,167][__main__][INFO] -    => N-GibbsQ E[Q]: 18.43 | GibbsQ E[Q]: 20.73
critical: 100%|██████████████████████████████████████████████████████████████| 1/1 [01:33<00:00, 93.74s/rho, rho=0.950]
[2026-04-02 01:56:01,565][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:56:01,684][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:56:01,819][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:56:02,531][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
Traceback (most recent call last):
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\experiments\evaluation\n_gibbsq_evals\critical_load.py", line 239, in <module>
    main()
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\hydra\main.py", line 94, in decorated_main
    _run_hydra(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\hydra\_internal\utils.py", line 394, in _run_hydra
    _run_app(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\hydra\_internal\utils.py", line 457, in _run_app
    run_and_report(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\hydra\_internal\utils.py", line 220, in run_and_report
    return func()
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\hydra\_internal\utils.py", line 458, in <lambda>
    lambda: hydra.run(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\hydra\_internal\hydra.py", line 119, in run
    ret = run_job(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\hydra\core\utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
  File "C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\experiments\evaluation\n_gibbsq_evals\critical_load.py", line 235, in main
    return test.execute(jax.random.PRNGKey(cfg.simulation.seed))
  File "C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\experiments\evaluation\n_gibbsq_evals\critical_load.py", line 180, in execute
    self._plot(self.rho_vals, neural_results, gibbs_results)
  File "C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\experiments\evaluation\n_gibbsq_evals\critical_load.py", line 198, in _plot
    fig = plot_critical_load(
  File "C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py", line 939, in plot_critical_load
    save_chart(fig, Path(save_path), formats or ["png", "pdf"])
  File "C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\utils\chart_exporter.py", line 95, in save_chart
    fig.savefig(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\figure.py", line 3490, in savefig
    self.canvas.print_figure(fname, **kwargs)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\backend_bases.py", line 2157, in print_figure
    self.figure.draw(renderer)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\artist.py", line 94, in draw_wrapper
    result = draw(artist, renderer, *args, **kwargs)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\figure.py", line 3257, in draw
    mimage._draw_list_compositing_images(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\image.py", line 134, in _draw_list_compositing_images
    a.draw(renderer)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\axes\_base.py", line 3226, in draw
    mimage._draw_list_compositing_images(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\image.py", line 134, in _draw_list_compositing_images
    a.draw(renderer)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\axis.py", line 1412, in draw
    self.label.draw(renderer)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\text.py", line 752, in draw
    bbox, info, descent = self._get_layout(renderer)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\text.py", line 382, in _get_layout
    w, h, d = _get_text_metrics_with_cache(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\text.py", line 69, in _get_text_metrics_with_cache
    return _get_text_metrics_with_cache_impl(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\text.py", line 77, in _get_text_metrics_with_cache_impl
    return renderer_ref().get_text_width_height_descent(text, fontprop, ismath)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\backends\_backend_pdf_ps.py", line 152, in get_text_width_height_descent
    parse = self._text2path.mathtext_parser.parse(s, 72, prop)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\mathtext.py", line 86, in parse
    return self._parse_cached(s, dpi, prop, antialiased, load_glyph_flags)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\mathtext.py", line 100, in _parse_cached
    box = self._parser.parse(s, fontset, fontsize, dpi)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\_mathtext.py", line 2164, in parse
    result = self._expression.parse_string(s)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1334, in parse_string
    loc, tokens = self._parse(instring, 0)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1094, in _parseCache
    value = self._parseNoCache(instring, loc, do_actions, callPreParse)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 956, in _parseNoCache
    loc, tokens = self.parseImpl(instring, pre_loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 4630, in parseImpl
    loc, exprtokens = e._parse(instring, loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1094, in _parseCache
    value = self._parseNoCache(instring, loc, do_actions, callPreParse)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 956, in _parseNoCache
    loc, tokens = self.parseImpl(instring, pre_loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 5790, in parseImpl
    return super().parseImpl(instring, loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 5675, in parseImpl
    loc, tokens = self_expr_parse(instring, loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1094, in _parseCache
    value = self._parseNoCache(instring, loc, do_actions, callPreParse)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 956, in _parseNoCache
    loc, tokens = self.parseImpl(instring, pre_loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 4608, in parseImpl
    loc, resultlist = self.exprs[0]._parse(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1094, in _parseCache
    value = self._parseNoCache(instring, loc, do_actions, callPreParse)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 995, in _parseNoCache
    tokens = fn(instring, tokens_start, ret_tokens)  # type: ignore [call-arg, arg-type]
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 287, in wrapper
    return func(*args[limit:])
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\_mathtext.py", line 2191, in math_string
    return self._math_expression.parse_string(toks[0][1:-1], parse_all=True)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1334, in parse_string
    loc, tokens = self._parse(instring, 0)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1094, in _parseCache
    value = self._parseNoCache(instring, loc, do_actions, callPreParse)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 956, in _parseNoCache
    loc, tokens = self.parseImpl(instring, pre_loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 5675, in parseImpl
    loc, tokens = self_expr_parse(instring, loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1094, in _parseCache
    value = self._parseNoCache(instring, loc, do_actions, callPreParse)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 956, in _parseNoCache
    loc, tokens = self.parseImpl(instring, pre_loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 6258, in parseImpl
    return super().parseImpl(instring, loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 5151, in parseImpl
    return self.expr._parse(instring, loc, do_actions, callPreParse=False)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1094, in _parseCache
    value = self._parseNoCache(instring, loc, do_actions, callPreParse)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 956, in _parseNoCache
    loc, tokens = self.parseImpl(instring, pre_loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 4870, in parseImpl
    return e._parse(instring, loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1094, in _parseCache
    value = self._parseNoCache(instring, loc, do_actions, callPreParse)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 956, in _parseNoCache
    loc, tokens = self.parseImpl(instring, pre_loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 4870, in parseImpl
    return e._parse(instring, loc, do_actions)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1094, in _parseCache
    value = self._parseNoCache(instring, loc, do_actions, callPreParse)
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 956, in _parseNoCache
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 4608, in parseImpl
    loc, resultlist = self.exprs[0]._parse(
  File "C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pyparsing\core.py", line 1088, in _parseCache
    with ParserElement.packrat_cache_lock:
KeyboardInterrupt

Experiment interrupted by user

[Experiment 'critical' Finished]
  Status: interrupted by user
  End Time: 2026-04-02 01:56:02+05:45
  Elapsed Duration: 101.323s
11/12 critical:  83%|███████████████████████████████████       | 10/12 [07:39<01:31, 45.92s/experiment, alias=critical]

Pipeline interrupted by user.

==========================================================
  Pipeline interrupted before completion.
  Pipeline Status: interrupted by user
  Pipeline Ended At: 2026-04-02 01:56:02+05:45
  Total Pipeline Runtime: 459.254s
  Review '/outputs/' for your plots and logs.
==========================================================
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ> ^C
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ> ^C
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ> python scripts/execution/reproduction_pipeline.py --config-name small
==========================================================
  GibbsQ Research Paper: Final Execution Pipeline
==========================================================
  Progress Mode: auto
  Pipeline Started At: 2026-04-02 01:56:12+05:45

[Initiating Pipeline...]

pipeline:   0%|                                                                         | 0/12 [00:00<?, ?experiment/s]
[Pre-Flight] Running Configuration Sanity Checks...
1/12 check_configs:   0%|                                                               | 0/12 [00:00<?, ?experiment/s][handoff] Launching check_configs (1/12)
  -> [check_configs] Started at 2026-04-02 01:56:12+05:45
==========================================================
 Starting Experiment: check_configs
 Start Time: 2026-04-02 01:56:16+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=check_configs
==========================================================
check_configs: roots:   0%|                                                                  | 0/4 [00:00<?, ?config/s][OK] Config debug validated successfully.
[OK] Config small validated successfully.
[OK] Config default validated successfully.
[OK] Config final_experiment validated successfully.
check_configs: resolved paths:   0%|                                                          | 0/48 [00:00<?, ?path/s][OK] Resolved experiment path hyperqual (profile=debug, overrides=['++active_profile=debug', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=debug, overrides=['++active_profile=debug', '++active_experiment=reinforce_check']) validated successfully.
check_configs: resolved paths:   4%|██                                                | 2/48 [00:00<00:12,  3.55path/s][OK] Resolved experiment path drift (profile=debug, overrides=['++active_profile=debug', '++active_experiment=drift']) validated successfully.
[OK] Resolved experiment path sweep (profile=debug, overrides=['++active_profile=debug', '++active_experiment=sweep']) validated successfully.
[OK] Resolved experiment path stress (profile=debug, overrides=['++active_profile=debug', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
check_configs: resolved paths:  10%|█████▏                                            | 5/48 [00:01<00:11,  3.89path/s][OK] Resolved experiment path policy (profile=debug, overrides=['++active_profile=debug', '++active_experiment=policy']) validated successfully.
[OK] Resolved experiment path bc_train (profile=debug, overrides=['++active_profile=debug', '++active_experiment=bc_train']) validated successfully.
check_configs: resolved paths:  15%|███████▎                                          | 7/48 [00:01<00:11,  3.70path/s][OK] Resolved experiment path reinforce_train (profile=debug, overrides=['++active_profile=debug', '++active_experiment=reinforce_train']) validated successfully.
[OK] Resolved experiment path stats (profile=debug, overrides=['++active_profile=debug', '++active_experiment=stats']) validated successfully.
check_configs: resolved paths:  19%|█████████▍                                        | 9/48 [00:02<00:10,  3.57path/s][OK] Resolved experiment path generalize (profile=debug, overrides=['++active_profile=debug', '++active_experiment=generalize']) validated successfully.
[OK] Resolved experiment path ablation (profile=debug, overrides=['++active_profile=debug', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=debug, overrides=['++active_profile=debug', '++active_experiment=critical']) validated successfully.
check_configs: resolved paths:  25%|████████████▎                                    | 12/48 [00:03<00:09,  3.78path/s][OK] Resolved experiment path hyperqual (profile=small, overrides=['++active_profile=small', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=small, overrides=['++active_profile=small', '++active_experiment=reinforce_check']) validated successfully.
[OK] Resolved experiment path drift (profile=small, overrides=['++active_profile=small', '++active_experiment=drift']) validated successfully.
check_configs: resolved paths:  31%|███████████████▎                                 | 15/48 [00:03<00:08,  3.91path/s][OK] Resolved experiment path sweep (profile=small, overrides=['++active_profile=small', '++active_experiment=sweep']) validated successfully.
[OK] Resolved experiment path stress (profile=small, overrides=['++active_profile=small', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
check_configs: resolved paths:  35%|█████████████████▎                               | 17/48 [00:04<00:07,  3.92path/s][OK] Resolved experiment path policy (profile=small, overrides=['++active_profile=small', '++active_experiment=policy']) validated successfully.
[OK] Resolved experiment path bc_train (profile=small, overrides=['++active_profile=small', '++active_experiment=bc_train']) validated successfully.
check_configs: resolved paths:  40%|███████████████████▍                             | 19/48 [00:04<00:07,  3.82path/s][OK] Resolved experiment path reinforce_train (profile=small, overrides=['++active_profile=small', '++active_experiment=reinforce_train']) validated successfully.
[OK] Resolved experiment path stats (profile=small, overrides=['++active_profile=small', '++active_experiment=stats']) validated successfully.
check_configs: resolved paths:  44%|█████████████████████▍                           | 21/48 [00:05<00:07,  3.80path/s][OK] Resolved experiment path generalize (profile=small, overrides=['++active_profile=small', '++active_experiment=generalize']) validated successfully.
[OK] Resolved experiment path ablation (profile=small, overrides=['++active_profile=small', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=small, overrides=['++active_profile=small', '++active_experiment=critical']) validated successfully.
check_configs: resolved paths:  50%|████████████████████████▌                        | 24/48 [00:06<00:06,  3.88path/s][OK] Resolved experiment path hyperqual (profile=default, overrides=['++active_profile=default', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=default, overrides=['++active_profile=default', '++active_experiment=reinforce_check']) validated successfully.
check_configs: resolved paths:  54%|██████████████████████████▌                      | 26/48 [00:06<00:06,  3.59path/s][OK] Resolved experiment path drift (profile=default, overrides=['++active_profile=default', '++active_experiment=drift']) validated successfully.
[OK] Resolved experiment path sweep (profile=default, overrides=['++active_profile=default', '++active_experiment=sweep']) validated successfully.
check_configs: resolved paths:  58%|████████████████████████████▌                    | 28/48 [00:07<00:05,  3.47path/s][OK] Resolved experiment path stress (profile=default, overrides=['++active_profile=default', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Resolved experiment path policy (profile=default, overrides=['++active_profile=default', '++active_experiment=policy']) validated successfully.
check_configs: resolved paths:  62%|██████████████████████████████▋                  | 30/48 [00:08<00:05,  3.15path/s][OK] Resolved experiment path bc_train (profile=default, overrides=['++active_profile=default', '++active_experiment=bc_train']) validated successfully.
[OK] Resolved experiment path reinforce_train (profile=default, overrides=['++active_profile=default', '++active_experiment=reinforce_train']) validated successfully.
check_configs: resolved paths:  67%|████████████████████████████████▋                | 32/48 [00:09<00:05,  3.06path/s][OK] Resolved experiment path stats (profile=default, overrides=['++active_profile=default', '++active_experiment=stats']) validated successfully.
[OK] Resolved experiment path generalize (profile=default, overrides=['++active_profile=default', '++active_experiment=generalize']) validated successfully.
check_configs: resolved paths:  71%|██████████████████████████████████▋              | 34/48 [00:09<00:04,  3.17path/s][OK] Resolved experiment path ablation (profile=default, overrides=['++active_profile=default', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=default, overrides=['++active_profile=default', '++active_experiment=critical']) validated successfully.
check_configs: resolved paths:  75%|████████████████████████████████████▊            | 36/48 [00:10<00:03,  3.26path/s][OK] Resolved experiment path hyperqual (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=hyperqual']) validated successfully.
[OK] Resolved experiment path reinforce_check (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=reinforce_check']) validated successfully.
check_configs: resolved paths:  79%|██████████████████████████████████████▊          | 38/48 [00:10<00:03,  3.32path/s][OK] Resolved experiment path drift (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=drift']) validated successfully.
[OK] Resolved experiment path sweep (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=sweep']) validated successfully.
check_configs: resolved paths:  83%|████████████████████████████████████████▊        | 40/48 [00:11<00:02,  3.17path/s][OK] Resolved experiment path stress (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Resolved experiment path policy (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=policy']) validated successfully.
check_configs: resolved paths:  88%|██████████████████████████████████████████▉      | 42/48 [00:12<00:02,  2.92path/s][OK] Resolved experiment path bc_train (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=bc_train']) validated successfully.
[OK] Resolved experiment path reinforce_train (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=reinforce_train']) validated successfully.
check_configs: resolved paths:  92%|████████████████████████████████████████████▉    | 44/48 [00:13<00:01,  2.62path/s][OK] Resolved experiment path stats (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=stats']) validated successfully.
[OK] Resolved experiment path generalize (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=generalize']) validated successfully.
check_configs: resolved paths:  96%|██████████████████████████████████████████████▉  | 46/48 [00:14<00:00,  2.50path/s][OK] Resolved experiment path ablation (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=ablation']) validated successfully.
[OK] Resolved experiment path critical (profile=final_experiment, overrides=['++active_profile=final_experiment', '++active_experiment=critical']) validated successfully.
check_configs: public paths:   0%|                                                            | 0/12 [00:00<?, ?path/s][OK] Public experiment path hyperqual (base=default, overrides=['++active_profile=default', '++active_experiment=hyperqual']) validated successfully.
[OK] Public experiment path reinforce_check (base=default, overrides=['++active_profile=default', '++active_experiment=reinforce_check']) validated successfully.
check_configs: public paths:  17%|████████▋                                           | 2/12 [00:00<00:03,  3.18path/s][OK] Public experiment path drift (base=default, overrides=['++active_profile=default', '++active_experiment=drift']) validated successfully.
[OK] Public experiment path sweep (base=default, overrides=['++active_profile=default', '++active_experiment=sweep']) validated successfully.
check_configs: public paths:  33%|█████████████████▎                                  | 4/12 [00:01<00:02,  2.99path/s][OK] Public experiment path stress (base=default, overrides=['++active_profile=default', '++active_experiment=stress', '++jax.enabled=True']) validated successfully.
[OK] Public experiment path policy (base=default, overrides=['++active_profile=default', '++active_experiment=policy']) validated successfully.
check_configs: public paths:  50%|██████████████████████████                          | 6/12 [00:01<00:01,  3.06path/s][OK] Public experiment path bc_train (base=default, overrides=['++active_profile=default', '++active_experiment=bc_train']) validated successfully.
[OK] Public experiment path reinforce_train (base=default, overrides=['++active_profile=default', '++active_experiment=reinforce_train']) validated successfully.
check_configs: public paths:  67%|██████████████████████████████████▋                 | 8/12 [00:02<00:01,  3.18path/s][OK] Public experiment path stats (base=default, overrides=['++active_profile=default', '++active_experiment=stats']) validated successfully.
[OK] Public experiment path generalize (base=default, overrides=['++active_profile=default', '++active_experiment=generalize']) validated successfully.
check_configs: public paths:  83%|██████████████████████████████████████████▌        | 10/12 [00:03<00:00,  3.03path/s][OK] Public experiment path ablation (base=default, overrides=['++active_profile=default', '++active_experiment=ablation']) validated successfully.
[OK] Public experiment path critical (base=default, overrides=['++active_profile=default', '++active_experiment=critical']) validated successfully.

[SUCCESS] All configs passed validation.

[Experiment 'check_configs' Finished]
  Status: completed
  End Time: 2026-04-02 01:56:40+05:45
  Elapsed Duration: 24.018s
  -> [check_configs] Ended at 2026-04-02 01:56:40+05:45
  -> [check_configs] Status: completed
  -> [check_configs] Elapsed: 28.393s
1/12 check_configs:   8%|██▊                               | 1/12 [00:28<05:12, 28.39s/experiment, alias=check_configs]
[1/11] Running REINFORCE Gradient validation...
2/12 reinforce_check:   8%|██▋                             | 1/12 [00:28<05:12, 28.39s/experiment, alias=check_configs][handoff] Launching reinforce_check (2/12)
  -> [reinforce_check] Started at 2026-04-02 01:56:40+05:45
==========================================================
 Starting Experiment: reinforce_check
 Start Time: 2026-04-02 01:56:44+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=reinforce_check
==========================================================
[2026-04-02 01:56:49,193][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\reinforce_check\run_20260402_015649
[2026-04-02 01:56:49,193][__main__][INFO] - ============================================================
[2026-04-02 01:56:49,194][__main__][INFO] -   REINFORCE Gradient Estimator Validation
[2026-04-02 01:56:49,194][__main__][INFO] - ============================================================
[2026-04-02 01:56:49,194][__main__][INFO] - Validating REINFORCE gradients against the trainer-aligned first-action objective.
INFO:2026-04-02 01:56:49,261:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:56:49,261][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:56:53,642][__main__][INFO] - Statistical Scaling: D=50 parameters.
[2026-04-02 01:56:53,643][__main__][INFO] - Adjusted Z-Critical Threshold: 3.29
[2026-04-02 01:56:53,643][__main__][INFO] - Scaled n_samples from 5000 -> 5000 to maintain confidence bounds.
[2026-04-02 01:56:53,644][__main__][INFO] - Computing REINFORCE gradient estimate...
[2026-04-02 01:57:00,984][__main__][INFO] - Computing finite-difference gradient estimate...
reinforce_check: FD params:   2%|█                                                   | 1/50 [00:00<00:43,  1.13param/s][2026-04-02 01:57:03,420][__main__][INFO] -   [OK] Param  10/50  (idx  2385): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
                                                                                                                       [2026-04-02 01:57:03,582][__main__][INFO] -   [OK] Param  20/50  (idx 16253): RF=  0.001668 | FD=  0.001258 | diff=  0.000410 | z= 0.00 | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
                                                                                                                       [2026-04-02 01:57:03,730][__main__][INFO] -   [OK] Param  30/50  (idx  2928): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
reinforce_check: FD params:  66%|█████████████████████████████████▋                 | 33/50 [00:01<00:00, 28.77param/s][2026-04-02 01:57:03,911][__main__][INFO] -   [OK] Param  40/50  (idx  1282): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
                                                                                                                       [2026-04-02 01:57:04,070][__main__][INFO] -   [OK] Param  50/50  (idx  9341): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RunningRelErr= 0.3261 | RunningCosSim= 1.0000
reinforce_check: FD params: 100%|███████████████████████████████████████████████████| 50/50 [00:01<00:00, 29.45param/s]
[2026-04-02 01:57:04,074][__main__][INFO] - Relative error: 0.3261
[2026-04-02 01:57:04,075][__main__][INFO] - Cosine similarity: 1.0000
[2026-04-02 01:57:04,075][__main__][INFO] - Bias estimate (L2): 0.001190
[2026-04-02 01:57:04,075][__main__][INFO] - Relative bias: 0.3261
[2026-04-02 01:57:04,075][__main__][INFO] - Variance estimate: 0.000000
[2026-04-02 01:57:04,076][__main__][INFO] - Passed: True
[2026-04-02 01:57:04,077][__main__][INFO] - Results saved to outputs\small\reinforce_check\run_20260402_015649\gradient_check_result.json
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\utils\chart_exporter.py:95: UserWarning: Glyph 10003 (\N{CHECK MARK}) missing from font(s) Times New Roman.
  fig.savefig(
[2026-04-02 01:57:05,928][__main__][INFO] - Gradient scatter plot saved to outputs\small\reinforce_check\run_20260402_015649\gradient_scatter.png, outputs\small\reinforce_check\run_20260402_015649\gradient_scatter.pdf
[2026-04-02 01:57:05,929][__main__][INFO] - GRADIENT CHECK PASSED - REINFORCE estimator is valid

[Experiment 'reinforce_check' Finished]
  Status: completed
  End Time: 2026-04-02 01:57:06+05:45
  Elapsed Duration: 21.857s
  -> [reinforce_check] Ended at 2026-04-02 01:57:06+05:45
  -> [reinforce_check] Status: completed
  -> [reinforce_check] Elapsed: 26.145s
2/12 reinforce_check:  17%|█████                         | 2/12 [00:54<04:30, 27.07s/experiment, alias=reinforce_check]
[2/11] Running Drift Verification (Phase 1a)...
3/12 drift:  17%|██████▋                                 | 2/12 [00:54<04:30, 27.07s/experiment, alias=reinforce_check][handoff] Launching drift (3/12)
  -> [drift] Started at 2026-04-02 01:57:06+05:45
==========================================================
 Starting Experiment: drift
 Start Time: 2026-04-02 01:57:11+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=drift
==========================================================
[2026-04-02 01:57:16,776][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\drift\run_20260402_015716
[2026-04-02 01:57:16,778][__main__][INFO] - System: N=2, lam=1.0, alpha=1.0, cap=2.5000
[2026-04-02 01:57:16,778][__main__][INFO] - Proof bounds: R=2.4431, eps=0.750000
[2026-04-02 01:57:16,778][__main__][INFO] - --- Grid Evaluation (q_max=50) ---
drift: grid:   0%|                                                                            | 0/3 [00:00<?, ?stage/s][2026-04-02 01:57:16,784][__main__][INFO] - States evaluated: 2,601
[2026-04-02 01:57:16,785][__main__][INFO] - Bound violations: 0
[2026-04-02 01:57:17,512][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:57:17,913][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:57:18,494][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:57:18,679][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:57:18,830][__main__][INFO] - Saved: outputs\small\drift\run_20260402_015716\drift_heatmap.png, outputs\small\drift\run_20260402_015716\drift_heatmap.pdf
[2026-04-02 01:57:19,049][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:57:19,229][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:57:19,839][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:57:19,932][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-04-02 01:57:20,143][__main__][INFO] - Saved: outputs\small\drift\run_20260402_015716\drift_vs_norm.png, outputs\small\drift\run_20260402_015716\drift_vs_norm.pdf
drift: grid: 100%|████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.12s/stage, N=2, mode=grid]
[2026-04-02 01:57:20,146][__main__][INFO] - Drift verification complete.

[Experiment 'drift' Finished]
  Status: completed
  End Time: 2026-04-02 01:57:20+05:45
  Elapsed Duration: 9.590s
  -> [drift] Ended at 2026-04-02 01:57:20+05:45
  -> [drift] Status: completed
  -> [drift] Elapsed: 14.117s
3/12 drift:  25%|████████████▌                                     | 3/12 [01:08<03:10, 21.16s/experiment, alias=drift]
[3/11] Running Stability Sweep (Phase 1b)...
4/12 sweep:  25%|████████████▌                                     | 3/12 [01:08<03:10, 21.16s/experiment, alias=drift][handoff] Launching sweep (4/12)
  -> [sweep] Started at 2026-04-02 01:57:20+05:45
==========================================================
 Starting Experiment: sweep
 Start Time: 2026-04-02 01:57:25+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=sweep
==========================================================
[2026-04-02 01:57:31,131][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-04-02 01:57:31,152:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:57:31,152][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:57:31,152][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-04-02 01:57:31,165][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\sweep\run_20260402_015731
[2026-04-02 01:57:31,165][gibbsq.utils.logging][INFO] - [Logging] WandB offline mode.
wandb: Tracking run with wandb version 0.23.1
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in outputs\small\sweep\run_20260402_015731\wandb\offline-run-20260402_015731-l4qt2xr3
[2026-04-02 01:57:34,270][gibbsq.utils.logging][INFO] - [Logging] WandB Run Linked: run_20260402_015731 (offline)
[2026-04-02 01:57:34,273][__main__][INFO] - System: N=2, cap=2.5000 | Backend: JAX
[2026-04-02 01:57:34,274][__main__][INFO] - Grid: 3 alpha x 3 rho x 10 reps
sweep:   0%|                                                                                   | 0/9 [00:00<?, ?cell/s][2026-04-02 01:57:34,280][__main__][INFO] -
------------------------------------------------------------
  rho = 0.50  (lam = 1.2500)
------------------------------------------------------------
                                                                                                                       [2026-04-02 01:57:36,512][__main__][INFO] -   alpha=  0.10 | E[Q_total]=    1.87 | OK  (1/9)
sweep:  11%|████▊                                      | 1/9 [00:02<00:17,  2.23s/cell, rho=0.50, alpha=0.10, done=1/9][2026-04-02 01:57:37,319][__main__][INFO] -   alpha=  1.00 | E[Q_total]=    1.60 | OK  (2/9)
sweep:  22%|█████████▌                                 | 2/9 [00:03<00:09,  1.39s/cell, rho=0.50, alpha=1.00, done=2/9][2026-04-02 01:57:38,054][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    1.42 | OK  (3/9)
sweep:  33%|██████████████▎                            | 3/9 [00:03<00:06,  1.09s/cell, rho=0.50, alpha=5.00, done=3/9][2026-04-02 01:57:38,056][__main__][INFO] -
------------------------------------------------------------
  rho = 0.80  (lam = 2.0000)
------------------------------------------------------------
                                                                                                                       [2026-04-02 01:57:40,598][__main__][INFO] -   alpha=  0.10 | E[Q_total]=    6.46 | OK  (4/9)
sweep:  44%|███████████████████                        | 4/9 [00:06<00:08,  1.67s/cell, rho=0.80, alpha=0.10, done=4/9][2026-04-02 01:57:41,821][__main__][INFO] -   alpha=  1.00 | E[Q_total]=    5.13 | NONSTATIONARY  (5/9)
sweep:  56%|███████████████████████▉                   | 5/9 [00:07<00:06,  1.51s/cell, rho=0.80, alpha=1.00, done=5/9][2026-04-02 01:57:42,759][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    4.75 | OK  (6/9)
sweep:  67%|████████████████████████████▋              | 6/9 [00:08<00:03,  1.31s/cell, rho=0.80, alpha=5.00, done=6/9][2026-04-02 01:57:42,761][__main__][INFO] -
------------------------------------------------------------
  rho = 0.90  (lam = 2.2500)
------------------------------------------------------------
                                                                                                                       [2026-04-02 01:57:45,313][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   11.82 | OK  (7/9)
sweep:  78%|█████████████████████████████████▍         | 7/9 [00:11<00:03,  1.72s/cell, rho=0.90, alpha=0.10, done=7/9][2026-04-02 01:57:46,177][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   10.41 | NONSTATIONARY  (8/9)
sweep:  89%|██████████████████████████████████████▏    | 8/9 [00:11<00:01,  1.45s/cell, rho=0.90, alpha=1.00, done=8/9][2026-04-02 01:57:47,040][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    9.88 | NONSTATIONARY  (9/9)
sweep: 100%|███████████████████████████████████████████| 9/9 [00:12<00:00,  1.42s/cell, rho=0.90, alpha=5.00, done=9/9]
[2026-04-02 01:57:47,480][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:57:47,726][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:57:48,393][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:57:48,500][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:57:48,734][__main__][INFO] -
Saved plot: outputs\small\sweep\run_20260402_015731\alpha_sweep.png, outputs\small\sweep\run_20260402_015731\alpha_sweep.pdf
[2026-04-02 01:57:48,888][__main__][INFO] -
Summary: 3/9 configurations non-stationary.
wandb:
wandb: Run history:
wandb:                   alpha ▁▂█▁▂█▁▂█
wandb:                     lam ▁▁▁▆▆▆███
wandb:            mean_q_total ▁▁▁▄▃▃█▇▇
wandb:        num_replications ▁▁▁▁▁▁▁▁▁
wandb:                     rho ▁▁▁▆▆▆███
wandb:       stationarity_rate ████▁██▁▁
wandb:  stationarity_threshold ▁▁▁▁▁▁▁▁▁
wandb: stationary_replications ████▁██▁▁
wandb:
wandb: Run summary:
wandb:                   alpha 5
wandb:                 backend JAX
wandb:           is_stationary False
wandb:                     lam 2.25
wandb:            mean_q_total 9.87866
wandb:        num_replications 10
wandb:                     rho 0.9
wandb:       stationarity_rate 0.9
wandb:  stationarity_threshold 1
wandb: stationary_replications 9
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync outputs\small\sweep\run_20260402_015731\wandb\offline-run-20260402_015731-l4qt2xr3
wandb: Find logs at: outputs\small\sweep\run_20260402_015731\wandb\offline-run-20260402_015731-l4qt2xr3\logs

[Experiment 'sweep' Finished]
  Status: completed
  End Time: 2026-04-02 01:57:49+05:45
  Elapsed Duration: 24.420s
  -> [sweep] Ended at 2026-04-02 01:57:49+05:45
  -> [sweep] Status: completed
  -> [sweep] Elapsed: 28.896s
4/12 sweep:  33%|████████████████▋                                 | 4/12 [01:37<03:13, 24.21s/experiment, alias=sweep]
[4/11] Running Scaling Stress Tests (Phase 1c)...
5/12 stress:  33%|████████████████▎                                | 4/12 [01:37<03:13, 24.21s/experiment, alias=sweep][handoff] Launching stress (5/12)
  -> [stress] Started at 2026-04-02 01:57:49+05:45
==========================================================
 Starting Experiment: stress
 Start Time: 2026-04-02 01:57:53+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=stress ++jax.enabled=True
==========================================================
[2026-04-02 01:57:58,675][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-04-02 01:57:58,697:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:57:58,697][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:57:58,698][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-04-02 01:57:58,715][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\stress\run_20260402_015758
[2026-04-02 01:57:58,715][__main__][INFO] - ============================================================
[2026-04-02 01:57:58,719][__main__][INFO] -   GibbsQ Stress Test (JAX Accelerator Active)
[2026-04-02 01:57:58,720][__main__][INFO] - ============================================================
stress:   0%|                                                                                 | 0/3 [00:00<?, ?stage/s][2026-04-02 01:57:58,724][__main__][INFO] -
[TEST 1] Massive-N Scaling Analysis
                                                                                                                       [2026-04-02 01:57:58,829][__main__][INFO] -   Simulating N=4 experts (rho=0.8)...                  | 0/3 [00:00<?, ?N/s]
                                                                                                                       [2026-04-02 01:58:00,799][__main__][INFO] -     -> Average Gini Imbalance: 0.0244
                                                                                                                       [2026-04-02 01:58:00,872][__main__][INFO] -   Simulating N=8 experts (rho=0.8)...     | 1/3 [00:02<00:04,  2.08s/N, N=4]
                                                                                                                       [2026-04-02 01:58:03,539][__main__][INFO] -     -> Average Gini Imbalance: 0.0288
                                                                                                                       [2026-04-02 01:58:03,594][__main__][INFO] -   Simulating N=16 experts (rho=0.8)...    | 2/3 [00:04<00:02,  2.47s/N, N=8]
                                                                                                                       [2026-04-02 01:58:06,207][__main__][INFO] -     -> Average Gini Imbalance: 0.0285
stress:  33%|████████████████████████▎                                                | 1/3 [00:07<00:14,  7.49s/stage][2026-04-02 01:58:06,210][__main__][INFO] -
[TEST 2] Critical Load Analysis (rho up to 0.95)
                                                                                                                       [2026-04-02 01:58:06,262][__main__][INFO] -   Simulating rho=0.900 (T=10000.0)...                | 0/2 [00:00<?, ?rho/s]
[2026-04-02 01:58:24,202][__main__][INFO] -     -> Gelman-Rubin R-hat across replicas (post MSER-5 burn-in): 1.0006
                                                                                                                       [2026-04-02 01:58:24,275][__main__][INFO] -     -> Avg E[Q_total]: 21.86 | Stationarity: 10/10
                                                                                                                       [2026-04-02 01:58:24,277][__main__][INFO] -   Simulating rho=0.950 (T=19999.999999999978)...0:18, 18.01s/rho, rho=0.900]
[2026-04-02 01:59:10,311][__main__][INFO] -     -> Gelman-Rubin R-hat across replicas (post MSER-5 burn-in): 1.0007
                                                                                                                       [2026-04-02 01:59:10,392][__main__][INFO] -     -> Avg E[Q_total]: 33.37 | Stationarity: 8/10
stress:  67%|████████████████████████████████████████████████▋                        | 2/3 [01:11<00:40, 40.84s/stage][2026-04-02 01:59:10,396][__main__][INFO] -
[TEST 3] Extreme Heterogeneity Resilience (100x Speed Gap)
[2026-04-02 01:59:10,410][__main__][INFO] -   Simulating heterogenous setup: mu=[10.   0.1  0.1  0.1]
                                                                                                                       [2026-04-02 01:59:12,364][__main__][INFO] -     -> Mean Queue per Expert: [1.07353308 0.         0.         0.        ]
[2026-04-02 01:59:12,365][__main__][INFO] -     -> Gini: 0.7500
stress: 100%|█████████████████████████████████████████████████████████████████████████| 3/3 [01:13<00:00, 24.55s/stage]
[2026-04-02 01:59:13,387][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:59:13,518][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:59:13,891][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:59:14,043][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:59:14,468][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:59:14,642][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:59:16,077][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:59:16,164][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:59:16,313][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:59:16,434][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 01:59:16,674][__main__][INFO] - Stress dashboard saved to outputs\small\stress\run_20260402_015758\stress_dashboard.png, outputs\small\stress\run_20260402_015758\stress_dashboard.pdf
[2026-04-02 01:59:16,674][__main__][INFO] -
Stress test complete.

[Experiment 'stress' Finished]
  Status: completed
  End Time: 2026-04-02 01:59:17+05:45
  Elapsed Duration: 83.432s
  -> [stress] Ended at 2026-04-02 01:59:17+05:45
  -> [stress] Status: completed
  -> [stress] Elapsed: 87.793s
5/12 stress:  42%|████████████████████                            | 5/12 [03:05<05:29, 47.14s/experiment, alias=stress]
[5/11] Running Platinum BC Pretraining (Phase 2a)...
6/12 bc_train:  42%|███████████████████▏                          | 5/12 [03:05<05:29, 47.14s/experiment, alias=stress][handoff] Launching bc_train (6/12)
  -> [bc_train] Started at 2026-04-02 01:59:17+05:45
==========================================================
 Starting Experiment: bc_train
 Start Time: 2026-04-02 01:59:22+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=bc_train
==========================================================
[2026-04-02 01:59:28,452][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\bc_train\run_20260402_015928
INFO:2026-04-02 01:59:28,484:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:59:28,484][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:59:29,879][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-04-02 01:59:31,736][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
bc_train:   0%|▎                                                                     | 1/201 [00:01<03:25,  1.03s/step][2026-04-02 01:59:32,771][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6862 | Acc: 22.50%
bc_train:  44%|███████████████████▍                        | 89/201 [00:01<00:01, 72.63step/s, loss=0.5635, acc=93.94%][2026-04-02 01:59:33,326][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5630 | Acc: 96.56%
bc_train:  92%|██████████████████████████████████████▋   | 185/201 [00:02<00:00, 116.68step/s, loss=0.5585, acc=99.28%][2026-04-02 01:59:33,855][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5579 | Acc: 99.28%
bc_train: 100%|███████████████████████████████████████████| 201/201 [00:02<00:00, 95.08step/s, loss=0.5579, acc=99.28%]
[2026-04-02 01:59:33,863][__main__][INFO] -
[DONE] Platinum BC Weights saved to outputs\small\bc_train\run_20260402_015928\n_gibbsq_platinum_bc_weights.eqx
[2026-04-02 01:59:33,864][__main__][INFO] - [Metadata] BC warm-start compatibility metadata saved to outputs\small\bc_train\run_20260402_015928\n_gibbsq_platinum_bc_weights.eqx.bc_metadata.json
[2026-04-02 01:59:33,866][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_bc_weights.txt at outputs\small\latest_bc_weights.txt

[Experiment 'bc_train' Finished]
  Status: completed
  End Time: 2026-04-02 01:59:34+05:45
  Elapsed Duration: 11.674s
  -> [bc_train] Ended at 2026-04-02 01:59:34+05:45
  -> [bc_train] Status: completed
  -> [bc_train] Elapsed: 17.077s
6/12 bc_train:  50%|██████████████████████                      | 6/12 [03:22<03:41, 36.92s/experiment, alias=bc_train]
[6/11] Running REINFORCE SSA Training (Phase 2b)...
7/12 reinforce_train:  50%|██████████████████▌                  | 6/12 [03:22<03:41, 36.92s/experiment, alias=bc_train][handoff] Launching reinforce_train (7/12)
  -> [reinforce_train] Started at 2026-04-02 01:59:34+05:45
==========================================================
 Starting Experiment: reinforce_train
 Start Time: 2026-04-02 01:59:38+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=reinforce_train
==========================================================
[2026-04-02 01:59:46,166][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\reinforce_train\run_20260402_015946
INFO:2026-04-02 01:59:46,195:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 01:59:46,195][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
reinforce: setup:   0%|                                                                       | 0/3 [00:00<?, ?stage/s][2026-04-02 01:59:48,869][__main__][INFO] -   JSQ Mean Queue (Target): 1.0803
[2026-04-02 01:59:48,870][__main__][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-04-02 01:59:48,888][__main__][INFO] - Reusing BC warm-start actor weights from C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\bc_train\run_20260402_015928\n_gibbsq_platinum_bc_weights.eqx
[2026-04-02 01:59:48,892][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-04-02 01:59:50,587][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
                                                                                                                       [2026-04-02 01:59:51,199][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 1045545.312500:00<02:01,  1.64step/s]
                                                                                                                       [2026-04-02 01:59:51,803][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 644921.56256step/s, loss=667455.5000]
                                                                                                                       [2026-04-02 01:59:52,223][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 535013.93753step/s, loss=543972.3125]
bc_value: 100%|█████████████████████████████████████████████████| 201/201 [00:01<00:00, 122.94step/s, loss=535013.9375]
reinforce: setup:  67%|██████████████████████████████████████████                     | 2/3 [00:03<00:01,  1.83s/stage][2026-04-02 01:59:52,230][__main__][INFO] - ============================================================
[2026-04-02 01:59:52,231][__main__][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-04-02 01:59:52,231][__main__][INFO] - ============================================================
[2026-04-02 01:59:52,231][__main__][INFO] -   Epochs: 5, Batch size: 4
[2026-04-02 01:59:52,231][__main__][INFO] -   Simulation time: 1000.0
[2026-04-02 01:59:52,232][__main__][INFO] - ------------------------------------------------------------
reinforce_train:   0%|                                                                        | 0/5 [00:00<?, ?epoch/s][2026-04-02 02:00:07,541][__main__][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: -0.0044 | mean_logp: -0.5819 | corr: -0.0414
[2026-04-02 02:00:07,542][__main__][INFO] -     [Grad Check] P-Grad Norm: 0.1596 | V-Grad Norm: 43368.4102
reinforce_train:  20%|███████▊                               | 1/5 [01:05<04:22, 65.71s/epoch, queue=2.064, pi=-112.9%]
[2026-04-02 02:01:02,357][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\small\latest_reinforce_weights.txt
[2026-04-02 02:01:02,357][__main__][INFO] - -------------------------------------------------------
[2026-04-02 02:01:02,358][__main__][INFO] - -------------------------------------------------------
[2026-04-02 02:01:02,358][__main__][INFO] - Running Final Deterministic Evaluation (N=3)...
[2026-04-02 02:01:04,333][__main__][INFO] - Stage profile written to outputs\small\reinforce_train\run_20260402_015946\reinforce_stage_profile.json
[2026-04-02 02:01:04,333][__main__][INFO] - Deterministic Policy Score: 93.87% ± 17.32%
[2026-04-02 02:01:04,334][__main__][INFO] - JSQ Target: 100.0% | Random Floor: 0.0% (Performance Index Scale)
[2026-04-02 02:01:04,334][__main__][INFO] - -------------------------------------------------------
[2026-04-02 02:01:04,334][__main__][INFO] - Training Complete! Final Loss: -0.0014
[2026-04-02 02:01:04,335][__main__][INFO] - Final Base-Regime Index Proxy: -112.87
[2026-04-02 02:01:04,335][__main__][INFO] - Policy weights: outputs\small\reinforce_train\run_20260402_015946\n_gibbsq_reinforce_weights.eqx
[2026-04-02 02:01:04,335][__main__][INFO] - Value weights: outputs\small\reinforce_train\run_20260402_015946\value_network_weights.eqx

[Experiment 'reinforce_train' Finished]
  Status: completed
  End Time: 2026-04-02 02:01:06+05:45
  Elapsed Duration: 87.799s
  -> [reinforce_train] Ended at 2026-04-02 02:01:07+05:45
  -> [reinforce_train] Status: completed
  -> [reinforce_train] Elapsed: 92.349s
7/12 reinforce_train:  58%|█████████████████▌            | 7/12 [04:54<04:35, 55.04s/experiment, alias=reinforce_train]
[7/11] Running Corrected Policy Evaluation Benchmark (Phase 3a)...
8/12 policy:  58%|██████████████████████▊                | 7/12 [04:54<04:35, 55.04s/experiment, alias=reinforce_train][handoff] Launching policy (8/12)
  -> [policy] Started at 2026-04-02 02:01:07+05:45
==========================================================
 Starting Experiment: policy
 Start Time: 2026-04-02 02:01:11+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=policy
==========================================================
[2026-04-02 02:01:16,651][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\policy\run_20260402_020116
[2026-04-02 02:01:16,652][gibbsq.utils.logging][INFO] - [Logging] WandB offline mode.
wandb: Tracking run with wandb version 0.23.1
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in outputs\small\policy\run_20260402_020116\wandb\offline-run-20260402_020117-q0hxf0qk
[2026-04-02 02:01:19,861][gibbsq.utils.logging][INFO] - [Logging] WandB Run Linked: run_20260402_020116 (offline)
[2026-04-02 02:01:19,862][__main__][INFO] - ============================================================
[2026-04-02 02:01:19,863][__main__][INFO] -   Corrected Policy Comparison
[2026-04-02 02:01:19,864][__main__][INFO] - ============================================================
[2026-04-02 02:01:19,865][__main__][INFO] - System: N=2, lambda=1.0000, Lambda=2.5000, rho=0.4000
[2026-04-02 02:01:19,866][__main__][INFO] - ------------------------------------------------------------
policy: tiers:   0%|                                                                         | 0/7 [00:00<?, ?policy/s][2026-04-02 02:01:19,871][__main__][INFO] - Evaluating Tier 2: JSQ (Min Queue)...
                                                                                                                       [2026-04-02 02:01:19,875][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:01:20,478][gibbsq.engines.numpy_engine][INFO] -   -> 5,020 arrivals, 5,020 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:20,482][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:05,  1.65rep/s]
[2026-04-02 02:01:21,070][gibbsq.engines.numpy_engine][INFO] -   -> 4,910 arrivals, 4,910 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:21,075][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:04,  1.68rep/s]
[2026-04-02 02:01:21,703][gibbsq.engines.numpy_engine][INFO] -   -> 5,118 arrivals, 5,117 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:21,707][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:01<00:04,  1.63rep/s]
[2026-04-02 02:01:22,312][gibbsq.engines.numpy_engine][INFO] -   -> 4,966 arrivals, 4,966 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:22,315][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:02<00:03,  1.64rep/s]
[2026-04-02 02:01:22,919][gibbsq.engines.numpy_engine][INFO] -   -> 5,093 arrivals, 5,091 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:22,923][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:03<00:03,  1.64rep/s]
[2026-04-02 02:01:23,532][gibbsq.engines.numpy_engine][INFO] -   -> 4,775 arrivals, 4,774 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:23,536][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:03<00:02,  1.64rep/s]
[2026-04-02 02:01:24,149][gibbsq.engines.numpy_engine][INFO] -   -> 4,899 arrivals, 4,898 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:24,152][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:04<00:01,  1.63rep/s]
[2026-04-02 02:01:24,768][gibbsq.engines.numpy_engine][INFO] -   -> 5,058 arrivals, 5,056 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:24,772][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:04<00:01,  1.63rep/s]
[2026-04-02 02:01:25,583][gibbsq.engines.numpy_engine][INFO] -   -> 5,011 arrivals, 5,010 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:25,586][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:05<00:00,  1.48rep/s]
[2026-04-02 02:01:26,281][gibbsq.engines.numpy_engine][INFO] -   -> 5,084 arrivals, 5,084 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:26,291][__main__][INFO] -   E[Q_total] = 1.0196 ± 0.0113
policy: tiers:  14%|█████████▎                                                       | 1/7 [00:06<00:38,  6.43s/policy][2026-04-02 02:01:26,297][__main__][INFO] - Evaluating Tier 2: JSSQ (Min Sojourn)...
                                                                                                                       [2026-04-02 02:01:26,301][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:01:26,971][gibbsq.engines.numpy_engine][INFO] -   -> 5,068 arrivals, 5,066 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:26,975][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:06,  1.49rep/s]
[2026-04-02 02:01:27,641][gibbsq.engines.numpy_engine][INFO] -   -> 4,960 arrivals, 4,957 departures, final Q_total = 3
                                                                                                                       [2026-04-02 02:01:27,645][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:05,  1.49rep/s]
[2026-04-02 02:01:28,276][gibbsq.engines.numpy_engine][INFO] -   -> 5,096 arrivals, 5,096 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:28,280][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:01<00:04,  1.53rep/s]
[2026-04-02 02:01:28,945][gibbsq.engines.numpy_engine][INFO] -   -> 4,998 arrivals, 4,998 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:28,950][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:02<00:03,  1.51rep/s]
[2026-04-02 02:01:29,799][gibbsq.engines.numpy_engine][INFO] -   -> 5,095 arrivals, 5,095 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:29,803][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:03<00:03,  1.37rep/s]
[2026-04-02 02:01:30,481][gibbsq.engines.numpy_engine][INFO] -   -> 4,804 arrivals, 4,804 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:30,484][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:04<00:02,  1.40rep/s]
[2026-04-02 02:01:31,131][gibbsq.engines.numpy_engine][INFO] -   -> 4,916 arrivals, 4,916 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:31,135][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:04<00:02,  1.44rep/s]
[2026-04-02 02:01:31,802][gibbsq.engines.numpy_engine][INFO] -   -> 5,056 arrivals, 5,056 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:31,805][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:05<00:01,  1.46rep/s]
[2026-04-02 02:01:32,458][gibbsq.engines.numpy_engine][INFO] -   -> 5,044 arrivals, 5,043 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:32,461][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:06<00:00,  1.48rep/s]
[2026-04-02 02:01:33,118][gibbsq.engines.numpy_engine][INFO] -   -> 5,074 arrivals, 5,074 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:33,128][__main__][INFO] -   E[Q_total] = 0.9599 ± 0.0111
policy: tiers:  29%|██████████████████▌                                              | 2/7 [00:13<00:33,  6.66s/policy][2026-04-02 02:01:33,129][__main__][INFO] - Evaluating Tier 3: UAS (alpha=1.0)...
                                                                                                                       [2026-04-02 02:01:33,133][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:01:34,047][gibbsq.engines.numpy_engine][INFO] -   -> 5,053 arrivals, 5,051 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:34,051][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:08,  1.09rep/s]
[2026-04-02 02:01:34,792][gibbsq.engines.numpy_engine][INFO] -   -> 4,947 arrivals, 4,945 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:34,795][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:06,  1.23rep/s]
[2026-04-02 02:01:35,473][gibbsq.engines.numpy_engine][INFO] -   -> 5,113 arrivals, 5,111 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:35,477][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:02<00:05,  1.33rep/s]
[2026-04-02 02:01:36,158][gibbsq.engines.numpy_engine][INFO] -   -> 4,947 arrivals, 4,946 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:36,162][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:03<00:04,  1.38rep/s]
[2026-04-02 02:01:36,838][gibbsq.engines.numpy_engine][INFO] -   -> 5,058 arrivals, 5,058 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:36,842][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:03<00:03,  1.41rep/s]
[2026-04-02 02:01:37,520][gibbsq.engines.numpy_engine][INFO] -   -> 4,798 arrivals, 4,793 departures, final Q_total = 5
                                                                                                                       [2026-04-02 02:01:37,523][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:04<00:02,  1.43rep/s]
[2026-04-02 02:01:38,169][gibbsq.engines.numpy_engine][INFO] -   -> 4,920 arrivals, 4,919 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:38,172][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:05<00:02,  1.46rep/s]
[2026-04-02 02:01:39,015][gibbsq.engines.numpy_engine][INFO] -   -> 5,061 arrivals, 5,061 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:39,020][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:05<00:01,  1.36rep/s]
[2026-04-02 02:01:39,812][gibbsq.engines.numpy_engine][INFO] -   -> 5,047 arrivals, 5,047 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:39,817][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:06<00:00,  1.33rep/s]
[2026-04-02 02:01:40,481][gibbsq.engines.numpy_engine][INFO] -   -> 5,064 arrivals, 5,061 departures, final Q_total = 3
                                                                                                                       [2026-04-02 02:01:40,489][__main__][INFO] -   E[Q_total] = 1.0985 ± 0.0136
policy: tiers:  43%|███████████████████████████▊                                     | 3/7 [00:20<00:27,  6.99s/policy][2026-04-02 02:01:40,496][__main__][INFO] - Evaluating Tier 3: UAS (alpha=10.0)...
                                                                                                                       [2026-04-02 02:01:40,499][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:01:41,180][gibbsq.engines.numpy_engine][INFO] -   -> 5,065 arrivals, 5,065 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:41,185][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:06,  1.47rep/s]
[2026-04-02 02:01:41,841][gibbsq.engines.numpy_engine][INFO] -   -> 4,956 arrivals, 4,954 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:41,845][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:05,  1.49rep/s]
[2026-04-02 02:01:42,489][gibbsq.engines.numpy_engine][INFO] -   -> 5,090 arrivals, 5,089 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:42,491][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:01<00:04,  1.52rep/s]
[2026-04-02 02:01:43,394][gibbsq.engines.numpy_engine][INFO] -   -> 4,993 arrivals, 4,993 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:43,398][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:02<00:04,  1.32rep/s]
[2026-04-02 02:01:44,114][gibbsq.engines.numpy_engine][INFO] -   -> 5,094 arrivals, 5,094 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:44,118][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:03<00:03,  1.35rep/s]
[2026-04-02 02:01:44,770][gibbsq.engines.numpy_engine][INFO] -   -> 4,808 arrivals, 4,806 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:44,773][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:04<00:02,  1.40rep/s]
[2026-04-02 02:01:45,505][gibbsq.engines.numpy_engine][INFO] -   -> 4,913 arrivals, 4,912 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:45,510][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:05<00:02,  1.39rep/s]
[2026-04-02 02:01:46,223][gibbsq.engines.numpy_engine][INFO] -   -> 5,060 arrivals, 5,060 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:46,227][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:05<00:01,  1.39rep/s]
[2026-04-02 02:01:47,008][gibbsq.engines.numpy_engine][INFO] -   -> 5,041 arrivals, 5,041 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:47,012][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:06<00:00,  1.35rep/s]
[2026-04-02 02:01:47,866][gibbsq.engines.numpy_engine][INFO] -   -> 5,073 arrivals, 5,073 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:47,875][__main__][INFO] -   E[Q_total] = 0.9700 ± 0.0117
policy: tiers:  57%|█████████████████████████████████████▏                           | 4/7 [00:28<00:21,  7.14s/policy][2026-04-02 02:01:47,877][__main__][INFO] - Evaluating Tier 3: UAS (alpha=5.0)...
                                                                                                                       [2026-04-02 02:01:47,880][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:01:48,543][gibbsq.engines.numpy_engine][INFO] -   -> 5,045 arrivals, 5,043 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:48,547][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:05,  1.51rep/s]
[2026-04-02 02:01:49,224][gibbsq.engines.numpy_engine][INFO] -   -> 4,966 arrivals, 4,962 departures, final Q_total = 4
                                                                                                                       [2026-04-02 02:01:49,226][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:05,  1.48rep/s]
[2026-04-02 02:01:49,901][gibbsq.engines.numpy_engine][INFO] -   -> 5,075 arrivals, 5,075 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:49,905][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:02<00:04,  1.48rep/s]
[2026-04-02 02:01:50,582][gibbsq.engines.numpy_engine][INFO] -   -> 4,993 arrivals, 4,993 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:50,586][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:02<00:04,  1.48rep/s]
[2026-04-02 02:01:51,434][gibbsq.engines.numpy_engine][INFO] -   -> 5,087 arrivals, 5,084 departures, final Q_total = 3
                                                                                                                       [2026-04-02 02:01:51,437][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:03<00:03,  1.35rep/s]
[2026-04-02 02:01:52,175][gibbsq.engines.numpy_engine][INFO] -   -> 4,810 arrivals, 4,809 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:52,179][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:04<00:02,  1.35rep/s]
[2026-04-02 02:01:52,841][gibbsq.engines.numpy_engine][INFO] -   -> 4,916 arrivals, 4,914 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:52,845][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:04<00:02,  1.40rep/s]
[2026-04-02 02:01:53,505][gibbsq.engines.numpy_engine][INFO] -   -> 5,053 arrivals, 5,052 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:53,510][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:05<00:01,  1.43rep/s]
[2026-04-02 02:01:54,205][gibbsq.engines.numpy_engine][INFO] -   -> 5,047 arrivals, 5,047 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:54,208][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:06<00:00,  1.43rep/s]
[2026-04-02 02:01:54,882][gibbsq.engines.numpy_engine][INFO] -   -> 5,061 arrivals, 5,059 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:54,892][__main__][INFO] -   E[Q_total] = 0.9971 ± 0.0106
policy: tiers:  71%|██████████████████████████████████████████████▍                  | 5/7 [00:35<00:14,  7.10s/policy][2026-04-02 02:01:54,893][__main__][INFO] - Evaluating Tier 4: Proportional (mu/Lambda)...
                                                                                                                       [2026-04-02 02:01:54,897][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:01:55,579][gibbsq.engines.numpy_engine][INFO] -   -> 5,026 arrivals, 5,026 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:55,584][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:06,  1.46rep/s]
[2026-04-02 02:01:56,201][gibbsq.engines.numpy_engine][INFO] -   -> 4,932 arrivals, 4,928 departures, final Q_total = 4
                                                                                                                       [2026-04-02 02:01:56,206][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:05,  1.54rep/s]
[2026-04-02 02:01:56,744][gibbsq.engines.numpy_engine][INFO] -   -> 5,136 arrivals, 5,136 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:01:56,747][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:01<00:04,  1.67rep/s]
[2026-04-02 02:01:57,249][gibbsq.engines.numpy_engine][INFO] -   -> 4,926 arrivals, 4,925 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:57,253][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:02<00:03,  1.78rep/s]
[2026-04-02 02:01:57,751][gibbsq.engines.numpy_engine][INFO] -   -> 5,054 arrivals, 5,053 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:01:57,755][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:02<00:02,  1.85rep/s]
[2026-04-02 02:01:58,248][gibbsq.engines.numpy_engine][INFO] -   -> 4,768 arrivals, 4,767 departures, final Q_total = 1
[2026-04-02 02:01:58,249][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 02:01:58,755][gibbsq.engines.numpy_engine][INFO] -   -> 4,908 arrivals, 4,906 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:01:58,758][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:03<00:01,  1.92rep/s]
[2026-04-02 02:01:59,244][gibbsq.engines.numpy_engine][INFO] -   -> 5,038 arrivals, 5,038 departures, final Q_total = 0
[2026-04-02 02:01:59,246][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-04-02 02:01:59,758][gibbsq.engines.numpy_engine][INFO] -   -> 5,031 arrivals, 5,026 departures, final Q_total = 5
                                                                                                                       [2026-04-02 02:01:59,761][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:04<00:00,  1.95rep/s]
[2026-04-02 02:02:00,359][gibbsq.engines.numpy_engine][INFO] -   -> 5,085 arrivals, 5,084 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:02:00,369][__main__][INFO] -   E[Q_total] = 1.3255 ± 0.0212
policy: tiers:  86%|███████████████████████████████████████████████████████▋         | 6/7 [00:40<00:06,  6.55s/policy][2026-04-02 02:02:00,373][__main__][INFO] - Evaluating Tier 4: Uniform (1/N)...
                                                                                                                       [2026-04-02 02:02:00,377][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:02:01,149][gibbsq.engines.numpy_engine][INFO] -   -> 5,002 arrivals, 5,002 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:02:01,153][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:06,  1.29rep/s]
[2026-04-02 02:02:01,679][gibbsq.engines.numpy_engine][INFO] -   -> 4,902 arrivals, 4,901 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:02:01,683][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:05,  1.59rep/s]
[2026-04-02 02:02:02,249][gibbsq.engines.numpy_engine][INFO] -   -> 5,144 arrivals, 5,143 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:02:02,253][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:01<00:04,  1.66rep/s]
[2026-04-02 02:02:02,809][gibbsq.engines.numpy_engine][INFO] -   -> 4,942 arrivals, 4,941 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:02:02,813][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:02<00:03,  1.71rep/s]
[2026-04-02 02:02:03,374][gibbsq.engines.numpy_engine][INFO] -   -> 5,058 arrivals, 5,058 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:02:03,378][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:02<00:02,  1.73rep/s]
[2026-04-02 02:02:03,878][gibbsq.engines.numpy_engine][INFO] -   -> 4,783 arrivals, 4,779 departures, final Q_total = 4
                                                                                                                       [2026-04-02 02:02:03,881][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:03<00:02,  1.81rep/s]
[2026-04-02 02:02:04,402][gibbsq.engines.numpy_engine][INFO] -   -> 4,932 arrivals, 4,929 departures, final Q_total = 3
                                                                                                                       [2026-04-02 02:02:04,405][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:04<00:01,  1.84rep/s]
[2026-04-02 02:02:04,940][gibbsq.engines.numpy_engine][INFO] -   -> 5,025 arrivals, 5,025 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:02:04,943][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:04<00:01,  1.85rep/s]
[2026-04-02 02:02:05,485][gibbsq.engines.numpy_engine][INFO] -   -> 5,004 arrivals, 5,004 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:02:05,489][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:05<00:00,  1.84rep/s]
[2026-04-02 02:02:06,052][gibbsq.engines.numpy_engine][INFO] -   -> 5,084 arrivals, 5,082 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:02:06,059][__main__][INFO] -   E[Q_total] = 1.4731 ± 0.0243
policy: tiers: 100%|█████████████████████████████████████████████████████████████████| 7/7 [00:46<00:00,  6.60s/policy]
[2026-04-02 02:02:06,064][__main__][INFO] -
Evaluating Tier 5: N-GibbsQ (REINFORCE trained)...
[2026-04-02 02:02:06,065][__main__][INFO] - Using neural weights from C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\reinforce_train\run_20260402_015946\n_gibbsq_reinforce_weights.eqx
INFO:2026-04-02 02:02:06,088:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 02:02:06,088][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 02:02:07,281][__main__][INFO] - Evaluating N-GibbsQ (deterministic)...
policy eval (DeterministicNeuralPolicy):   0%|                                                 | 0/10 [00:00<?, ?rep/s][2026-04-02 02:02:07,283][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)
[2026-04-02 02:02:11,179][gibbsq.engines.numpy_engine][INFO] -   -> 5,063 arrivals, 5,063 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  10%|████                                     | 1/10 [00:03<00:35,  3.90s/rep][2026-04-02 02:02:11,182][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-04-02 02:02:14,772][gibbsq.engines.numpy_engine][INFO] -   -> 4,954 arrivals, 4,954 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  20%|████████▏                                | 2/10 [00:07<00:29,  3.72s/rep][2026-04-02 02:02:14,775][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-04-02 02:02:18,616][gibbsq.engines.numpy_engine][INFO] -   -> 5,083 arrivals, 5,083 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  30%|████████████▎                            | 3/10 [00:11<00:26,  3.78s/rep][2026-04-02 02:02:18,618][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-04-02 02:02:22,202][gibbsq.engines.numpy_engine][INFO] -   -> 4,966 arrivals, 4,966 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  40%|████████████████▍                        | 4/10 [00:14<00:22,  3.70s/rep][2026-04-02 02:02:22,204][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-04-02 02:02:26,125][gibbsq.engines.numpy_engine][INFO] -   -> 5,054 arrivals, 5,053 departures, final Q_total = 1
policy eval (DeterministicNeuralPolicy):  50%|████████████████████▌                    | 5/10 [00:18<00:18,  3.78s/rep][2026-04-02 02:02:26,127][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-04-02 02:02:29,828][gibbsq.engines.numpy_engine][INFO] -   -> 4,810 arrivals, 4,810 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  60%|████████████████████████▌                | 6/10 [00:22<00:15,  3.75s/rep][2026-04-02 02:02:29,830][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 02:02:33,665][gibbsq.engines.numpy_engine][INFO] -   -> 4,916 arrivals, 4,916 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  70%|████████████████████████████▋            | 7/10 [00:26<00:11,  3.78s/rep][2026-04-02 02:02:33,667][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-04-02 02:02:37,328][gibbsq.engines.numpy_engine][INFO] -   -> 5,051 arrivals, 5,051 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  80%|████████████████████████████████▊        | 8/10 [00:30<00:07,  3.74s/rep][2026-04-02 02:02:37,330][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-04-02 02:02:41,220][gibbsq.engines.numpy_engine][INFO] -   -> 5,047 arrivals, 5,047 departures, final Q_total = 0
policy eval (DeterministicNeuralPolicy):  90%|████████████████████████████████████▉    | 9/10 [00:33<00:03,  3.79s/rep][2026-04-02 02:02:41,222][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-04-02 02:02:45,164][gibbsq.engines.numpy_engine][INFO] -   -> 5,066 arrivals, 5,061 departures, final Q_total = 5
[2026-04-02 02:02:45,172][__main__][INFO] -   E[Q_total] = 1.0710 ± 0.0118
[2026-04-02 02:02:45,172][__main__][INFO] -
============================================================
[2026-04-02 02:02:45,173][__main__][INFO] -   Parity Analysis (Corrected Criteria)
[2026-04-02 02:02:45,174][__main__][INFO] - ============================================================
[2026-04-02 02:02:45,175][__main__][INFO] - N-GibbsQ (Platinum/Greedy): E[Q] = 1.0710
[2026-04-02 02:02:45,176][__main__][INFO] - Reference thresholds:
[2026-04-02 02:02:45,176][__main__][INFO] -   JSSQ (Tier 2): E[Q] = 0.9599
[2026-04-02 02:02:45,177][__main__][INFO] -   UAS (Tier 3): E[Q] = 1.0985
[2026-04-02 02:02:45,177][__main__][INFO] -   Proportional (Tier 4): E[Q] = 1.3255
[2026-04-02 02:02:45,178][__main__][INFO] - Reference statistical bounds (95% CI):
[2026-04-02 02:02:45,179][__main__][INFO] -   JSSQ (Tier 2): E[Q] = 0.9599 ± 0.0111
[2026-04-02 02:02:45,179][__main__][INFO] -   UAS (Tier 3): E[Q] = 1.0985 ± 0.0136
[2026-04-02 02:02:45,180][__main__][INFO] -   Proportional (Tier 4): E[Q] = 1.3255 ± 0.0212
[2026-04-02 02:02:45,181][__main__][INFO] - PARITY RESULT: SILVER [OK] (Statistically matches empirical UAS baseline)
[2026-04-02 02:02:46,523][__main__][INFO] - Comparison plot saved to outputs\small\policy\run_20260402_020116\corrected_policy_comparison.png, outputs\small\policy\run_20260402_020116\corrected_policy_comparison.pdf
wandb: You can sync this run to the cloud by running:
wandb: wandb sync outputs\small\policy\run_20260402_020116\wandb\offline-run-20260402_020117-q0hxf0qk
wandb: Find logs at: outputs\small\policy\run_20260402_020116\wandb\offline-run-20260402_020117-q0hxf0qk\logs

[Experiment 'policy' Finished]
  Status: completed
  End Time: 2026-04-02 02:02:47+05:45
  Elapsed Duration: 95.709s
  -> [policy] Ended at 2026-04-02 02:02:47+05:45
  -> [policy] Status: completed
  -> [policy] Elapsed: 100.262s
8/12 policy:  67%|████████████████████████████████                | 8/12 [06:35<04:37, 69.44s/experiment, alias=policy]
[8/11] Running Statistical Verification Analysis (Phase 3b)...
9/12 stats:  67%|████████████████████████████████▋                | 8/12 [06:35<04:37, 69.44s/experiment, alias=policy][handoff] Launching stats (9/12)
  -> [stats] Started at 2026-04-02 02:02:47+05:45
==========================================================
 Starting Experiment: stats
 Start Time: 2026-04-02 02:02:51+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=stats
==========================================================
[2026-04-02 02:02:55,931][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-04-02 02:02:55,953:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 02:02:55,953][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 02:02:55,954][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-04-02 02:02:55,966][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\stats\run_20260402_020255
[2026-04-02 02:02:55,967][__main__][INFO] - ============================================================
[2026-04-02 02:02:55,969][__main__][INFO] -   Phase VII: Statistical Summary
[2026-04-02 02:02:55,970][__main__][INFO] - ============================================================
[2026-04-02 02:02:56,147][__main__][INFO] - Initiating statistical comparison (n=10 seeds).
[2026-04-02 02:02:56,181][__main__][INFO] - Environment: N=2, rho=0.40
[2026-04-02 02:02:57,227][__main__][INFO] - Loaded trained model from C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\reinforce_train\run_20260402_015946\n_gibbsq_reinforce_weights.eqx
stats:   0%|                                                                                  | 0/2 [00:00<?, ?stage/s][2026-04-02 02:02:57,235][__main__][INFO] - Running 10 GibbsQ SSA simulations with policy='uas'...
stats:  50%|█████████████████████████████████████                                     | 1/2 [00:01<00:01,  1.84s/stage][2026-04-02 02:02:59,075][__main__][INFO] - Running 10 Neural SSA simulations...
[2026-04-02 02:02:59,075][__main__][INFO] - Neural evaluation mode: deterministic
stats: 100%|██████████████████████████████████████████████████████████████████████████| 2/2 [00:40<00:00, 20.14s/stage]
[2026-04-02 02:03:37,527][__main__][INFO] -
============================================================
[2026-04-02 02:03:37,528][__main__][INFO] -   STATISTICAL SUMMARY
[2026-04-02 02:03:37,528][__main__][INFO] - ============================================================
[2026-04-02 02:03:37,528][__main__][INFO] - GibbsQ E[Q]:   1.1150 ± 0.0308
[2026-04-02 02:03:37,529][__main__][INFO] - N-GibbsQ E[Q]:   1.0710 ± 0.0372
[2026-04-02 02:03:37,529][__main__][INFO] - Rel. Improve:  3.95%
[2026-04-02 02:03:37,529][__main__][INFO] - ----------------------------------------
[2026-04-02 02:03:37,529][__main__][INFO] - P-Value:       9.95e-03 (SIGNIFICANT)
[2026-04-02 02:03:37,529][__main__][INFO] - Effect Size:   -1.29 (Cohen's d)
[2026-04-02 02:03:37,530][__main__][INFO] - 95% CI (Diff): [-0.0761, -0.0119]
[2026-04-02 02:03:37,530][__main__][INFO] - ============================================================
[2026-04-02 02:03:38,031][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:03:38,105][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:03:38,194][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:03:38,812][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:03:38,867][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode

[Experiment 'stats' Finished]
  Status: completed
  End Time: 2026-04-02 02:03:39+05:45
  Elapsed Duration: 48.471s
  -> [stats] Ended at 2026-04-02 02:03:39+05:45
  -> [stats] Status: completed
  -> [stats] Elapsed: 52.468s
9/12 stats:  75%|█████████████████████████████████████▌            | 9/12 [07:27<03:12, 64.13s/experiment, alias=stats]
[9/11] Running Generalization Stress Heatmaps...
10/12 generalize:  75%|█████████████████████████████████           | 9/12 [07:27<03:12, 64.13s/experiment, alias=stats][handoff] Launching generalize (10/12)
  -> [generalize] Started at 2026-04-02 02:03:39+05:45
==========================================================
 Starting Experiment: generalize
 Start Time: 2026-04-02 02:03:43+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=generalize
==========================================================
[2026-04-02 02:03:48,423][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-04-02 02:03:48,448:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 02:03:48,448][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 02:03:48,449][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-04-02 02:03:48,462][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\generalize\run_20260402_020348
[2026-04-02 02:03:48,463][__main__][INFO] - ============================================================
[2026-04-02 02:03:48,466][__main__][INFO] -   Phase VIII: Generalization & Stress Heatmap
[2026-04-02 02:03:48,467][__main__][INFO] - ============================================================
[2026-04-02 02:03:48,639][__main__][INFO] - Initiating Generalization Sweep (Scales=[0.5, 2.0], rho=[0.5, 0.85])
[2026-04-02 02:03:49,738][__main__][INFO] - Evaluating N-GibbsQ improvement ratio (GibbsQ / Neural) on 5x5 Grid...
generalize:   0%|                                                                              | 0/4 [00:00<?, ?cell/s][2026-04-02 02:04:15,198][__main__][INFO] -    Scale=  0.5x | rho=0.50 | Improvement=1.09x
generalize:  25%|███████████▊                                   | 1/4 [00:25<01:16, 25.46s/cell, scale=0.50x, rho=0.50][2026-04-02 02:04:56,425][__main__][INFO] -    Scale=  0.5x | rho=0.85 | Improvement=0.95x
generalize:  50%|███████████████████████▌                       | 2/4 [01:06<01:09, 34.73s/cell, scale=0.50x, rho=0.85][2026-04-02 02:06:36,093][__main__][INFO] -    Scale=  2.0x | rho=0.50 | Improvement=1.08x
generalize:  75%|███████████████████████████████████▎           | 3/4 [02:46<01:04, 64.38s/cell, scale=2.00x, rho=0.50][2026-04-02 02:09:20,798][__main__][INFO] -    Scale=  2.0x | rho=0.85 | Improvement=0.95x
generalize: 100%|███████████████████████████████████████████████| 4/4 [05:31<00:00, 82.76s/cell, scale=2.00x, rho=0.85]
[2026-04-02 02:09:23,104][__main__][INFO] - Generalization analysis complete. Heatmap saved to outputs\small\generalize\run_20260402_020348\generalization_heatmap.png, outputs\small\generalize\run_20260402_020348\generalization_heatmap.pdf

[Experiment 'generalize' Finished]
  Status: completed
  End Time: 2026-04-02 02:09:23+05:45
  Elapsed Duration: 339.947s
  -> [generalize] Ended at 2026-04-02 02:09:23+05:45
  -> [generalize] Status: completed
  -> [generalize] Elapsed: 344.137s
10/12 generalize:  83%|██████████████████████████████▊      | 10/12 [13:11<05:01, 150.58s/experiment, alias=generalize]
[10/11] Running Critical Load Boundary Analysis...
11/12 critical:  83%|████████████████████████████████▌      | 10/12 [13:11<05:01, 150.58s/experiment, alias=generalize][handoff] Launching critical (11/12)
  -> [critical] Started at 2026-04-02 02:09:23+05:45
==========================================================
 Starting Experiment: critical
 Start Time: 2026-04-02 02:09:27+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=critical
==========================================================
[2026-04-02 02:09:32,280][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-04-02 02:09:32,300:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 02:09:32,300][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 02:09:32,301][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-04-02 02:09:32,312][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\critical\run_20260402_020932
[2026-04-02 02:09:32,313][__main__][INFO] - ============================================================
[2026-04-02 02:09:32,316][__main__][INFO] -   Phase VIII: The Critical Stability Boundary
[2026-04-02 02:09:32,317][__main__][INFO] - ============================================================
[2026-04-02 02:09:33,500][__main__][INFO] - System Capacity: 2.50
[2026-04-02 02:09:33,500][__main__][INFO] - Targeting Load Boundary: [0.95]
critical:   0%|                                                                                 | 0/1 [00:00<?, ?rho/s][2026-04-02 02:09:33,521][__main__][INFO] - Evaluating Boundary rho=0.950 (Arrival=2.375)...
                                                                                                                       [2026-04-02 02:16:06,962][__main__][INFO] -    => N-GibbsQ E[Q]: 22.08 | GibbsQ E[Q]: 18.22
critical: 100%|█████████████████████████████████████████████████████████████| 1/1 [06:33<00:00, 393.46s/rho, rho=0.950]
[2026-04-02 02:16:07,465][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:16:07,591][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:16:07,747][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:16:08,492][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:16:08,617][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:16:08,766][__main__][INFO] - Critical load test complete. Curve saved to outputs\small\critical\run_20260402_020932\critical_load_curve.png, outputs\small\critical\run_20260402_020932\critical_load_curve.pdf

[Experiment 'critical' Finished]
  Status: completed
  End Time: 2026-04-02 02:16:09+05:45
  Elapsed Duration: 401.635s
  -> [critical] Ended at 2026-04-02 02:16:09+05:45
  -> [critical] Status: completed
  -> [critical] Elapsed: 405.625s
11/12 critical:  92%|█████████████████████████████████████▌   | 11/12 [19:57<03:48, 228.64s/experiment, alias=critical]
[11/11] Running SSA Component Ablation...
12/12 ablation:  92%|█████████████████████████████████████▌   | 11/12 [19:57<03:48, 228.64s/experiment, alias=critical][handoff] Launching ablation (12/12)
  -> [ablation] Started at 2026-04-02 02:16:09+05:45
==========================================================
 Starting Experiment: ablation
 Start Time: 2026-04-02 02:16:13+05:45
 Config Profile: small
 Progress Mode: auto
 Remaining Args (Hydra Overrides): ++active_experiment=ablation
==========================================================
[2026-04-02 02:16:17,886][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\outputs\small\ablation\run_20260402_021617
[2026-04-02 02:16:17,887][__main__][INFO] - ============================================================
[2026-04-02 02:16:17,887][__main__][INFO] -   SSA-Based Ablation Study
[2026-04-02 02:16:17,888][__main__][INFO] - ============================================================
ablation:   0%|                                                                             | 0/4 [00:00<?, ?variant/s][2026-04-02 02:16:17,893][__main__][INFO] - ------------------------------------------------------------
[2026-04-02 02:16:17,893][__main__][INFO] - Training variant: Full Model
[2026-04-02 02:16:17,895][__main__][INFO] -   preprocessing=log1p, init_type=zero_final
INFO:2026-04-02 02:16:17,917:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-04-02 02:16:17,917][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
                                                                                                                       [2026-04-02 02:16:20,511][experiments.training.train_reinforce][INFO] -   JSQ Mean Queue (Target): 1.080300<?, ?stage/s]
[2026-04-02 02:16:20,511][experiments.training.train_reinforce][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-04-02 02:16:20,513][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-04-02 02:16:23,859][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
                                                                                                                       [2026-04-02 02:16:24,907][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6851 | Acc: 59.81%
bc_train:   0%|▎                                                                     | 1/201 [00:01<03:29,  1.05s/step][2026-04-02 02:16:26,036][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5689 | Acc: 94.88%
bc_train:  45%|███████████████████▉                        | 91/201 [00:02<00:01, 55.84step/s, loss=0.5697, acc=93.94%][2026-04-02 02:16:27,188][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5640 | Acc: 99.78%
bc_train: 100%|███████████████████████████████████████████| 201/201 [00:03<00:00, 60.38step/s, loss=0.5640, acc=99.78%]
[2026-04-02 02:16:27,189][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-04-02 02:16:30,806][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
                                                                                                                       [2026-04-02 02:16:31,267][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 1183523.6250
bc_value:   0%|                                                                              | 0/201 [00:00<?, ?step/s][2026-04-02 02:16:32,459][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 844392.2500
bc_value:  45%|██████████████████████▊                            | 90/201 [00:01<00:01, 67.29step/s, loss=852699.9375][2026-04-02 02:16:33,654][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 720637.8750
bc_value: 100%|██████████████████████████████████████████████████| 201/201 [00:02<00:00, 70.58step/s, loss=720637.8750]
                                                                                                                       [2026-04-02 02:16:33,661][experiments.training.train_reinforce][INFO] - ============================================================
[2026-04-02 02:16:33,662][experiments.training.train_reinforce][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-04-02 02:16:33,662][experiments.training.train_reinforce][INFO] - ============================================================
[2026-04-02 02:16:33,662][experiments.training.train_reinforce][INFO] -   Epochs: 5, Batch size: 4
[2026-04-02 02:16:33,662][experiments.training.train_reinforce][INFO] -   Simulation time: 1000.0
[2026-04-02 02:16:33,663][experiments.training.train_reinforce][INFO] - ------------------------------------------------------------
                                                                                                                       [2026-04-02 02:16:48,505][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: -0.0054 | mean_logp: -0.5794 | corr: -0.0428
[2026-04-02 02:16:48,506][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.1203 | V-Grad Norm: 40083.5195
reinforce_train:  20%|███████▊                               | 1/5 [01:03<04:15, 63.84s/epoch, queue=2.060, pi=-112.0%]
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:613: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_c.legend(loc="lower right", fontsize=7)
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:632: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_d.legend(loc="upper right", fontsize=7)
[2026-04-02 02:17:40,266][__main__][INFO] - Saved variant artifacts in outputs\small\ablation\run_20260402_021617\variant_1_full_model
                                                                                                                       [2026-04-02 02:17:40,305][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:17:41,254][gibbsq.engines.numpy_engine][INFO] -   -> 1,013 arrivals, 1,012 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:17:41,255][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:08,  1.05rep/s]
[2026-04-02 02:17:42,004][gibbsq.engines.numpy_engine][INFO] -   -> 1,015 arrivals, 1,015 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:17:42,005][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:06,  1.20rep/s]
[2026-04-02 02:17:42,709][gibbsq.engines.numpy_engine][INFO] -   -> 961 arrivals, 961 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:17:42,710][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:02<00:05,  1.29rep/s]
[2026-04-02 02:17:43,476][gibbsq.engines.numpy_engine][INFO] -   -> 1,043 arrivals, 1,042 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:17:43,477][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:03<00:04,  1.30rep/s]
[2026-04-02 02:17:44,243][gibbsq.engines.numpy_engine][INFO] -   -> 1,016 arrivals, 1,016 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:17:44,244][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:03<00:03,  1.30rep/s]
[2026-04-02 02:17:44,909][gibbsq.engines.numpy_engine][INFO] -   -> 908 arrivals, 907 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:17:44,910][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:04<00:02,  1.36rep/s]
[2026-04-02 02:17:45,642][gibbsq.engines.numpy_engine][INFO] -   -> 983 arrivals, 982 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:17:45,644][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:05<00:02,  1.36rep/s]
[2026-04-02 02:17:46,432][gibbsq.engines.numpy_engine][INFO] -   -> 1,023 arrivals, 1,022 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:17:46,434][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:06<00:01,  1.33rep/s]
[2026-04-02 02:17:47,447][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,048 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:17:47,449][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:07<00:00,  1.20rep/s]
[2026-04-02 02:17:48,250][gibbsq.engines.numpy_engine][INFO] -   -> 1,074 arrivals, 1,074 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:17:48,256][__main__][INFO] -   SSA E[Q_total] = 1.1137 +/- 0.0397
ablation:  25%|████████████▎                                    | 1/4 [01:30<04:31, 90.36s/variant, variant=Full Model][2026-04-02 02:17:48,258][__main__][INFO] - ------------------------------------------------------------
[2026-04-02 02:17:48,258][__main__][INFO] - Training variant: Ablated: No Log-Norm
[2026-04-02 02:17:48,259][__main__][INFO] -   preprocessing=none, init_type=zero_final
                                                                                                                       [2026-04-02 02:17:48,563][experiments.training.train_reinforce][INFO] -   JSQ Mean Queue (Target): 1.080300<?, ?stage/s]
[2026-04-02 02:17:48,564][experiments.training.train_reinforce][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-04-02 02:17:48,565][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-04-02 02:17:52,008][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
                                                                                                                       [2026-04-02 02:17:52,764][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6738 | Acc: 75.81%
bc_train:   0%|▎                                                                     | 1/201 [00:00<02:30,  1.33step/s][2026-04-02 02:17:53,950][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5613 | Acc: 100.00%
bc_train:  43%|██████████████████▌                        | 87/201 [00:01<00:01, 59.59step/s, loss=0.5614, acc=100.00%][2026-04-02 02:17:55,139][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5612 | Acc: 100.00%
bc_train: 100%|██████████████████████████████████████████| 201/201 [00:03<00:00, 64.22step/s, loss=0.5612, acc=100.00%]
[2026-04-02 02:17:55,141][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-04-02 02:17:58,699][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
                                                                                                                       [2026-04-02 02:17:59,141][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 1185054.5000
bc_value:   0%|                                                                              | 0/201 [00:00<?, ?step/s][2026-04-02 02:18:00,378][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 405025.2812
bc_value:  44%|██████████████████████▎                            | 88/201 [00:01<00:01, 65.57step/s, loss=455751.8125][2026-04-02 02:18:01,763][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 324025.5312
bc_value: 100%|██████████████████████████████████████████████████| 201/201 [00:03<00:00, 65.64step/s, loss=324025.5312]
                                                                                                                       [2026-04-02 02:18:01,771][experiments.training.train_reinforce][INFO] - ============================================================
[2026-04-02 02:18:01,771][experiments.training.train_reinforce][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-04-02 02:18:01,771][experiments.training.train_reinforce][INFO] - ============================================================
[2026-04-02 02:18:01,772][experiments.training.train_reinforce][INFO] -   Epochs: 5, Batch size: 4
[2026-04-02 02:18:01,772][experiments.training.train_reinforce][INFO] -   Simulation time: 1000.0
[2026-04-02 02:18:01,772][experiments.training.train_reinforce][INFO] - ------------------------------------------------------------
                                                                                                                       [2026-04-02 02:18:14,098][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: -0.0157 | mean_logp: -0.5848 | corr: -0.0440
[2026-04-02 02:18:14,098][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.0491 | V-Grad Norm: 72278.1641
reinforce_train:  20%|███████▊                               | 1/5 [01:01<04:05, 61.37s/epoch, queue=2.073, pi=-114.6%]
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:613: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_c.legend(loc="lower right", fontsize=7)
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:632: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_d.legend(loc="upper right", fontsize=7)
[2026-04-02 02:19:05,577][__main__][INFO] - Saved variant artifacts in outputs\small\ablation\run_20260402_021617\variant_2_ablated_no_log-norm
                                                                                                                       [2026-04-02 02:19:05,592][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:19:06,360][gibbsq.engines.numpy_engine][INFO] -   -> 1,013 arrivals, 1,012 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:19:06,361][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:06,  1.30rep/s]
[2026-04-02 02:19:07,162][gibbsq.engines.numpy_engine][INFO] -   -> 1,015 arrivals, 1,015 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:19:07,164][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:06,  1.27rep/s]
[2026-04-02 02:19:07,879][gibbsq.engines.numpy_engine][INFO] -   -> 961 arrivals, 961 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:19:07,880][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:02<00:05,  1.32rep/s]
[2026-04-02 02:19:08,669][gibbsq.engines.numpy_engine][INFO] -   -> 1,043 arrivals, 1,042 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:19:08,671][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:03<00:04,  1.30rep/s]
[2026-04-02 02:19:09,667][gibbsq.engines.numpy_engine][INFO] -   -> 1,016 arrivals, 1,016 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:19:09,669][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:04<00:04,  1.17rep/s]
[2026-04-02 02:19:10,421][gibbsq.engines.numpy_engine][INFO] -   -> 909 arrivals, 907 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:19:10,423][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:04<00:03,  1.22rep/s]
[2026-04-02 02:19:11,180][gibbsq.engines.numpy_engine][INFO] -   -> 983 arrivals, 982 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:19:11,182][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:05<00:02,  1.25rep/s]
[2026-04-02 02:19:11,986][gibbsq.engines.numpy_engine][INFO] -   -> 1,023 arrivals, 1,022 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:19:11,987][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:06<00:01,  1.25rep/s]
[2026-04-02 02:19:12,782][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,048 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:19:12,783][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:07<00:00,  1.25rep/s]
[2026-04-02 02:19:13,632][gibbsq.engines.numpy_engine][INFO] -   -> 1,074 arrivals, 1,074 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:19:13,635][__main__][INFO] -   SSA E[Q_total] = 1.1104 +/- 0.0404
ablation:  50%|███████████████████▌                   | 2/4 [02:55<02:54, 87.43s/variant, variant=Ablated: No Log-Norm][2026-04-02 02:19:13,637][__main__][INFO] - ------------------------------------------------------------
[2026-04-02 02:19:13,637][__main__][INFO] - Training variant: Ablated: No Zero-Init
[2026-04-02 02:19:13,637][__main__][INFO] -   preprocessing=log1p, init_type=standard
                                                                                                                       [2026-04-02 02:19:13,948][experiments.training.train_reinforce][INFO] -   JSQ Mean Queue (Target): 1.080300<?, ?stage/s]
[2026-04-02 02:19:13,948][experiments.training.train_reinforce][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-04-02 02:19:13,949][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-04-02 02:19:17,590][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
                                                                                                                       [2026-04-02 02:19:18,235][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.8240 | Acc: 24.19%
bc_train:   0%|▎                                                                     | 1/201 [00:00<02:08,  1.56step/s][2026-04-02 02:19:19,517][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5703 | Acc: 92.43%
bc_train:  40%|█████████████████▋                          | 81/201 [00:01<00:02, 57.95step/s, loss=0.5715, acc=91.89%][2026-04-02 02:19:20,911][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5652 | Acc: 97.41%
bc_train: 100%|███████████████████████████████████████████| 201/201 [00:03<00:00, 60.56step/s, loss=0.5652, acc=97.41%]
[2026-04-02 02:19:20,913][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-04-02 02:19:24,568][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
                                                                                                                       [2026-04-02 02:19:25,036][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 1184283.8750
bc_value:   0%|                                                                              | 0/201 [00:00<?, ?step/s][2026-04-02 02:19:26,491][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 842404.1250
bc_value:  37%|██████████████████▊                                | 74/201 [00:01<00:02, 55.10step/s, loss=878796.2500][2026-04-02 02:19:27,932][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 717838.1250
bc_value: 100%|██████████████████████████████████████████████████| 201/201 [00:03<00:00, 59.76step/s, loss=717838.1250]
                                                                                                                       [2026-04-02 02:19:27,940][experiments.training.train_reinforce][INFO] - ============================================================
[2026-04-02 02:19:27,941][experiments.training.train_reinforce][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-04-02 02:19:27,941][experiments.training.train_reinforce][INFO] - ============================================================
[2026-04-02 02:19:27,941][experiments.training.train_reinforce][INFO] -   Epochs: 5, Batch size: 4
[2026-04-02 02:19:27,942][experiments.training.train_reinforce][INFO] -   Simulation time: 1000.0
[2026-04-02 02:19:27,942][experiments.training.train_reinforce][INFO] - ------------------------------------------------------------
                                                                                                                       [2026-04-02 02:19:40,788][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: -0.0070 | mean_logp: -0.5826 | corr: -0.0422
[2026-04-02 02:19:40,788][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.1299 | V-Grad Norm: 42108.5664
reinforce_train:  20%|███████▊                               | 1/5 [01:06<04:24, 66.14s/epoch, queue=2.058, pi=-111.6%]
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:613: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_c.legend(loc="lower right", fontsize=7)
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ\src\gibbsq\analysis\plotting.py:632: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_d.legend(loc="upper right", fontsize=7)
[2026-04-02 02:20:36,582][__main__][INFO] - Saved variant artifacts in outputs\small\ablation\run_20260402_021617\variant_3_ablated_no_zero-init
                                                                                                                       [2026-04-02 02:20:36,597][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:20:37,431][gibbsq.engines.numpy_engine][INFO] -   -> 1,013 arrivals, 1,012 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:20:37,432][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)1/10 [00:00<00:07,  1.20rep/s]
[2026-04-02 02:20:38,535][gibbsq.engines.numpy_engine][INFO] -   -> 1,015 arrivals, 1,015 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:20:38,537][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)2/10 [00:01<00:07,  1.01rep/s]
[2026-04-02 02:20:39,365][gibbsq.engines.numpy_engine][INFO] -   -> 961 arrivals, 961 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:20:39,366][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)3/10 [00:02<00:06,  1.09rep/s]
[2026-04-02 02:20:40,235][gibbsq.engines.numpy_engine][INFO] -   -> 1,043 arrivals, 1,042 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:20:40,237][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)4/10 [00:03<00:05,  1.11rep/s]
[2026-04-02 02:20:41,090][gibbsq.engines.numpy_engine][INFO] -   -> 1,016 arrivals, 1,016 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:20:41,092][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:04<00:04,  1.13rep/s]
[2026-04-02 02:20:41,858][gibbsq.engines.numpy_engine][INFO] -   -> 908 arrivals, 907 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:20:41,859][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)6/10 [00:05<00:03,  1.18rep/s]
[2026-04-02 02:20:42,691][gibbsq.engines.numpy_engine][INFO] -   -> 983 arrivals, 982 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:20:42,692][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)7/10 [00:06<00:02,  1.19rep/s]
[2026-04-02 02:20:43,548][gibbsq.engines.numpy_engine][INFO] -   -> 1,023 arrivals, 1,022 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:20:43,549][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)8/10 [00:06<00:01,  1.18rep/s]
[2026-04-02 02:20:44,424][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,048 departures, final Q_total = 2
                                                                                                                       [2026-04-02 02:20:44,426][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)/10 [00:07<00:00,  1.17rep/s]
[2026-04-02 02:20:45,338][gibbsq.engines.numpy_engine][INFO] -   -> 1,074 arrivals, 1,074 departures, final Q_total = 0
                                                                                                                       [2026-04-02 02:20:45,341][__main__][INFO] -   SSA E[Q_total] = 1.1137 +/- 0.0397
ablation:  75%|████████████████████████████▌         | 3/4 [04:27<01:29, 89.38s/variant, variant=Ablated: No Zero-Init][2026-04-02 02:20:45,343][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)      | 0/10 [00:00<?, ?rep/s]
[2026-04-02 02:20:45,440][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-04-02 02:20:45,440][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-04-02 02:20:45,540][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-04-02 02:20:45,541][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-04-02 02:20:45,643][gibbsq.engines.numpy_engine][INFO] -   -> 991 arrivals, 990 departures, final Q_total = 1
[2026-04-02 02:20:45,644][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-04-02 02:20:45,753][gibbsq.engines.numpy_engine][INFO] -   -> 1,040 arrivals, 1,040 departures, final Q_total = 0
[2026-04-02 02:20:45,754][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-04-02 02:20:45,856][gibbsq.engines.numpy_engine][INFO] -   -> 1,027 arrivals, 1,026 departures, final Q_total = 1
                                                                                                                       [2026-04-02 02:20:45,858][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)5/10 [00:00<00:00,  9.73rep/s]
[2026-04-02 02:20:45,946][gibbsq.engines.numpy_engine][INFO] -   -> 911 arrivals, 911 departures, final Q_total = 0
[2026-04-02 02:20:45,946][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-04-02 02:20:46,048][gibbsq.engines.numpy_engine][INFO] -   -> 993 arrivals, 991 departures, final Q_total = 2
[2026-04-02 02:20:46,049][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-04-02 02:20:46,152][gibbsq.engines.numpy_engine][INFO] -   -> 1,029 arrivals, 1,027 departures, final Q_total = 2
[2026-04-02 02:20:46,153][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-04-02 02:20:46,254][gibbsq.engines.numpy_engine][INFO] -   -> 1,020 arrivals, 1,016 departures, final Q_total = 4
[2026-04-02 02:20:46,254][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-04-02 02:20:46,362][gibbsq.engines.numpy_engine][INFO] -   -> 1,058 arrivals, 1,058 departures, final Q_total = 0
ablation: 100%|██████████████████████████████████████| 4/4 [04:28<00:00, 67.12s/variant, variant=Ablated: No Zero-Init]
[2026-04-02 02:20:46,649][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:20:46,720][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:20:46,812][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:20:47,326][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-04-02 02:20:47,385][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode

[Experiment 'ablation' Finished]
  Status: completed
  End Time: 2026-04-02 02:20:55+05:45
  Elapsed Duration: 282.122s
  -> [ablation] Ended at 2026-04-02 02:20:55+05:45
  -> [ablation] Status: completed
  -> [ablation] Elapsed: 285.981s
12/12 ablation: 100%|█████████████████████████████████████████| 12/12 [24:43<00:00, 123.61s/experiment, alias=ablation]

==========================================================
  Pipeline fully complete.
  Pipeline Status: completed
  Pipeline Ended At: 2026-04-02 02:20:55+05:45
  Total Pipeline Runtime: 1483.284s
  Review '/outputs/' for your plots and logs.
==========================================================
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ>