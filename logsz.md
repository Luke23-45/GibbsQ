PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ> python scripts/execution/reproduction_pipeline.py +configs=fast
==========================================================
  GibbsQ Research Paper: Final Execution Pipeline
==========================================================

[Initiating Pipeline...]


[0/10] Running REINFORCE Gradient validation (Track 5)...
Running: reinforce_check
==========================================================
 Starting Experiment: reinforce_check
 Remaining Args (Hydra Overrides): +configs=fast
==========================================================
[2026-03-23 10:42:55,081][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\gradient_check\run_20260323_104255
[2026-03-23 10:42:55,082][__main__][INFO] - ============================================================
[2026-03-23 10:42:55,083][__main__][INFO] -   REINFORCE Gradient Estimator Validation
[2026-03-23 10:42:55,083][__main__][INFO] - ============================================================
INFO:2026-03-23 10:42:55,107:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 10:42:55,107][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 10:42:57,392][__main__][INFO] - Statistical Scaling: D=50 parameters.
[2026-03-23 10:42:57,392][__main__][INFO] - Adjusted Z-Critical Threshold: 3.29
[2026-03-23 10:42:57,396][__main__][INFO] - Scaled n_samples from 15000 -> 15000 to maintain confidence bounds.
[2026-03-23 10:42:57,397][__main__][INFO] - Computing REINFORCE gradient estimate...
[2026-03-23 10:43:13,144][__main__][INFO] - Computing finite-difference gradient estimate...
[2026-03-23 10:44:20,603][__main__][INFO] -   [OK] Param  10/50  (idx   845): RF=  2.234568 | FD=  1.853458 | diff=  0.381109 | z= 1.76 | RelErr= 0.2324 | CosSim= 0.9780
[2026-03-23 10:45:27,448][__main__][INFO] -   [OK] Param  20/50  (idx  5756): RF= 89.378365 | FD= 90.223021 | diff=  0.844656 | z= 0.24 | RelErr= 0.0360 | CosSim= 0.9994
[2026-03-23 10:46:34,188][__main__][INFO] -   [!!] Param  30/50  (idx  1038): RF=  2.128893 | FD=  0.708458 | diff=  1.420434 | z= 6.38 | RelErr= 0.0393 | CosSim= 0.9992
[2026-03-23 10:47:42,951][__main__][INFO] -   [OK] Param  40/50  (idx   454): RF=  0.109584 | FD= -0.044167 | diff=  0.153751 | z= 0.77 | RelErr= 0.0402 | CosSim= 0.9992
[2026-03-23 10:48:51,365][__main__][INFO] -   [OK] Param  50/50  (idx  3302): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.0434 | CosSim= 0.9991
[2026-03-23 10:48:51,378][__main__][INFO] - Relative error: 0.0434
[2026-03-23 10:48:51,378][__main__][INFO] - Cosine similarity: 0.9991
[2026-03-23 10:48:51,379][__main__][INFO] - Bias estimate (L2): 4.120166
[2026-03-23 10:48:51,379][__main__][INFO] - Relative bias: 0.0434
[2026-03-23 10:48:51,379][__main__][INFO] - Variance estimate: 2150.707764
[2026-03-23 10:48:51,380][__main__][INFO] - Passed: True
[2026-03-23 10:48:51,384][__main__][INFO] - Results saved to outputs\gradient_check\run_20260323_104255\gradient_check_result.json
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\src\gibbsq\utils\chart_exporter.py:108: UserWarning: Glyph 10003 (\N{CHECK MARK}) missing from font(s) Times New Roman.
  fig.savefig(
[2026-03-23 10:48:54,619][__main__][INFO] - Gradient scatter plot saved to outputs\gradient_check\run_20260323_104255\gradient_scatter.png, outputs\gradient_check\run_20260323_104255\gradient_scatter.pdf
[2026-03-23 10:48:54,619][__main__][INFO] - GRADIENT CHECK PASSED - REINFORCE estimator is valid

[1/11] Running Drift Verification (Phase 1a)...
Running: drift
==========================================================
 Starting Experiment: drift
 Remaining Args (Hydra Overrides): +configs=fast
==========================================================
[2026-03-23 10:49:00,390][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\small\drift_verification\run_20260323_104900
[2026-03-23 10:49:00,391][__main__][INFO] - System: N=2, lam=1.0, alpha=1.0, cap=2.5000
[2026-03-23 10:49:00,391][__main__][INFO] - Proof bounds: R=2.4431, eps=0.750000
[2026-03-23 10:49:00,392][__main__][INFO] - --- Grid Evaluation (q_max=50) ---
[2026-03-23 10:49:00,395][__main__][INFO] - States evaluated: 2,601
[2026-03-23 10:49:00,395][__main__][INFO] - Bound violations: 0
[2026-03-23 10:49:01,394][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-23 10:49:01,888][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-23 10:49:02,378][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-23 10:49:02,522][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-23 10:49:02,654][__main__][INFO] - Saved: outputs\small\drift_verification\run_20260323_104900\drift_heatmap.png, outputs\small\drift_verification\run_20260323_104900\drift_heatmap.pdf
[2026-03-23 10:49:02,861][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-23 10:49:02,985][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-23 10:49:03,554][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-23 10:49:03,650][matplotlib.mathtext][INFO] - Substituting symbol L from STIXNonUnicode
[2026-03-23 10:49:03,859][__main__][INFO] - Saved: outputs\small\drift_verification\run_20260323_104900\drift_vs_norm.png, outputs\small\drift_verification\run_20260323_104900\drift_vs_norm.pdf
[2026-03-23 10:49:03,861][__main__][INFO] - Drift verification complete.

[4/10] Running Corrected Policy Evaluation Benchmark (Track 4)...
Running: policy
==========================================================
 Starting Experiment: policy
 Remaining Args (Hydra Overrides): +configs=fast
==========================================================
[2026-03-23 10:49:09,911][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\policy_comparison\run_20260323_104909
[2026-03-23 10:49:09,912][__main__][INFO] - ============================================================
[2026-03-23 10:49:09,912][__main__][INFO] -   Corrected Policy Comparison
[2026-03-23 10:49:09,912][__main__][INFO] - ============================================================
[2026-03-23 10:49:09,913][__main__][INFO] - System: N=10, lambda=11.2000, Lambda=14.0000, rho=0.8000
[2026-03-23 10:49:09,914][__main__][INFO] - ------------------------------------------------------------
[2026-03-23 10:49:09,914][__main__][INFO] - Evaluating Tier 2: JSQ (Min Queue)...
[2026-03-23 10:49:09,915][gibbsq.engines.numpy_engine][INFO] - Replication 1/50  (seed=42)
[2026-03-23 10:49:17,875][gibbsq.engines.numpy_engine][INFO] -   -> 55,960 arrivals, 55,953 departures, final Q_total = 7
[2026-03-23 10:49:17,875][gibbsq.engines.numpy_engine][INFO] - Replication 2/50  (seed=43)
[2026-03-23 10:49:26,154][gibbsq.engines.numpy_engine][INFO] -   -> 56,220 arrivals, 56,205 departures, final Q_total = 15
[2026-03-23 10:49:26,155][gibbsq.engines.numpy_engine][INFO] - Replication 3/50  (seed=44)
[2026-03-23 10:49:35,367][gibbsq.engines.numpy_engine][INFO] -   -> 55,894 arrivals, 55,879 departures, final Q_total = 15
[2026-03-23 10:49:35,368][gibbsq.engines.numpy_engine][INFO] - Replication 4/50  (seed=45)
[2026-03-23 10:49:44,880][gibbsq.engines.numpy_engine][INFO] -   -> 55,593 arrivals, 55,578 departures, final Q_total = 15
[2026-03-23 10:49:44,880][gibbsq.engines.numpy_engine][INFO] - Replication 5/50  (seed=46)
[2026-03-23 10:49:52,553][gibbsq.engines.numpy_engine][INFO] -   -> 56,222 arrivals, 56,207 departures, final Q_total = 15
[2026-03-23 10:49:52,554][gibbsq.engines.numpy_engine][INFO] - Replication 6/50  (seed=47)
[2026-03-23 10:50:00,311][gibbsq.engines.numpy_engine][INFO] -   -> 55,985 arrivals, 55,971 departures, final Q_total = 14
[2026-03-23 10:50:00,312][gibbsq.engines.numpy_engine][INFO] - Replication 7/50  (seed=48)
[2026-03-23 10:50:08,568][gibbsq.engines.numpy_engine][INFO] -   -> 55,589 arrivals, 55,567 departures, final Q_total = 22
[2026-03-23 10:50:08,569][gibbsq.engines.numpy_engine][INFO] - Replication 8/50  (seed=49)
[2026-03-23 10:50:16,669][gibbsq.engines.numpy_engine][INFO] -   -> 56,101 arrivals, 56,096 departures, final Q_total = 5
[2026-03-23 10:50:16,669][gibbsq.engines.numpy_engine][INFO] - Replication 9/50  (seed=50)
[2026-03-23 10:50:24,289][gibbsq.engines.numpy_engine][INFO] -   -> 55,533 arrivals, 55,517 departures, final Q_total = 16
[2026-03-23 10:50:24,290][gibbsq.engines.numpy_engine][INFO] - Replication 10/50  (seed=51)
[2026-03-23 10:50:32,471][gibbsq.engines.numpy_engine][INFO] -   -> 55,776 arrivals, 55,770 departures, final Q_total = 6
[2026-03-23 10:50:32,472][gibbsq.engines.numpy_engine][INFO] - Replication 11/50  (seed=52)
[2026-03-23 10:50:40,269][gibbsq.engines.numpy_engine][INFO] -   -> 56,235 arrivals, 56,222 departures, final Q_total = 13
[2026-03-23 10:50:40,270][gibbsq.engines.numpy_engine][INFO] - Replication 12/50  (seed=53)
[2026-03-23 10:50:48,193][gibbsq.engines.numpy_engine][INFO] -   -> 56,184 arrivals, 56,175 departures, final Q_total = 9
[2026-03-23 10:50:48,194][gibbsq.engines.numpy_engine][INFO] - Replication 13/50  (seed=54)
[2026-03-23 10:50:55,867][gibbsq.engines.numpy_engine][INFO] -   -> 56,138 arrivals, 56,126 departures, final Q_total = 12
[2026-03-23 10:50:55,868][gibbsq.engines.numpy_engine][INFO] - Replication 14/50  (seed=55)
[2026-03-23 10:51:04,016][gibbsq.engines.numpy_engine][INFO] -   -> 55,576 arrivals, 55,565 departures, final Q_total = 11
[2026-03-23 10:51:04,017][gibbsq.engines.numpy_engine][INFO] - Replication 15/50  (seed=56)
[2026-03-23 10:51:11,735][gibbsq.engines.numpy_engine][INFO] -   -> 56,127 arrivals, 56,098 departures, final Q_total = 29
[2026-03-23 10:51:11,736][gibbsq.engines.numpy_engine][INFO] - Replication 16/50  (seed=57)
[2026-03-23 10:51:19,577][gibbsq.engines.numpy_engine][INFO] -   -> 55,703 arrivals, 55,690 departures, final Q_total = 13
[2026-03-23 10:51:19,578][gibbsq.engines.numpy_engine][INFO] - Replication 17/50  (seed=58)
[2026-03-23 10:51:27,673][gibbsq.engines.numpy_engine][INFO] -   -> 55,857 arrivals, 55,847 departures, final Q_total = 10
[2026-03-23 10:51:27,674][gibbsq.engines.numpy_engine][INFO] - Replication 18/50  (seed=59)
[2026-03-23 10:51:35,834][gibbsq.engines.numpy_engine][INFO] -   -> 56,201 arrivals, 56,190 departures, final Q_total = 11
[2026-03-23 10:51:35,835][gibbsq.engines.numpy_engine][INFO] - Replication 19/50  (seed=60)
[2026-03-23 10:51:44,269][gibbsq.engines.numpy_engine][INFO] -   -> 55,818 arrivals, 55,804 departures, final Q_total = 14
[2026-03-23 10:51:44,270][gibbsq.engines.numpy_engine][INFO] - Replication 20/50  (seed=61)
[2026-03-23 10:51:52,595][gibbsq.engines.numpy_engine][INFO] -   -> 56,188 arrivals, 56,182 departures, final Q_total = 6
[2026-03-23 10:51:52,596][gibbsq.engines.numpy_engine][INFO] - Replication 21/50  (seed=62)
[2026-03-23 10:52:01,052][gibbsq.engines.numpy_engine][INFO] -   -> 55,799 arrivals, 55,787 departures, final Q_total = 12
[2026-03-23 10:52:01,053][gibbsq.engines.numpy_engine][INFO] - Replication 22/50  (seed=63)
[2026-03-23 10:52:09,001][gibbsq.engines.numpy_engine][INFO] -   -> 55,714 arrivals, 55,710 departures, final Q_total = 4
[2026-03-23 10:52:09,001][gibbsq.engines.numpy_engine][INFO] - Replication 23/50  (seed=64)
[2026-03-23 10:52:17,233][gibbsq.engines.numpy_engine][INFO] -   -> 56,394 arrivals, 56,377 departures, final Q_total = 17
[2026-03-23 10:52:17,234][gibbsq.engines.numpy_engine][INFO] - Replication 24/50  (seed=65)
[2026-03-23 10:52:25,120][gibbsq.engines.numpy_engine][INFO] -   -> 55,876 arrivals, 55,868 departures, final Q_total = 8
[2026-03-23 10:52:25,120][gibbsq.engines.numpy_engine][INFO] - Replication 25/50  (seed=66)
[2026-03-23 10:52:33,211][gibbsq.engines.numpy_engine][INFO] -   -> 55,873 arrivals, 55,859 departures, final Q_total = 14
[2026-03-23 10:52:33,212][gibbsq.engines.numpy_engine][INFO] - Replication 26/50  (seed=67)
[2026-03-23 10:52:40,859][gibbsq.engines.numpy_engine][INFO] -   -> 56,025 arrivals, 56,017 departures, final Q_total = 8
[2026-03-23 10:52:40,860][gibbsq.engines.numpy_engine][INFO] - Replication 27/50  (seed=68)
[2026-03-23 10:52:49,324][gibbsq.engines.numpy_engine][INFO] -   -> 56,195 arrivals, 56,189 departures, final Q_total = 6
[2026-03-23 10:52:49,324][gibbsq.engines.numpy_engine][INFO] - Replication 28/50  (seed=69)
[2026-03-23 10:52:57,175][gibbsq.engines.numpy_engine][INFO] -   -> 55,535 arrivals, 55,515 departures, final Q_total = 20
[2026-03-23 10:52:57,176][gibbsq.engines.numpy_engine][INFO] - Replication 29/50  (seed=70)
[2026-03-23 10:53:05,161][gibbsq.engines.numpy_engine][INFO] -   -> 56,433 arrivals, 56,424 departures, final Q_total = 9
[2026-03-23 10:53:05,162][gibbsq.engines.numpy_engine][INFO] - Replication 30/50  (seed=71)
[2026-03-23 10:53:13,176][gibbsq.engines.numpy_engine][INFO] -   -> 56,000 arrivals, 55,989 departures, final Q_total = 11
[2026-03-23 10:53:13,176][gibbsq.engines.numpy_engine][INFO] - Replication 31/50  (seed=72)
[2026-03-23 10:53:21,362][gibbsq.engines.numpy_engine][INFO] -   -> 56,029 arrivals, 56,014 departures, final Q_total = 15
[2026-03-23 10:53:21,362][gibbsq.engines.numpy_engine][INFO] - Replication 32/50  (seed=73)
[2026-03-23 10:53:29,757][gibbsq.engines.numpy_engine][INFO] -   -> 55,963 arrivals, 55,941 departures, final Q_total = 22
[2026-03-23 10:53:29,758][gibbsq.engines.numpy_engine][INFO] - Replication 33/50  (seed=74)
[2026-03-23 10:53:37,534][gibbsq.engines.numpy_engine][INFO] -   -> 55,917 arrivals, 55,908 departures, final Q_total = 9
[2026-03-23 10:53:37,534][gibbsq.engines.numpy_engine][INFO] - Replication 34/50  (seed=75)
[2026-03-23 10:53:45,648][gibbsq.engines.numpy_engine][INFO] -   -> 56,153 arrivals, 56,136 departures, final Q_total = 17
[2026-03-23 10:53:45,649][gibbsq.engines.numpy_engine][INFO] - Replication 35/50  (seed=76)
[2026-03-23 10:53:53,332][gibbsq.engines.numpy_engine][INFO] -   -> 55,907 arrivals, 55,895 departures, final Q_total = 12
[2026-03-23 10:53:53,333][gibbsq.engines.numpy_engine][INFO] - Replication 36/50  (seed=77)
[2026-03-23 10:54:03,169][gibbsq.engines.numpy_engine][INFO] -   -> 55,649 arrivals, 55,640 departures, final Q_total = 9
[2026-03-23 10:54:03,170][gibbsq.engines.numpy_engine][INFO] - Replication 37/50  (seed=78)
[2026-03-23 10:54:11,519][gibbsq.engines.numpy_engine][INFO] -   -> 56,024 arrivals, 56,017 departures, final Q_total = 7
[2026-03-23 10:54:11,519][gibbsq.engines.numpy_engine][INFO] - Replication 38/50  (seed=79)
[2026-03-23 10:54:19,469][gibbsq.engines.numpy_engine][INFO] -   -> 56,163 arrivals, 56,149 departures, final Q_total = 14
[2026-03-23 10:54:19,469][gibbsq.engines.numpy_engine][INFO] - Replication 39/50  (seed=80)
[2026-03-23 10:54:27,536][gibbsq.engines.numpy_engine][INFO] -   -> 55,609 arrivals, 55,595 departures, final Q_total = 14
[2026-03-23 10:54:27,536][gibbsq.engines.numpy_engine][INFO] - Replication 40/50  (seed=81)
[2026-03-23 10:54:35,753][gibbsq.engines.numpy_engine][INFO] -   -> 56,011 arrivals, 56,000 departures, final Q_total = 11
[2026-03-23 10:54:35,754][gibbsq.engines.numpy_engine][INFO] - Replication 41/50  (seed=82)
[2026-03-23 10:54:43,710][gibbsq.engines.numpy_engine][INFO] -   -> 56,361 arrivals, 56,352 departures, final Q_total = 9
[2026-03-23 10:54:43,710][gibbsq.engines.numpy_engine][INFO] - Replication 42/50  (seed=83)
[2026-03-23 10:54:51,566][gibbsq.engines.numpy_engine][INFO] -   -> 55,676 arrivals, 55,664 departures, final Q_total = 12
[2026-03-23 10:54:51,566][gibbsq.engines.numpy_engine][INFO] - Replication 43/50  (seed=84)
[2026-03-23 10:54:59,594][gibbsq.engines.numpy_engine][INFO] -   -> 55,894 arrivals, 55,875 departures, final Q_total = 19
[2026-03-23 10:54:59,594][gibbsq.engines.numpy_engine][INFO] - Replication 44/50  (seed=85)
[2026-03-23 10:55:07,681][gibbsq.engines.numpy_engine][INFO] -   -> 55,906 arrivals, 55,893 departures, final Q_total = 13
[2026-03-23 10:55:07,681][gibbsq.engines.numpy_engine][INFO] - Replication 45/50  (seed=86)
[2026-03-23 10:55:15,620][gibbsq.engines.numpy_engine][INFO] -   -> 55,750 arrivals, 55,735 departures, final Q_total = 15
[2026-03-23 10:55:15,621][gibbsq.engines.numpy_engine][INFO] - Replication 46/50  (seed=87)
[2026-03-23 10:55:23,953][gibbsq.engines.numpy_engine][INFO] -   -> 55,838 arrivals, 55,825 departures, final Q_total = 13
[2026-03-23 10:55:23,954][gibbsq.engines.numpy_engine][INFO] - Replication 47/50  (seed=88)
[2026-03-23 10:55:31,681][gibbsq.engines.numpy_engine][INFO] -   -> 55,943 arrivals, 55,934 departures, final Q_total = 9
[2026-03-23 10:55:31,681][gibbsq.engines.numpy_engine][INFO] - Replication 48/50  (seed=89)
[2026-03-23 10:55:40,536][gibbsq.engines.numpy_engine][INFO] -   -> 56,311 arrivals, 56,292 departures, final Q_total = 19
[2026-03-23 10:55:40,536][gibbsq.engines.numpy_engine][INFO] - Replication 49/50  (seed=90)
[2026-03-23 10:55:48,629][gibbsq.engines.numpy_engine][INFO] -   -> 55,790 arrivals, 55,779 departures, final Q_total = 11
[2026-03-23 10:55:48,630][gibbsq.engines.numpy_engine][INFO] - Replication 50/50  (seed=91)
[2026-03-23 10:55:56,656][gibbsq.engines.numpy_engine][INFO] -   -> 56,206 arrivals, 56,194 departures, final Q_total = 12
[2026-03-23 10:55:56,699][__main__][INFO] -   E[Q_total] = 11.7770 ± 0.0336
[2026-03-23 10:55:56,703][__main__][INFO] - Evaluating Tier 2: JSSQ (Min Sojourn)...
[2026-03-23 10:55:56,704][gibbsq.engines.numpy_engine][INFO] - Replication 1/50  (seed=42)
[2026-03-23 10:56:05,184][gibbsq.engines.numpy_engine][INFO] -   -> 55,934 arrivals, 55,925 departures, final Q_total = 9
[2026-03-23 10:56:05,185][gibbsq.engines.numpy_engine][INFO] - Replication 2/50  (seed=43)
[2026-03-23 10:56:13,656][gibbsq.engines.numpy_engine][INFO] -   -> 56,218 arrivals, 56,202 departures, final Q_total = 16
[2026-03-23 10:56:13,657][gibbsq.engines.numpy_engine][INFO] - Replication 3/50  (seed=44)
[2026-03-23 10:56:22,013][gibbsq.engines.numpy_engine][INFO] -   -> 55,887 arrivals, 55,862 departures, final Q_total = 25
[2026-03-23 10:56:22,014][gibbsq.engines.numpy_engine][INFO] - Replication 4/50  (seed=45)
[2026-03-23 10:56:30,346][gibbsq.engines.numpy_engine][INFO] -   -> 55,568 arrivals, 55,560 departures, final Q_total = 8
[2026-03-23 10:56:30,346][gibbsq.engines.numpy_engine][INFO] - Replication 5/50  (seed=46)
[2026-03-23 10:56:38,538][gibbsq.engines.numpy_engine][INFO] -   -> 56,180 arrivals, 56,174 departures, final Q_total = 6
[2026-03-23 10:56:38,538][gibbsq.engines.numpy_engine][INFO] - Replication 6/50  (seed=47)
[2026-03-23 10:56:46,561][gibbsq.engines.numpy_engine][INFO] -   -> 55,952 arrivals, 55,923 departures, final Q_total = 29
[2026-03-23 10:56:46,562][gibbsq.engines.numpy_engine][INFO] - Replication 7/50  (seed=48)
[2026-03-23 10:56:55,494][gibbsq.engines.numpy_engine][INFO] -   -> 55,608 arrivals, 55,588 departures, final Q_total = 20
[2026-03-23 10:56:55,495][gibbsq.engines.numpy_engine][INFO] - Replication 8/50  (seed=49)
[2026-03-23 10:57:03,810][gibbsq.engines.numpy_engine][INFO] -   -> 56,112 arrivals, 56,109 departures, final Q_total = 3
[2026-03-23 10:57:03,811][gibbsq.engines.numpy_engine][INFO] - Replication 9/50  (seed=50)
[2026-03-23 10:57:12,456][gibbsq.engines.numpy_engine][INFO] -   -> 55,558 arrivals, 55,538 departures, final Q_total = 20
[2026-03-23 10:57:12,457][gibbsq.engines.numpy_engine][INFO] - Replication 10/50  (seed=51)
[2026-03-23 10:57:20,551][gibbsq.engines.numpy_engine][INFO] -   -> 55,800 arrivals, 55,796 departures, final Q_total = 4
[2026-03-23 10:57:20,552][gibbsq.engines.numpy_engine][INFO] - Replication 11/50  (seed=52)
[2026-03-23 10:57:28,702][gibbsq.engines.numpy_engine][INFO] -   -> 56,272 arrivals, 56,258 departures, final Q_total = 14
[2026-03-23 10:57:28,703][gibbsq.engines.numpy_engine][INFO] - Replication 12/50  (seed=53)
[2026-03-23 10:57:36,996][gibbsq.engines.numpy_engine][INFO] -   -> 56,187 arrivals, 56,179 departures, final Q_total = 8
[2026-03-23 10:57:36,997][gibbsq.engines.numpy_engine][INFO] - Replication 13/50  (seed=54)
[2026-03-23 10:57:45,729][gibbsq.engines.numpy_engine][INFO] -   -> 56,098 arrivals, 56,091 departures, final Q_total = 7
[2026-03-23 10:57:45,730][gibbsq.engines.numpy_engine][INFO] - Replication 14/50  (seed=55)
[2026-03-23 10:57:53,945][gibbsq.engines.numpy_engine][INFO] -   -> 55,611 arrivals, 55,598 departures, final Q_total = 13
[2026-03-23 10:57:53,946][gibbsq.engines.numpy_engine][INFO] - Replication 15/50  (seed=56)
[2026-03-23 10:58:02,482][gibbsq.engines.numpy_engine][INFO] -   -> 56,063 arrivals, 56,043 departures, final Q_total = 20
[2026-03-23 10:58:02,483][gibbsq.engines.numpy_engine][INFO] - Replication 16/50  (seed=57)
[2026-03-23 10:58:10,827][gibbsq.engines.numpy_engine][INFO] -   -> 55,664 arrivals, 55,651 departures, final Q_total = 13
[2026-03-23 10:58:10,827][gibbsq.engines.numpy_engine][INFO] - Replication 17/50  (seed=58)
[2026-03-23 10:58:19,089][gibbsq.engines.numpy_engine][INFO] -   -> 55,776 arrivals, 55,764 departures, final Q_total = 12
[2026-03-23 10:58:19,090][gibbsq.engines.numpy_engine][INFO] - Replication 18/50  (seed=59)
[2026-03-23 10:58:27,416][gibbsq.engines.numpy_engine][INFO] -   -> 56,155 arrivals, 56,148 departures, final Q_total = 7
[2026-03-23 10:58:27,416][gibbsq.engines.numpy_engine][INFO] - Replication 19/50  (seed=60)
[2026-03-23 10:58:35,857][gibbsq.engines.numpy_engine][INFO] -   -> 55,831 arrivals, 55,815 departures, final Q_total = 16
[2026-03-23 10:58:35,857][gibbsq.engines.numpy_engine][INFO] - Replication 20/50  (seed=61)
[2026-03-23 10:58:44,351][gibbsq.engines.numpy_engine][INFO] -   -> 56,153 arrivals, 56,136 departures, final Q_total = 17
[2026-03-23 10:58:44,352][gibbsq.engines.numpy_engine][INFO] - Replication 21/50  (seed=62)
[2026-03-23 10:58:52,636][gibbsq.engines.numpy_engine][INFO] -   -> 55,850 arrivals, 55,836 departures, final Q_total = 14
[2026-03-23 10:58:52,637][gibbsq.engines.numpy_engine][INFO] - Replication 22/50  (seed=63)
[2026-03-23 10:59:01,062][gibbsq.engines.numpy_engine][INFO] -   -> 55,693 arrivals, 55,681 departures, final Q_total = 12
[2026-03-23 10:59:01,063][gibbsq.engines.numpy_engine][INFO] - Replication 23/50  (seed=64)
[2026-03-23 10:59:09,285][gibbsq.engines.numpy_engine][INFO] -   -> 56,442 arrivals, 56,425 departures, final Q_total = 17
[2026-03-23 10:59:09,286][gibbsq.engines.numpy_engine][INFO] - Replication 24/50  (seed=65)
[2026-03-23 10:59:17,917][gibbsq.engines.numpy_engine][INFO] -   -> 55,806 arrivals, 55,802 departures, final Q_total = 4
[2026-03-23 10:59:17,917][gibbsq.engines.numpy_engine][INFO] - Replication 25/50  (seed=66)
[2026-03-23 10:59:26,359][gibbsq.engines.numpy_engine][INFO] -   -> 55,773 arrivals, 55,765 departures, final Q_total = 8
[2026-03-23 10:59:26,360][gibbsq.engines.numpy_engine][INFO] - Replication 26/50  (seed=67)
[2026-03-23 10:59:35,037][gibbsq.engines.numpy_engine][INFO] -   -> 56,049 arrivals, 56,044 departures, final Q_total = 5
[2026-03-23 10:59:35,038][gibbsq.engines.numpy_engine][INFO] - Replication 27/50  (seed=68)
[2026-03-23 10:59:43,671][gibbsq.engines.numpy_engine][INFO] -   -> 56,204 arrivals, 56,201 departures, final Q_total = 3
[2026-03-23 10:59:43,672][gibbsq.engines.numpy_engine][INFO] - Replication 28/50  (seed=69)
[2026-03-23 10:59:52,340][gibbsq.engines.numpy_engine][INFO] -   -> 55,544 arrivals, 55,524 departures, final Q_total = 20
[2026-03-23 10:59:52,341][gibbsq.engines.numpy_engine][INFO] - Replication 29/50  (seed=70)
[2026-03-23 11:00:00,790][gibbsq.engines.numpy_engine][INFO] -   -> 56,384 arrivals, 56,376 departures, final Q_total = 8
[2026-03-23 11:00:00,791][gibbsq.engines.numpy_engine][INFO] - Replication 30/50  (seed=71)
[2026-03-23 11:00:09,453][gibbsq.engines.numpy_engine][INFO] -   -> 56,059 arrivals, 56,054 departures, final Q_total = 5
[2026-03-23 11:00:09,453][gibbsq.engines.numpy_engine][INFO] - Replication 31/50  (seed=72)
[2026-03-23 11:00:18,056][gibbsq.engines.numpy_engine][INFO] -   -> 56,031 arrivals, 56,017 departures, final Q_total = 14
[2026-03-23 11:00:18,057][gibbsq.engines.numpy_engine][INFO] - Replication 32/50  (seed=73)
[2026-03-23 11:00:26,704][gibbsq.engines.numpy_engine][INFO] -   -> 56,003 arrivals, 55,994 departures, final Q_total = 9
[2026-03-23 11:00:26,705][gibbsq.engines.numpy_engine][INFO] - Replication 33/50  (seed=74)
[2026-03-23 11:00:34,954][gibbsq.engines.numpy_engine][INFO] -   -> 55,860 arrivals, 55,851 departures, final Q_total = 9
[2026-03-23 11:00:34,955][gibbsq.engines.numpy_engine][INFO] - Replication 34/50  (seed=75)
[2026-03-23 11:00:43,755][gibbsq.engines.numpy_engine][INFO] -   -> 56,104 arrivals, 56,095 departures, final Q_total = 9
[2026-03-23 11:00:43,756][gibbsq.engines.numpy_engine][INFO] - Replication 35/50  (seed=76)
[2026-03-23 11:00:52,784][gibbsq.engines.numpy_engine][INFO] -   -> 55,844 arrivals, 55,828 departures, final Q_total = 16
[2026-03-23 11:00:52,784][gibbsq.engines.numpy_engine][INFO] - Replication 36/50  (seed=77)
[2026-03-23 11:01:01,147][gibbsq.engines.numpy_engine][INFO] -   -> 55,617 arrivals, 55,609 departures, final Q_total = 8
[2026-03-23 11:01:01,147][gibbsq.engines.numpy_engine][INFO] - Replication 37/50  (seed=78)
[2026-03-23 11:01:09,774][gibbsq.engines.numpy_engine][INFO] -   -> 55,951 arrivals, 55,929 departures, final Q_total = 22
[2026-03-23 11:01:09,774][gibbsq.engines.numpy_engine][INFO] - Replication 38/50  (seed=79)
[2026-03-23 11:01:20,380][gibbsq.engines.numpy_engine][INFO] -   -> 56,143 arrivals, 56,132 departures, final Q_total = 11
[2026-03-23 11:01:20,381][gibbsq.engines.numpy_engine][INFO] - Replication 39/50  (seed=80)
[2026-03-23 11:01:29,229][gibbsq.engines.numpy_engine][INFO] -   -> 55,570 arrivals, 55,551 departures, final Q_total = 19
[2026-03-23 11:01:29,229][gibbsq.engines.numpy_engine][INFO] - Replication 40/50  (seed=81)
[2026-03-23 11:01:37,134][gibbsq.engines.numpy_engine][INFO] -   -> 55,953 arrivals, 55,939 departures, final Q_total = 14
[2026-03-23 11:01:37,135][gibbsq.engines.numpy_engine][INFO] - Replication 41/50  (seed=82)
[2026-03-23 11:01:45,285][gibbsq.engines.numpy_engine][INFO] -   -> 56,454 arrivals, 56,449 departures, final Q_total = 5
[2026-03-23 11:01:45,286][gibbsq.engines.numpy_engine][INFO] - Replication 42/50  (seed=83)
[2026-03-23 11:01:54,082][gibbsq.engines.numpy_engine][INFO] -   -> 55,699 arrivals, 55,681 departures, final Q_total = 18
[2026-03-23 11:01:54,083][gibbsq.engines.numpy_engine][INFO] - Replication 43/50  (seed=84)
[2026-03-23 11:02:02,541][gibbsq.engines.numpy_engine][INFO] -   -> 55,878 arrivals, 55,861 departures, final Q_total = 17
[2026-03-23 11:02:02,542][gibbsq.engines.numpy_engine][INFO] - Replication 44/50  (seed=85)
[2026-03-23 11:02:10,708][gibbsq.engines.numpy_engine][INFO] -   -> 55,878 arrivals, 55,873 departures, final Q_total = 5
[2026-03-23 11:02:10,709][gibbsq.engines.numpy_engine][INFO] - Replication 45/50  (seed=86)
[2026-03-23 11:02:19,338][gibbsq.engines.numpy_engine][INFO] -   -> 55,865 arrivals, 55,858 departures, final Q_total = 7
[2026-03-23 11:02:19,339][gibbsq.engines.numpy_engine][INFO] - Replication 46/50  (seed=87)
[2026-03-23 11:02:27,537][gibbsq.engines.numpy_engine][INFO] -   -> 55,817 arrivals, 55,805 departures, final Q_total = 12
[2026-03-23 11:02:27,538][gibbsq.engines.numpy_engine][INFO] - Replication 47/50  (seed=88)
[2026-03-23 11:02:35,493][gibbsq.engines.numpy_engine][INFO] -   -> 55,910 arrivals, 55,906 departures, final Q_total = 4
[2026-03-23 11:02:35,494][gibbsq.engines.numpy_engine][INFO] - Replication 48/50  (seed=89)
[2026-03-23 11:02:43,485][gibbsq.engines.numpy_engine][INFO] -   -> 56,270 arrivals, 56,260 departures, final Q_total = 10
[2026-03-23 11:02:43,485][gibbsq.engines.numpy_engine][INFO] - Replication 49/50  (seed=90)
[2026-03-23 11:02:51,835][gibbsq.engines.numpy_engine][INFO] -   -> 55,799 arrivals, 55,793 departures, final Q_total = 6
[2026-03-23 11:02:51,836][gibbsq.engines.numpy_engine][INFO] - Replication 50/50  (seed=91)
[2026-03-23 11:02:59,671][gibbsq.engines.numpy_engine][INFO] -   -> 56,161 arrivals, 56,151 departures, final Q_total = 10
[2026-03-23 11:02:59,715][__main__][INFO] -   E[Q_total] = 10.9659 ± 0.0385
[2026-03-23 11:02:59,716][__main__][INFO] - Evaluating Tier 3: GibbsQ-Sojourn (alpha=1.0)...
[2026-03-23 11:02:59,717][gibbsq.engines.numpy_engine][INFO] - Replication 1/50  (seed=42)
[2026-03-23 11:03:09,132][gibbsq.engines.numpy_engine][INFO] -   -> 55,953 arrivals, 55,940 departures, final Q_total = 13
[2026-03-23 11:03:09,132][gibbsq.engines.numpy_engine][INFO] - Replication 2/50  (seed=43)
[2026-03-23 11:03:18,506][gibbsq.engines.numpy_engine][INFO] -   -> 56,205 arrivals, 56,181 departures, final Q_total = 24
[2026-03-23 11:03:18,507][gibbsq.engines.numpy_engine][INFO] - Replication 3/50  (seed=44)
[2026-03-23 11:03:27,078][gibbsq.engines.numpy_engine][INFO] -   -> 55,919 arrivals, 55,907 departures, final Q_total = 12
[2026-03-23 11:03:27,079][gibbsq.engines.numpy_engine][INFO] - Replication 4/50  (seed=45)
[2026-03-23 11:03:35,416][gibbsq.engines.numpy_engine][INFO] -   -> 55,609 arrivals, 55,585 departures, final Q_total = 24
[2026-03-23 11:03:35,417][gibbsq.engines.numpy_engine][INFO] - Replication 5/50  (seed=46)
[2026-03-23 11:03:43,667][gibbsq.engines.numpy_engine][INFO] -   -> 56,119 arrivals, 56,097 departures, final Q_total = 22
[2026-03-23 11:03:43,668][gibbsq.engines.numpy_engine][INFO] - Replication 6/50  (seed=47)
[2026-03-23 11:03:51,584][gibbsq.engines.numpy_engine][INFO] -   -> 55,826 arrivals, 55,816 departures, final Q_total = 10
[2026-03-23 11:03:51,585][gibbsq.engines.numpy_engine][INFO] - Replication 7/50  (seed=48)
[2026-03-23 11:04:00,138][gibbsq.engines.numpy_engine][INFO] -   -> 55,590 arrivals, 55,562 departures, final Q_total = 28
[2026-03-23 11:04:00,138][gibbsq.engines.numpy_engine][INFO] - Replication 8/50  (seed=49)
[2026-03-23 11:04:08,566][gibbsq.engines.numpy_engine][INFO] -   -> 56,130 arrivals, 56,119 departures, final Q_total = 11
[2026-03-23 11:04:08,567][gibbsq.engines.numpy_engine][INFO] - Replication 9/50  (seed=50)
[2026-03-23 11:04:17,166][gibbsq.engines.numpy_engine][INFO] -   -> 55,518 arrivals, 55,500 departures, final Q_total = 18
[2026-03-23 11:04:17,166][gibbsq.engines.numpy_engine][INFO] - Replication 10/50  (seed=51)
[2026-03-23 11:04:25,089][gibbsq.engines.numpy_engine][INFO] -   -> 55,826 arrivals, 55,809 departures, final Q_total = 17
[2026-03-23 11:04:25,089][gibbsq.engines.numpy_engine][INFO] - Replication 11/50  (seed=52)
[2026-03-23 11:04:33,195][gibbsq.engines.numpy_engine][INFO] -   -> 56,199 arrivals, 56,188 departures, final Q_total = 11
[2026-03-23 11:04:33,196][gibbsq.engines.numpy_engine][INFO] - Replication 12/50  (seed=53)
[2026-03-23 11:04:41,850][gibbsq.engines.numpy_engine][INFO] -   -> 56,220 arrivals, 56,205 departures, final Q_total = 15
[2026-03-23 11:04:41,850][gibbsq.engines.numpy_engine][INFO] - Replication 13/50  (seed=54)
[2026-03-23 11:04:50,195][gibbsq.engines.numpy_engine][INFO] -   -> 56,100 arrivals, 56,088 departures, final Q_total = 12
[2026-03-23 11:04:50,196][gibbsq.engines.numpy_engine][INFO] - Replication 14/50  (seed=55)
[2026-03-23 11:04:57,962][gibbsq.engines.numpy_engine][INFO] -   -> 55,631 arrivals, 55,613 departures, final Q_total = 18
[2026-03-23 11:04:57,962][gibbsq.engines.numpy_engine][INFO] - Replication 15/50  (seed=56)
[2026-03-23 11:05:06,192][gibbsq.engines.numpy_engine][INFO] -   -> 56,198 arrivals, 56,180 departures, final Q_total = 18
[2026-03-23 11:05:06,193][gibbsq.engines.numpy_engine][INFO] - Replication 16/50  (seed=57)
[2026-03-23 11:05:14,285][gibbsq.engines.numpy_engine][INFO] -   -> 55,656 arrivals, 55,636 departures, final Q_total = 20
[2026-03-23 11:05:14,286][gibbsq.engines.numpy_engine][INFO] - Replication 17/50  (seed=58)
[2026-03-23 11:05:23,629][gibbsq.engines.numpy_engine][INFO] -   -> 55,772 arrivals, 55,759 departures, final Q_total = 13
[2026-03-23 11:05:23,630][gibbsq.engines.numpy_engine][INFO] - Replication 18/50  (seed=59)
[2026-03-23 11:05:31,893][gibbsq.engines.numpy_engine][INFO] -   -> 56,128 arrivals, 56,114 departures, final Q_total = 14
[2026-03-23 11:05:31,893][gibbsq.engines.numpy_engine][INFO] - Replication 19/50  (seed=60)
[2026-03-23 11:05:40,435][gibbsq.engines.numpy_engine][INFO] -   -> 55,771 arrivals, 55,751 departures, final Q_total = 20
[2026-03-23 11:05:40,436][gibbsq.engines.numpy_engine][INFO] - Replication 20/50  (seed=61)
[2026-03-23 11:05:49,307][gibbsq.engines.numpy_engine][INFO] -   -> 56,235 arrivals, 56,219 departures, final Q_total = 16
[2026-03-23 11:05:49,307][gibbsq.engines.numpy_engine][INFO] - Replication 21/50  (seed=62)
[2026-03-23 11:05:57,655][gibbsq.engines.numpy_engine][INFO] -   -> 55,821 arrivals, 55,805 departures, final Q_total = 16
[2026-03-23 11:05:57,656][gibbsq.engines.numpy_engine][INFO] - Replication 22/50  (seed=63)
[2026-03-23 11:06:07,586][gibbsq.engines.numpy_engine][INFO] -   -> 55,632 arrivals, 55,619 departures, final Q_total = 13
[2026-03-23 11:06:07,586][gibbsq.engines.numpy_engine][INFO] - Replication 23/50  (seed=64)
[2026-03-23 11:06:16,105][gibbsq.engines.numpy_engine][INFO] -   -> 56,359 arrivals, 56,349 departures, final Q_total = 10
[2026-03-23 11:06:16,106][gibbsq.engines.numpy_engine][INFO] - Replication 24/50  (seed=65)
[2026-03-23 11:06:25,980][gibbsq.engines.numpy_engine][INFO] -   -> 55,839 arrivals, 55,828 departures, final Q_total = 11
[2026-03-23 11:06:25,981][gibbsq.engines.numpy_engine][INFO] - Replication 25/50  (seed=66)
[2026-03-23 11:06:34,402][gibbsq.engines.numpy_engine][INFO] -   -> 55,793 arrivals, 55,781 departures, final Q_total = 12
[2026-03-23 11:06:34,402][gibbsq.engines.numpy_engine][INFO] - Replication 26/50  (seed=67)
[2026-03-23 11:06:42,746][gibbsq.engines.numpy_engine][INFO] -   -> 55,946 arrivals, 55,935 departures, final Q_total = 11
[2026-03-23 11:06:42,747][gibbsq.engines.numpy_engine][INFO] - Replication 27/50  (seed=68)
[2026-03-23 11:06:52,415][gibbsq.engines.numpy_engine][INFO] -   -> 56,295 arrivals, 56,282 departures, final Q_total = 13
[2026-03-23 11:06:52,415][gibbsq.engines.numpy_engine][INFO] - Replication 28/50  (seed=69)
[2026-03-23 11:07:00,918][gibbsq.engines.numpy_engine][INFO] -   -> 55,361 arrivals, 55,339 departures, final Q_total = 22
[2026-03-23 11:07:00,919][gibbsq.engines.numpy_engine][INFO] - Replication 29/50  (seed=70)
[2026-03-23 11:07:09,198][gibbsq.engines.numpy_engine][INFO] -   -> 56,479 arrivals, 56,469 departures, final Q_total = 10
[2026-03-23 11:07:09,199][gibbsq.engines.numpy_engine][INFO] - Replication 30/50  (seed=71)
[2026-03-23 11:07:17,165][gibbsq.engines.numpy_engine][INFO] -   -> 55,980 arrivals, 55,970 departures, final Q_total = 10
[2026-03-23 11:07:17,165][gibbsq.engines.numpy_engine][INFO] - Replication 31/50  (seed=72)
[2026-03-23 11:07:26,175][gibbsq.engines.numpy_engine][INFO] -   -> 56,051 arrivals, 56,028 departures, final Q_total = 23
[2026-03-23 11:07:26,175][gibbsq.engines.numpy_engine][INFO] - Replication 32/50  (seed=73)
[2026-03-23 11:07:34,121][gibbsq.engines.numpy_engine][INFO] -   -> 56,027 arrivals, 56,006 departures, final Q_total = 21
[2026-03-23 11:07:34,121][gibbsq.engines.numpy_engine][INFO] - Replication 33/50  (seed=74)
[2026-03-23 11:07:42,586][gibbsq.engines.numpy_engine][INFO] -   -> 55,887 arrivals, 55,874 departures, final Q_total = 13
[2026-03-23 11:07:42,587][gibbsq.engines.numpy_engine][INFO] - Replication 34/50  (seed=75)
[2026-03-23 11:07:50,550][gibbsq.engines.numpy_engine][INFO] -   -> 56,211 arrivals, 56,195 departures, final Q_total = 16
[2026-03-23 11:07:50,551][gibbsq.engines.numpy_engine][INFO] - Replication 35/50  (seed=76)
[2026-03-23 11:07:59,735][gibbsq.engines.numpy_engine][INFO] -   -> 55,909 arrivals, 55,895 departures, final Q_total = 14
[2026-03-23 11:07:59,736][gibbsq.engines.numpy_engine][INFO] - Replication 36/50  (seed=77)
[2026-03-23 11:08:07,878][gibbsq.engines.numpy_engine][INFO] -   -> 55,633 arrivals, 55,622 departures, final Q_total = 11
[2026-03-23 11:08:07,879][gibbsq.engines.numpy_engine][INFO] - Replication 37/50  (seed=78)
[2026-03-23 11:08:16,191][gibbsq.engines.numpy_engine][INFO] -   -> 55,963 arrivals, 55,944 departures, final Q_total = 19
[2026-03-23 11:08:16,192][gibbsq.engines.numpy_engine][INFO] - Replication 38/50  (seed=79)
[2026-03-23 11:08:24,371][gibbsq.engines.numpy_engine][INFO] -   -> 56,083 arrivals, 56,061 departures, final Q_total = 22
[2026-03-23 11:08:24,371][gibbsq.engines.numpy_engine][INFO] - Replication 39/50  (seed=80)
[2026-03-23 11:08:32,334][gibbsq.engines.numpy_engine][INFO] -   -> 55,631 arrivals, 55,616 departures, final Q_total = 15
[2026-03-23 11:08:32,335][gibbsq.engines.numpy_engine][INFO] - Replication 40/50  (seed=81)
[2026-03-23 11:08:40,590][gibbsq.engines.numpy_engine][INFO] -   -> 55,999 arrivals, 55,981 departures, final Q_total = 18
[2026-03-23 11:08:40,590][gibbsq.engines.numpy_engine][INFO] - Replication 41/50  (seed=82)
[2026-03-23 11:08:49,074][gibbsq.engines.numpy_engine][INFO] -   -> 56,425 arrivals, 56,408 departures, final Q_total = 17
[2026-03-23 11:08:49,074][gibbsq.engines.numpy_engine][INFO] - Replication 42/50  (seed=83)
[2026-03-23 11:08:57,503][gibbsq.engines.numpy_engine][INFO] -   -> 55,636 arrivals, 55,611 departures, final Q_total = 25
[2026-03-23 11:08:57,503][gibbsq.engines.numpy_engine][INFO] - Replication 43/50  (seed=84)
[2026-03-23 11:09:05,709][gibbsq.engines.numpy_engine][INFO] -   -> 55,881 arrivals, 55,858 departures, final Q_total = 23
[2026-03-23 11:09:05,710][gibbsq.engines.numpy_engine][INFO] - Replication 44/50  (seed=85)
[2026-03-23 11:09:14,071][gibbsq.engines.numpy_engine][INFO] -   -> 55,798 arrivals, 55,783 departures, final Q_total = 15
[2026-03-23 11:09:14,072][gibbsq.engines.numpy_engine][INFO] - Replication 45/50  (seed=86)
[2026-03-23 11:09:22,686][gibbsq.engines.numpy_engine][INFO] -   -> 55,792 arrivals, 55,784 departures, final Q_total = 8
[2026-03-23 11:09:22,687][gibbsq.engines.numpy_engine][INFO] - Replication 46/50  (seed=87)
[2026-03-23 11:09:31,393][gibbsq.engines.numpy_engine][INFO] -   -> 55,759 arrivals, 55,750 departures, final Q_total = 9
[2026-03-23 11:09:31,393][gibbsq.engines.numpy_engine][INFO] - Replication 47/50  (seed=88)
[2026-03-23 11:09:39,513][gibbsq.engines.numpy_engine][INFO] -   -> 55,974 arrivals, 55,966 departures, final Q_total = 8
[2026-03-23 11:09:39,513][gibbsq.engines.numpy_engine][INFO] - Replication 48/50  (seed=89)
[2026-03-23 11:09:47,719][gibbsq.engines.numpy_engine][INFO] -   -> 56,259 arrivals, 56,251 departures, final Q_total = 8
[2026-03-23 11:09:47,719][gibbsq.engines.numpy_engine][INFO] - Replication 49/50  (seed=90)
[2026-03-23 11:09:57,445][gibbsq.engines.numpy_engine][INFO] -   -> 55,871 arrivals, 55,850 departures, final Q_total = 21
[2026-03-23 11:09:57,446][gibbsq.engines.numpy_engine][INFO] - Replication 50/50  (seed=91)
[2026-03-23 11:10:06,283][gibbsq.engines.numpy_engine][INFO] -   -> 56,159 arrivals, 56,140 departures, final Q_total = 19
[2026-03-23 11:10:06,331][__main__][INFO] -   E[Q_total] = 15.3433 ± 0.0416
[2026-03-23 11:10:06,333][__main__][INFO] - Evaluating Tier 3: GibbsQ-Sojourn (alpha=10.0)...
[2026-03-23 11:10:06,334][gibbsq.engines.numpy_engine][INFO] - Replication 1/50  (seed=42)
[2026-03-23 11:10:14,736][gibbsq.engines.numpy_engine][INFO] -   -> 55,905 arrivals, 55,895 departures, final Q_total = 10
[2026-03-23 11:10:14,737][gibbsq.engines.numpy_engine][INFO] - Replication 2/50  (seed=43)
[2026-03-23 11:10:23,448][gibbsq.engines.numpy_engine][INFO] -   -> 56,213 arrivals, 56,198 departures, final Q_total = 15
[2026-03-23 11:10:23,449][gibbsq.engines.numpy_engine][INFO] - Replication 3/50  (seed=44)
[2026-03-23 11:10:31,901][gibbsq.engines.numpy_engine][INFO] -   -> 55,900 arrivals, 55,888 departures, final Q_total = 12
[2026-03-23 11:10:31,901][gibbsq.engines.numpy_engine][INFO] - Replication 4/50  (seed=45)
[2026-03-23 11:10:40,540][gibbsq.engines.numpy_engine][INFO] -   -> 55,586 arrivals, 55,576 departures, final Q_total = 10
[2026-03-23 11:10:40,540][gibbsq.engines.numpy_engine][INFO] - Replication 5/50  (seed=46)
[2026-03-23 11:10:48,760][gibbsq.engines.numpy_engine][INFO] -   -> 56,174 arrivals, 56,161 departures, final Q_total = 13
[2026-03-23 11:10:48,760][gibbsq.engines.numpy_engine][INFO] - Replication 6/50  (seed=47)
[2026-03-23 11:10:57,210][gibbsq.engines.numpy_engine][INFO] -   -> 55,951 arrivals, 55,923 departures, final Q_total = 28
[2026-03-23 11:10:57,211][gibbsq.engines.numpy_engine][INFO] - Replication 7/50  (seed=48)
[2026-03-23 11:11:05,289][gibbsq.engines.numpy_engine][INFO] -   -> 55,674 arrivals, 55,660 departures, final Q_total = 14
[2026-03-23 11:11:05,290][gibbsq.engines.numpy_engine][INFO] - Replication 8/50  (seed=49)
[2026-03-23 11:11:13,641][gibbsq.engines.numpy_engine][INFO] -   -> 56,095 arrivals, 56,085 departures, final Q_total = 10
[2026-03-23 11:11:13,641][gibbsq.engines.numpy_engine][INFO] - Replication 9/50  (seed=50)
[2026-03-23 11:11:23,675][gibbsq.engines.numpy_engine][INFO] -   -> 55,570 arrivals, 55,552 departures, final Q_total = 18
[2026-03-23 11:11:23,676][gibbsq.engines.numpy_engine][INFO] - Replication 10/50  (seed=51)
[2026-03-23 11:11:34,358][gibbsq.engines.numpy_engine][INFO] -   -> 55,796 arrivals, 55,792 departures, final Q_total = 4
[2026-03-23 11:11:34,358][gibbsq.engines.numpy_engine][INFO] - Replication 11/50  (seed=52)
[2026-03-23 11:11:43,567][gibbsq.engines.numpy_engine][INFO] -   -> 56,266 arrivals, 56,253 departures, final Q_total = 13
[2026-03-23 11:11:43,567][gibbsq.engines.numpy_engine][INFO] - Replication 12/50  (seed=53)
[2026-03-23 11:11:52,153][gibbsq.engines.numpy_engine][INFO] -   -> 56,216 arrivals, 56,207 departures, final Q_total = 9
[2026-03-23 11:11:52,153][gibbsq.engines.numpy_engine][INFO] - Replication 13/50  (seed=54)
[2026-03-23 11:12:00,299][gibbsq.engines.numpy_engine][INFO] -   -> 56,168 arrivals, 56,155 departures, final Q_total = 13
[2026-03-23 11:12:00,299][gibbsq.engines.numpy_engine][INFO] - Replication 14/50  (seed=55)
[2026-03-23 11:12:08,300][gibbsq.engines.numpy_engine][INFO] -   -> 55,640 arrivals, 55,628 departures, final Q_total = 12
[2026-03-23 11:12:08,301][gibbsq.engines.numpy_engine][INFO] - Replication 15/50  (seed=56)
[2026-03-23 11:12:16,639][gibbsq.engines.numpy_engine][INFO] -   -> 56,119 arrivals, 56,092 departures, final Q_total = 27
[2026-03-23 11:12:16,639][gibbsq.engines.numpy_engine][INFO] - Replication 16/50  (seed=57)
[2026-03-23 11:12:25,133][gibbsq.engines.numpy_engine][INFO] -   -> 55,591 arrivals, 55,584 departures, final Q_total = 7
[2026-03-23 11:12:25,133][gibbsq.engines.numpy_engine][INFO] - Replication 17/50  (seed=58)
[2026-03-23 11:12:33,608][gibbsq.engines.numpy_engine][INFO] -   -> 55,717 arrivals, 55,707 departures, final Q_total = 10
[2026-03-23 11:12:33,608][gibbsq.engines.numpy_engine][INFO] - Replication 18/50  (seed=59)
[2026-03-23 11:12:41,831][gibbsq.engines.numpy_engine][INFO] -   -> 56,133 arrivals, 56,126 departures, final Q_total = 7
[2026-03-23 11:12:41,832][gibbsq.engines.numpy_engine][INFO] - Replication 19/50  (seed=60)
[2026-03-23 11:12:50,153][gibbsq.engines.numpy_engine][INFO] -   -> 55,731 arrivals, 55,712 departures, final Q_total = 19
[2026-03-23 11:12:50,154][gibbsq.engines.numpy_engine][INFO] - Replication 20/50  (seed=61)
[2026-03-23 11:12:58,480][gibbsq.engines.numpy_engine][INFO] -   -> 56,175 arrivals, 56,167 departures, final Q_total = 8
[2026-03-23 11:12:58,481][gibbsq.engines.numpy_engine][INFO] - Replication 21/50  (seed=62)
[2026-03-23 11:13:06,850][gibbsq.engines.numpy_engine][INFO] -   -> 55,789 arrivals, 55,777 departures, final Q_total = 12
[2026-03-23 11:13:06,851][gibbsq.engines.numpy_engine][INFO] - Replication 22/50  (seed=63)
[2026-03-23 11:13:14,984][gibbsq.engines.numpy_engine][INFO] -   -> 55,682 arrivals, 55,665 departures, final Q_total = 17
[2026-03-23 11:13:14,985][gibbsq.engines.numpy_engine][INFO] - Replication 23/50  (seed=64)
[2026-03-23 11:13:23,048][gibbsq.engines.numpy_engine][INFO] -   -> 56,467 arrivals, 56,461 departures, final Q_total = 6
[2026-03-23 11:13:23,049][gibbsq.engines.numpy_engine][INFO] - Replication 24/50  (seed=65)
[2026-03-23 11:13:31,676][gibbsq.engines.numpy_engine][INFO] -   -> 55,788 arrivals, 55,782 departures, final Q_total = 6
[2026-03-23 11:13:31,677][gibbsq.engines.numpy_engine][INFO] - Replication 25/50  (seed=66)
[2026-03-23 11:13:40,037][gibbsq.engines.numpy_engine][INFO] -   -> 55,821 arrivals, 55,812 departures, final Q_total = 9
[2026-03-23 11:13:40,037][gibbsq.engines.numpy_engine][INFO] - Replication 26/50  (seed=67)
[2026-03-23 11:13:47,807][gibbsq.engines.numpy_engine][INFO] -   -> 56,004 arrivals, 56,000 departures, final Q_total = 4
[2026-03-23 11:13:47,808][gibbsq.engines.numpy_engine][INFO] - Replication 27/50  (seed=68)
[2026-03-23 11:13:56,562][gibbsq.engines.numpy_engine][INFO] -   -> 56,180 arrivals, 56,172 departures, final Q_total = 8
[2026-03-23 11:13:56,563][gibbsq.engines.numpy_engine][INFO] - Replication 28/50  (seed=69)
[2026-03-23 11:14:04,936][gibbsq.engines.numpy_engine][INFO] -   -> 55,442 arrivals, 55,423 departures, final Q_total = 19
[2026-03-23 11:14:04,936][gibbsq.engines.numpy_engine][INFO] - Replication 29/50  (seed=70)
[2026-03-23 11:14:14,212][gibbsq.engines.numpy_engine][INFO] -   -> 56,463 arrivals, 56,454 departures, final Q_total = 9
[2026-03-23 11:14:14,213][gibbsq.engines.numpy_engine][INFO] - Replication 30/50  (seed=71)
[2026-03-23 11:14:22,818][gibbsq.engines.numpy_engine][INFO] -   -> 56,067 arrivals, 56,061 departures, final Q_total = 6
[2026-03-23 11:14:22,819][gibbsq.engines.numpy_engine][INFO] - Replication 31/50  (seed=72)
[2026-03-23 11:14:32,632][gibbsq.engines.numpy_engine][INFO] -   -> 56,057 arrivals, 56,039 departures, final Q_total = 18
[2026-03-23 11:14:32,632][gibbsq.engines.numpy_engine][INFO] - Replication 32/50  (seed=73)
[2026-03-23 11:14:40,556][gibbsq.engines.numpy_engine][INFO] -   -> 56,073 arrivals, 56,062 departures, final Q_total = 11
[2026-03-23 11:14:40,556][gibbsq.engines.numpy_engine][INFO] - Replication 33/50  (seed=74)
[2026-03-23 11:14:48,807][gibbsq.engines.numpy_engine][INFO] -   -> 55,885 arrivals, 55,875 departures, final Q_total = 10
[2026-03-23 11:14:48,808][gibbsq.engines.numpy_engine][INFO] - Replication 34/50  (seed=75)
[2026-03-23 11:14:56,762][gibbsq.engines.numpy_engine][INFO] -   -> 56,172 arrivals, 56,155 departures, final Q_total = 17
[2026-03-23 11:14:56,763][gibbsq.engines.numpy_engine][INFO] - Replication 35/50  (seed=76)
[2026-03-23 11:15:04,673][gibbsq.engines.numpy_engine][INFO] -   -> 55,826 arrivals, 55,816 departures, final Q_total = 10
[2026-03-23 11:15:04,673][gibbsq.engines.numpy_engine][INFO] - Replication 36/50  (seed=77)
[2026-03-23 11:15:13,167][gibbsq.engines.numpy_engine][INFO] -   -> 55,608 arrivals, 55,598 departures, final Q_total = 10
[2026-03-23 11:15:13,167][gibbsq.engines.numpy_engine][INFO] - Replication 37/50  (seed=78)
[2026-03-23 11:15:21,791][gibbsq.engines.numpy_engine][INFO] -   -> 55,975 arrivals, 55,958 departures, final Q_total = 17
[2026-03-23 11:15:21,792][gibbsq.engines.numpy_engine][INFO] - Replication 38/50  (seed=79)
[2026-03-23 11:15:30,088][gibbsq.engines.numpy_engine][INFO] -   -> 56,162 arrivals, 56,155 departures, final Q_total = 7
[2026-03-23 11:15:30,088][gibbsq.engines.numpy_engine][INFO] - Replication 39/50  (seed=80)
[2026-03-23 11:15:38,147][gibbsq.engines.numpy_engine][INFO] -   -> 55,564 arrivals, 55,549 departures, final Q_total = 15
[2026-03-23 11:15:38,148][gibbsq.engines.numpy_engine][INFO] - Replication 40/50  (seed=81)
[2026-03-23 11:15:46,654][gibbsq.engines.numpy_engine][INFO] -   -> 55,923 arrivals, 55,909 departures, final Q_total = 14
[2026-03-23 11:15:46,655][gibbsq.engines.numpy_engine][INFO] - Replication 41/50  (seed=82)
[2026-03-23 11:15:55,122][gibbsq.engines.numpy_engine][INFO] -   -> 56,494 arrivals, 56,479 departures, final Q_total = 15
[2026-03-23 11:15:55,123][gibbsq.engines.numpy_engine][INFO] - Replication 42/50  (seed=83)
[2026-03-23 11:16:03,608][gibbsq.engines.numpy_engine][INFO] -   -> 55,702 arrivals, 55,681 departures, final Q_total = 21
[2026-03-23 11:16:03,609][gibbsq.engines.numpy_engine][INFO] - Replication 43/50  (seed=84)
[2026-03-23 11:16:12,902][gibbsq.engines.numpy_engine][INFO] -   -> 55,855 arrivals, 55,849 departures, final Q_total = 6
[2026-03-23 11:16:12,902][gibbsq.engines.numpy_engine][INFO] - Replication 44/50  (seed=85)
[2026-03-23 11:16:21,138][gibbsq.engines.numpy_engine][INFO] -   -> 55,889 arrivals, 55,876 departures, final Q_total = 13
[2026-03-23 11:16:21,138][gibbsq.engines.numpy_engine][INFO] - Replication 45/50  (seed=86)
[2026-03-23 11:16:29,374][gibbsq.engines.numpy_engine][INFO] -   -> 55,886 arrivals, 55,871 departures, final Q_total = 15
[2026-03-23 11:16:29,374][gibbsq.engines.numpy_engine][INFO] - Replication 46/50  (seed=87)
[2026-03-23 11:16:37,312][gibbsq.engines.numpy_engine][INFO] -   -> 55,806 arrivals, 55,796 departures, final Q_total = 10
[2026-03-23 11:16:37,313][gibbsq.engines.numpy_engine][INFO] - Replication 47/50  (seed=88)
[2026-03-23 11:16:45,372][gibbsq.engines.numpy_engine][INFO] -   -> 55,929 arrivals, 55,919 departures, final Q_total = 10
[2026-03-23 11:16:45,372][gibbsq.engines.numpy_engine][INFO] - Replication 48/50  (seed=89)
[2026-03-23 11:16:53,467][gibbsq.engines.numpy_engine][INFO] -   -> 56,308 arrivals, 56,292 departures, final Q_total = 16
[2026-03-23 11:16:53,468][gibbsq.engines.numpy_engine][INFO] - Replication 49/50  (seed=90)
[2026-03-23 11:17:02,221][gibbsq.engines.numpy_engine][INFO] -   -> 55,854 arrivals, 55,839 departures, final Q_total = 15
[2026-03-23 11:17:02,221][gibbsq.engines.numpy_engine][INFO] - Replication 50/50  (seed=91)
[2026-03-23 11:17:10,855][gibbsq.engines.numpy_engine][INFO] -   -> 56,147 arrivals, 56,135 departures, final Q_total = 12
[2026-03-23 11:17:10,884][__main__][INFO] -   E[Q_total] = 11.1548 ± 0.0378
[2026-03-23 11:17:10,885][__main__][INFO] - Evaluating Tier 3: GibbsQ-Sojourn (alpha=opt)...
[2026-03-23 11:17:10,886][gibbsq.engines.numpy_engine][INFO] - Replication 1/50  (seed=42)
[2026-03-23 11:17:19,522][gibbsq.engines.numpy_engine][INFO] -   -> 55,968 arrivals, 55,959 departures, final Q_total = 9
[2026-03-23 11:17:19,523][gibbsq.engines.numpy_engine][INFO] - Replication 2/50  (seed=43)
[2026-03-23 11:17:28,283][gibbsq.engines.numpy_engine][INFO] -   -> 56,201 arrivals, 56,180 departures, final Q_total = 21
[2026-03-23 11:17:28,284][gibbsq.engines.numpy_engine][INFO] - Replication 3/50  (seed=44)
[2026-03-23 11:17:36,558][gibbsq.engines.numpy_engine][INFO] -   -> 55,896 arrivals, 55,872 departures, final Q_total = 24
[2026-03-23 11:17:36,559][gibbsq.engines.numpy_engine][INFO] - Replication 4/50  (seed=45)
[2026-03-23 11:17:44,515][gibbsq.engines.numpy_engine][INFO] -   -> 55,552 arrivals, 55,536 departures, final Q_total = 16
[2026-03-23 11:17:44,515][gibbsq.engines.numpy_engine][INFO] - Replication 5/50  (seed=46)
[2026-03-23 11:17:52,667][gibbsq.engines.numpy_engine][INFO] -   -> 56,184 arrivals, 56,176 departures, final Q_total = 8
[2026-03-23 11:17:52,667][gibbsq.engines.numpy_engine][INFO] - Replication 6/50  (seed=47)
[2026-03-23 11:18:02,386][gibbsq.engines.numpy_engine][INFO] -   -> 55,917 arrivals, 55,904 departures, final Q_total = 13
[2026-03-23 11:18:02,386][gibbsq.engines.numpy_engine][INFO] - Replication 7/50  (seed=48)
[2026-03-23 11:18:10,785][gibbsq.engines.numpy_engine][INFO] -   -> 55,628 arrivals, 55,605 departures, final Q_total = 23
[2026-03-23 11:18:10,786][gibbsq.engines.numpy_engine][INFO] - Replication 8/50  (seed=49)
[2026-03-23 11:18:20,177][gibbsq.engines.numpy_engine][INFO] -   -> 56,056 arrivals, 56,047 departures, final Q_total = 9
[2026-03-23 11:18:20,177][gibbsq.engines.numpy_engine][INFO] - Replication 9/50  (seed=50)
[2026-03-23 11:18:28,052][gibbsq.engines.numpy_engine][INFO] -   -> 55,506 arrivals, 55,489 departures, final Q_total = 17
[2026-03-23 11:18:28,052][gibbsq.engines.numpy_engine][INFO] - Replication 10/50  (seed=51)
[2026-03-23 11:18:36,393][gibbsq.engines.numpy_engine][INFO] -   -> 55,833 arrivals, 55,825 departures, final Q_total = 8
[2026-03-23 11:18:36,394][gibbsq.engines.numpy_engine][INFO] - Replication 11/50  (seed=52)
[2026-03-23 11:18:44,838][gibbsq.engines.numpy_engine][INFO] -   -> 56,215 arrivals, 56,205 departures, final Q_total = 10
[2026-03-23 11:18:44,839][gibbsq.engines.numpy_engine][INFO] - Replication 12/50  (seed=53)
[2026-03-23 11:18:53,177][gibbsq.engines.numpy_engine][INFO] -   -> 56,172 arrivals, 56,166 departures, final Q_total = 6
[2026-03-23 11:18:53,178][gibbsq.engines.numpy_engine][INFO] - Replication 13/50  (seed=54)
[2026-03-23 11:19:01,794][gibbsq.engines.numpy_engine][INFO] -   -> 56,136 arrivals, 56,129 departures, final Q_total = 7
[2026-03-23 11:19:01,794][gibbsq.engines.numpy_engine][INFO] - Replication 14/50  (seed=55)
[2026-03-23 11:19:09,941][gibbsq.engines.numpy_engine][INFO] -   -> 55,606 arrivals, 55,596 departures, final Q_total = 10
[2026-03-23 11:19:09,942][gibbsq.engines.numpy_engine][INFO] - Replication 15/50  (seed=56)
[2026-03-23 11:19:18,417][gibbsq.engines.numpy_engine][INFO] -   -> 56,083 arrivals, 56,062 departures, final Q_total = 21
[2026-03-23 11:19:18,418][gibbsq.engines.numpy_engine][INFO] - Replication 16/50  (seed=57)
[2026-03-23 11:19:26,440][gibbsq.engines.numpy_engine][INFO] -   -> 55,641 arrivals, 55,625 departures, final Q_total = 16
[2026-03-23 11:19:26,440][gibbsq.engines.numpy_engine][INFO] - Replication 17/50  (seed=58)
[2026-03-23 11:19:35,027][gibbsq.engines.numpy_engine][INFO] -   -> 55,747 arrivals, 55,730 departures, final Q_total = 17
[2026-03-23 11:19:35,027][gibbsq.engines.numpy_engine][INFO] - Replication 18/50  (seed=59)
[2026-03-23 11:19:44,966][gibbsq.engines.numpy_engine][INFO] -   -> 56,201 arrivals, 56,188 departures, final Q_total = 13
[2026-03-23 11:19:44,966][gibbsq.engines.numpy_engine][INFO] - Replication 19/50  (seed=60)
[2026-03-23 11:19:54,108][gibbsq.engines.numpy_engine][INFO] -   -> 55,887 arrivals, 55,865 departures, final Q_total = 22
[2026-03-23 11:19:54,109][gibbsq.engines.numpy_engine][INFO] - Replication 20/50  (seed=61)
[2026-03-23 11:20:02,390][gibbsq.engines.numpy_engine][INFO] -   -> 56,152 arrivals, 56,137 departures, final Q_total = 15
[2026-03-23 11:20:02,391][gibbsq.engines.numpy_engine][INFO] - Replication 21/50  (seed=62)
[2026-03-23 11:20:10,420][gibbsq.engines.numpy_engine][INFO] -   -> 55,829 arrivals, 55,818 departures, final Q_total = 11
[2026-03-23 11:20:10,420][gibbsq.engines.numpy_engine][INFO] - Replication 22/50  (seed=63)
[2026-03-23 11:20:18,782][gibbsq.engines.numpy_engine][INFO] -   -> 55,690 arrivals, 55,675 departures, final Q_total = 15
[2026-03-23 11:20:18,783][gibbsq.engines.numpy_engine][INFO] - Replication 23/50  (seed=64)
[2026-03-23 11:20:27,423][gibbsq.engines.numpy_engine][INFO] -   -> 56,410 arrivals, 56,389 departures, final Q_total = 21
[2026-03-23 11:20:27,424][gibbsq.engines.numpy_engine][INFO] - Replication 24/50  (seed=65)
[2026-03-23 11:20:35,565][gibbsq.engines.numpy_engine][INFO] -   -> 55,873 arrivals, 55,867 departures, final Q_total = 6
[2026-03-23 11:20:35,566][gibbsq.engines.numpy_engine][INFO] - Replication 25/50  (seed=66)
[2026-03-23 11:20:43,520][gibbsq.engines.numpy_engine][INFO] -   -> 55,792 arrivals, 55,783 departures, final Q_total = 9
[2026-03-23 11:20:43,521][gibbsq.engines.numpy_engine][INFO] - Replication 26/50  (seed=67)
[2026-03-23 11:20:51,987][gibbsq.engines.numpy_engine][INFO] -   -> 56,030 arrivals, 56,021 departures, final Q_total = 9
[2026-03-23 11:20:51,988][gibbsq.engines.numpy_engine][INFO] - Replication 27/50  (seed=68)
[2026-03-23 11:21:00,678][gibbsq.engines.numpy_engine][INFO] -   -> 56,152 arrivals, 56,138 departures, final Q_total = 14
[2026-03-23 11:21:00,679][gibbsq.engines.numpy_engine][INFO] - Replication 28/50  (seed=69)
[2026-03-23 11:21:09,067][gibbsq.engines.numpy_engine][INFO] -   -> 55,516 arrivals, 55,502 departures, final Q_total = 14
[2026-03-23 11:21:09,068][gibbsq.engines.numpy_engine][INFO] - Replication 29/50  (seed=70)
[2026-03-23 11:21:18,007][gibbsq.engines.numpy_engine][INFO] -   -> 56,448 arrivals, 56,442 departures, final Q_total = 6
[2026-03-23 11:21:18,008][gibbsq.engines.numpy_engine][INFO] - Replication 30/50  (seed=71)
[2026-03-23 11:21:26,335][gibbsq.engines.numpy_engine][INFO] -   -> 56,042 arrivals, 56,033 departures, final Q_total = 9
[2026-03-23 11:21:26,336][gibbsq.engines.numpy_engine][INFO] - Replication 31/50  (seed=72)
[2026-03-23 11:21:35,385][gibbsq.engines.numpy_engine][INFO] -   -> 56,114 arrivals, 56,109 departures, final Q_total = 5
[2026-03-23 11:21:35,386][gibbsq.engines.numpy_engine][INFO] - Replication 32/50  (seed=73)
[2026-03-23 11:21:43,714][gibbsq.engines.numpy_engine][INFO] -   -> 56,114 arrivals, 56,109 departures, final Q_total = 5
[2026-03-23 11:21:43,714][gibbsq.engines.numpy_engine][INFO] - Replication 33/50  (seed=74)
[2026-03-23 11:21:51,935][gibbsq.engines.numpy_engine][INFO] -   -> 55,852 arrivals, 55,845 departures, final Q_total = 7
[2026-03-23 11:21:51,936][gibbsq.engines.numpy_engine][INFO] - Replication 34/50  (seed=75)
[2026-03-23 11:22:00,441][gibbsq.engines.numpy_engine][INFO] -   -> 56,202 arrivals, 56,191 departures, final Q_total = 11
[2026-03-23 11:22:00,442][gibbsq.engines.numpy_engine][INFO] - Replication 35/50  (seed=76)
[2026-03-23 11:22:08,390][gibbsq.engines.numpy_engine][INFO] -   -> 55,895 arrivals, 55,877 departures, final Q_total = 18
[2026-03-23 11:22:08,390][gibbsq.engines.numpy_engine][INFO] - Replication 36/50  (seed=77)
[2026-03-23 11:22:17,066][gibbsq.engines.numpy_engine][INFO] -   -> 55,641 arrivals, 55,631 departures, final Q_total = 10
[2026-03-23 11:22:17,066][gibbsq.engines.numpy_engine][INFO] - Replication 37/50  (seed=78)
[2026-03-23 11:22:25,688][gibbsq.engines.numpy_engine][INFO] -   -> 55,948 arrivals, 55,930 departures, final Q_total = 18
[2026-03-23 11:22:25,689][gibbsq.engines.numpy_engine][INFO] - Replication 38/50  (seed=79)
[2026-03-23 11:22:34,805][gibbsq.engines.numpy_engine][INFO] -   -> 56,139 arrivals, 56,129 departures, final Q_total = 10
[2026-03-23 11:22:34,806][gibbsq.engines.numpy_engine][INFO] - Replication 39/50  (seed=80)
[2026-03-23 11:22:43,852][gibbsq.engines.numpy_engine][INFO] -   -> 55,602 arrivals, 55,588 departures, final Q_total = 14
[2026-03-23 11:22:43,852][gibbsq.engines.numpy_engine][INFO] - Replication 40/50  (seed=81)
[2026-03-23 11:22:51,843][gibbsq.engines.numpy_engine][INFO] -   -> 55,974 arrivals, 55,954 departures, final Q_total = 20
[2026-03-23 11:22:51,844][gibbsq.engines.numpy_engine][INFO] - Replication 41/50  (seed=82)
[2026-03-23 11:23:00,132][gibbsq.engines.numpy_engine][INFO] -   -> 56,444 arrivals, 56,437 departures, final Q_total = 7
[2026-03-23 11:23:00,133][gibbsq.engines.numpy_engine][INFO] - Replication 42/50  (seed=83)
[2026-03-23 11:23:08,492][gibbsq.engines.numpy_engine][INFO] -   -> 55,753 arrivals, 55,729 departures, final Q_total = 24
[2026-03-23 11:23:08,492][gibbsq.engines.numpy_engine][INFO] - Replication 43/50  (seed=84)
[2026-03-23 11:23:16,899][gibbsq.engines.numpy_engine][INFO] -   -> 55,860 arrivals, 55,848 departures, final Q_total = 12
[2026-03-23 11:23:16,899][gibbsq.engines.numpy_engine][INFO] - Replication 44/50  (seed=85)
[2026-03-23 11:23:24,902][gibbsq.engines.numpy_engine][INFO] -   -> 55,823 arrivals, 55,818 departures, final Q_total = 5
[2026-03-23 11:23:24,903][gibbsq.engines.numpy_engine][INFO] - Replication 45/50  (seed=86)
[2026-03-23 11:23:32,915][gibbsq.engines.numpy_engine][INFO] -   -> 55,800 arrivals, 55,794 departures, final Q_total = 6
[2026-03-23 11:23:32,915][gibbsq.engines.numpy_engine][INFO] - Replication 46/50  (seed=87)
[2026-03-23 11:23:41,416][gibbsq.engines.numpy_engine][INFO] -   -> 55,804 arrivals, 55,796 departures, final Q_total = 8
[2026-03-23 11:23:41,417][gibbsq.engines.numpy_engine][INFO] - Replication 47/50  (seed=88)
[2026-03-23 11:23:49,822][gibbsq.engines.numpy_engine][INFO] -   -> 55,959 arrivals, 55,948 departures, final Q_total = 11
[2026-03-23 11:23:49,823][gibbsq.engines.numpy_engine][INFO] - Replication 48/50  (seed=89)
[2026-03-23 11:23:57,958][gibbsq.engines.numpy_engine][INFO] -   -> 56,249 arrivals, 56,243 departures, final Q_total = 6
[2026-03-23 11:23:57,959][gibbsq.engines.numpy_engine][INFO] - Replication 49/50  (seed=90)
[2026-03-23 11:24:06,794][gibbsq.engines.numpy_engine][INFO] -   -> 55,842 arrivals, 55,828 departures, final Q_total = 14
[2026-03-23 11:24:06,794][gibbsq.engines.numpy_engine][INFO] - Replication 50/50  (seed=91)
[2026-03-23 11:24:15,482][gibbsq.engines.numpy_engine][INFO] -   -> 56,133 arrivals, 56,122 departures, final Q_total = 11
[2026-03-23 11:24:15,513][__main__][INFO] -   E[Q_total] = 11.6386 ± 0.0372
[2026-03-23 11:24:15,515][__main__][INFO] - Evaluating Tier 4: Proportional (mu/Lambda)...
[2026-03-23 11:24:15,516][gibbsq.engines.numpy_engine][INFO] - Replication 1/50  (seed=42)
[2026-03-23 11:24:21,812][gibbsq.engines.numpy_engine][INFO] -   -> 55,949 arrivals, 55,920 departures, final Q_total = 29
[2026-03-23 11:24:21,813][gibbsq.engines.numpy_engine][INFO] - Replication 2/50  (seed=43)
[2026-03-23 11:24:29,532][gibbsq.engines.numpy_engine][INFO] -   -> 56,148 arrivals, 56,077 departures, final Q_total = 71
[2026-03-23 11:24:29,532][gibbsq.engines.numpy_engine][INFO] - Replication 3/50  (seed=44)
[2026-03-23 11:24:35,957][gibbsq.engines.numpy_engine][INFO] -   -> 55,912 arrivals, 55,845 departures, final Q_total = 67
[2026-03-23 11:24:35,957][gibbsq.engines.numpy_engine][INFO] - Replication 4/50  (seed=45)
[2026-03-23 11:24:42,216][gibbsq.engines.numpy_engine][INFO] -   -> 55,574 arrivals, 55,531 departures, final Q_total = 43
[2026-03-23 11:24:42,216][gibbsq.engines.numpy_engine][INFO] - Replication 5/50  (seed=46)
[2026-03-23 11:24:48,781][gibbsq.engines.numpy_engine][INFO] -   -> 56,152 arrivals, 56,114 departures, final Q_total = 38
[2026-03-23 11:24:48,782][gibbsq.engines.numpy_engine][INFO] - Replication 6/50  (seed=47)
[2026-03-23 11:24:55,308][gibbsq.engines.numpy_engine][INFO] -   -> 56,013 arrivals, 55,975 departures, final Q_total = 38
[2026-03-23 11:24:55,309][gibbsq.engines.numpy_engine][INFO] - Replication 7/50  (seed=48)
[2026-03-23 11:25:01,191][gibbsq.engines.numpy_engine][INFO] -   -> 55,566 arrivals, 55,508 departures, final Q_total = 58
[2026-03-23 11:25:01,192][gibbsq.engines.numpy_engine][INFO] - Replication 8/50  (seed=49)
[2026-03-23 11:25:07,908][gibbsq.engines.numpy_engine][INFO] -   -> 56,135 arrivals, 56,111 departures, final Q_total = 24
[2026-03-23 11:25:07,908][gibbsq.engines.numpy_engine][INFO] - Replication 9/50  (seed=50)
[2026-03-23 11:25:14,131][gibbsq.engines.numpy_engine][INFO] -   -> 55,509 arrivals, 55,454 departures, final Q_total = 55
[2026-03-23 11:25:14,131][gibbsq.engines.numpy_engine][INFO] - Replication 10/50  (seed=51)
[2026-03-23 11:25:20,602][gibbsq.engines.numpy_engine][INFO] -   -> 55,821 arrivals, 55,790 departures, final Q_total = 31
[2026-03-23 11:25:20,603][gibbsq.engines.numpy_engine][INFO] - Replication 11/50  (seed=52)
[2026-03-23 11:25:26,925][gibbsq.engines.numpy_engine][INFO] -   -> 56,204 arrivals, 56,186 departures, final Q_total = 18
[2026-03-23 11:25:26,925][gibbsq.engines.numpy_engine][INFO] - Replication 12/50  (seed=53)
[2026-03-23 11:25:33,159][gibbsq.engines.numpy_engine][INFO] -   -> 56,230 arrivals, 56,185 departures, final Q_total = 45
[2026-03-23 11:25:33,160][gibbsq.engines.numpy_engine][INFO] - Replication 13/50  (seed=54)
[2026-03-23 11:25:39,461][gibbsq.engines.numpy_engine][INFO] -   -> 56,135 arrivals, 56,108 departures, final Q_total = 27
[2026-03-23 11:25:39,461][gibbsq.engines.numpy_engine][INFO] - Replication 14/50  (seed=55)
[2026-03-23 11:25:45,920][gibbsq.engines.numpy_engine][INFO] -   -> 55,616 arrivals, 55,569 departures, final Q_total = 47
[2026-03-23 11:25:45,920][gibbsq.engines.numpy_engine][INFO] - Replication 15/50  (seed=56)
[2026-03-23 11:25:52,195][gibbsq.engines.numpy_engine][INFO] -   -> 56,168 arrivals, 56,092 departures, final Q_total = 76
[2026-03-23 11:25:52,196][gibbsq.engines.numpy_engine][INFO] - Replication 16/50  (seed=57)
[2026-03-23 11:25:58,875][gibbsq.engines.numpy_engine][INFO] -   -> 55,685 arrivals, 55,627 departures, final Q_total = 58
[2026-03-23 11:25:58,876][gibbsq.engines.numpy_engine][INFO] - Replication 17/50  (seed=58)
[2026-03-23 11:26:07,361][gibbsq.engines.numpy_engine][INFO] -   -> 55,868 arrivals, 55,802 departures, final Q_total = 66
[2026-03-23 11:26:07,362][gibbsq.engines.numpy_engine][INFO] - Replication 18/50  (seed=59)
[2026-03-23 11:26:14,431][gibbsq.engines.numpy_engine][INFO] -   -> 56,145 arrivals, 56,099 departures, final Q_total = 46
[2026-03-23 11:26:14,431][gibbsq.engines.numpy_engine][INFO] - Replication 19/50  (seed=60)
[2026-03-23 11:26:20,811][gibbsq.engines.numpy_engine][INFO] -   -> 55,697 arrivals, 55,666 departures, final Q_total = 31
[2026-03-23 11:26:20,811][gibbsq.engines.numpy_engine][INFO] - Replication 20/50  (seed=61)
[2026-03-23 11:26:27,252][gibbsq.engines.numpy_engine][INFO] -   -> 56,094 arrivals, 56,005 departures, final Q_total = 89
[2026-03-23 11:26:27,252][gibbsq.engines.numpy_engine][INFO] - Replication 21/50  (seed=62)
[2026-03-23 11:26:33,192][gibbsq.engines.numpy_engine][INFO] -   -> 55,792 arrivals, 55,762 departures, final Q_total = 30
[2026-03-23 11:26:33,193][gibbsq.engines.numpy_engine][INFO] - Replication 22/50  (seed=63)
[2026-03-23 11:26:39,599][gibbsq.engines.numpy_engine][INFO] -   -> 55,720 arrivals, 55,672 departures, final Q_total = 48
[2026-03-23 11:26:39,599][gibbsq.engines.numpy_engine][INFO] - Replication 23/50  (seed=64)
[2026-03-23 11:26:46,168][gibbsq.engines.numpy_engine][INFO] -   -> 56,342 arrivals, 56,303 departures, final Q_total = 39
[2026-03-23 11:26:46,168][gibbsq.engines.numpy_engine][INFO] - Replication 24/50  (seed=65)
[2026-03-23 11:26:52,440][gibbsq.engines.numpy_engine][INFO] -   -> 55,774 arrivals, 55,734 departures, final Q_total = 40
[2026-03-23 11:26:52,441][gibbsq.engines.numpy_engine][INFO] - Replication 25/50  (seed=66)
[2026-03-23 11:26:58,952][gibbsq.engines.numpy_engine][INFO] -   -> 55,866 arrivals, 55,825 departures, final Q_total = 41
[2026-03-23 11:26:58,953][gibbsq.engines.numpy_engine][INFO] - Replication 26/50  (seed=67)
[2026-03-23 11:27:05,488][gibbsq.engines.numpy_engine][INFO] -   -> 55,956 arrivals, 55,915 departures, final Q_total = 41
[2026-03-23 11:27:05,488][gibbsq.engines.numpy_engine][INFO] - Replication 27/50  (seed=68)
[2026-03-23 11:27:12,017][gibbsq.engines.numpy_engine][INFO] -   -> 56,244 arrivals, 56,216 departures, final Q_total = 28
[2026-03-23 11:27:12,018][gibbsq.engines.numpy_engine][INFO] - Replication 28/50  (seed=69)
[2026-03-23 11:27:18,450][gibbsq.engines.numpy_engine][INFO] -   -> 55,464 arrivals, 55,410 departures, final Q_total = 54
[2026-03-23 11:27:18,451][gibbsq.engines.numpy_engine][INFO] - Replication 29/50  (seed=70)
[2026-03-23 11:27:25,269][gibbsq.engines.numpy_engine][INFO] -   -> 56,564 arrivals, 56,531 departures, final Q_total = 33
[2026-03-23 11:27:25,270][gibbsq.engines.numpy_engine][INFO] - Replication 30/50  (seed=71)
[2026-03-23 11:27:31,990][gibbsq.engines.numpy_engine][INFO] -   -> 55,937 arrivals, 55,910 departures, final Q_total = 27
[2026-03-23 11:27:31,991][gibbsq.engines.numpy_engine][INFO] - Replication 31/50  (seed=72)
[2026-03-23 11:27:38,515][gibbsq.engines.numpy_engine][INFO] -   -> 56,011 arrivals, 55,977 departures, final Q_total = 34
[2026-03-23 11:27:38,516][gibbsq.engines.numpy_engine][INFO] - Replication 32/50  (seed=73)
[2026-03-23 11:27:45,138][gibbsq.engines.numpy_engine][INFO] -   -> 56,008 arrivals, 55,978 departures, final Q_total = 30
[2026-03-23 11:27:45,139][gibbsq.engines.numpy_engine][INFO] - Replication 33/50  (seed=74)
[2026-03-23 11:27:51,770][gibbsq.engines.numpy_engine][INFO] -   -> 55,861 arrivals, 55,842 departures, final Q_total = 19
[2026-03-23 11:27:51,770][gibbsq.engines.numpy_engine][INFO] - Replication 34/50  (seed=75)
[2026-03-23 11:27:58,275][gibbsq.engines.numpy_engine][INFO] -   -> 56,171 arrivals, 56,141 departures, final Q_total = 30
[2026-03-23 11:27:58,276][gibbsq.engines.numpy_engine][INFO] - Replication 35/50  (seed=76)
[2026-03-23 11:28:05,052][gibbsq.engines.numpy_engine][INFO] -   -> 55,929 arrivals, 55,897 departures, final Q_total = 32
[2026-03-23 11:28:05,052][gibbsq.engines.numpy_engine][INFO] - Replication 36/50  (seed=77)
[2026-03-23 11:28:11,058][gibbsq.engines.numpy_engine][INFO] -   -> 55,651 arrivals, 55,595 departures, final Q_total = 56
[2026-03-23 11:28:11,058][gibbsq.engines.numpy_engine][INFO] - Replication 37/50  (seed=78)
[2026-03-23 11:28:17,689][gibbsq.engines.numpy_engine][INFO] -   -> 55,985 arrivals, 55,886 departures, final Q_total = 99
[2026-03-23 11:28:17,689][gibbsq.engines.numpy_engine][INFO] - Replication 38/50  (seed=79)
[2026-03-23 11:28:25,180][gibbsq.engines.numpy_engine][INFO] -   -> 56,113 arrivals, 56,070 departures, final Q_total = 43
[2026-03-23 11:28:25,180][gibbsq.engines.numpy_engine][INFO] - Replication 39/50  (seed=80)
[2026-03-23 11:28:31,816][gibbsq.engines.numpy_engine][INFO] -   -> 55,698 arrivals, 55,643 departures, final Q_total = 55
[2026-03-23 11:28:31,816][gibbsq.engines.numpy_engine][INFO] - Replication 40/50  (seed=81)
[2026-03-23 11:28:39,022][gibbsq.engines.numpy_engine][INFO] -   -> 55,972 arrivals, 55,933 departures, final Q_total = 39
[2026-03-23 11:28:39,023][gibbsq.engines.numpy_engine][INFO] - Replication 41/50  (seed=82)
[2026-03-23 11:28:47,336][gibbsq.engines.numpy_engine][INFO] -   -> 56,425 arrivals, 56,397 departures, final Q_total = 28
[2026-03-23 11:28:47,337][gibbsq.engines.numpy_engine][INFO] - Replication 42/50  (seed=83)
[2026-03-23 11:28:54,122][gibbsq.engines.numpy_engine][INFO] -   -> 55,696 arrivals, 55,652 departures, final Q_total = 44
[2026-03-23 11:28:54,123][gibbsq.engines.numpy_engine][INFO] - Replication 43/50  (seed=84)
[2026-03-23 11:29:01,463][gibbsq.engines.numpy_engine][INFO] -   -> 55,858 arrivals, 55,837 departures, final Q_total = 21
[2026-03-23 11:29:01,463][gibbsq.engines.numpy_engine][INFO] - Replication 44/50  (seed=85)
[2026-03-23 11:29:08,283][gibbsq.engines.numpy_engine][INFO] -   -> 55,936 arrivals, 55,900 departures, final Q_total = 36
[2026-03-23 11:29:08,284][gibbsq.engines.numpy_engine][INFO] - Replication 45/50  (seed=86)
[2026-03-23 11:29:15,229][gibbsq.engines.numpy_engine][INFO] -   -> 55,912 arrivals, 55,884 departures, final Q_total = 28
[2026-03-23 11:29:15,229][gibbsq.engines.numpy_engine][INFO] - Replication 46/50  (seed=87)
[2026-03-23 11:29:21,994][gibbsq.engines.numpy_engine][INFO] -   -> 55,791 arrivals, 55,758 departures, final Q_total = 33
[2026-03-23 11:29:21,995][gibbsq.engines.numpy_engine][INFO] - Replication 47/50  (seed=88)
[2026-03-23 11:29:34,209][gibbsq.engines.numpy_engine][INFO] -   -> 55,916 arrivals, 55,898 departures, final Q_total = 18
[2026-03-23 11:29:34,209][gibbsq.engines.numpy_engine][INFO] - Replication 48/50  (seed=89)
[2026-03-23 11:29:52,459][gibbsq.engines.numpy_engine][INFO] -   -> 56,216 arrivals, 56,167 departures, final Q_total = 49
[2026-03-23 11:29:52,483][gibbsq.engines.numpy_engine][INFO] - Replication 49/50  (seed=90)
[2026-03-23 11:30:13,582][gibbsq.engines.numpy_engine][INFO] -   -> 55,736 arrivals, 55,717 departures, final Q_total = 19
[2026-03-23 11:30:13,643][gibbsq.engines.numpy_engine][INFO] - Replication 50/50  (seed=91)
[2026-03-23 11:30:47,095][gibbsq.engines.numpy_engine][INFO] -   -> 56,165 arrivals, 56,134 departures, final Q_total = 31
[2026-03-23 11:30:47,277][__main__][INFO] -   E[Q_total] = 39.5373 ± 0.2991
[2026-03-23 11:30:47,293][__main__][INFO] - Evaluating Tier 4: Uniform (1/N)...
[2026-03-23 11:30:47,336][gibbsq.engines.numpy_engine][INFO] - Replication 1/50  (seed=42)
[2026-03-23 11:31:01,699][gibbsq.engines.numpy_engine][INFO] -   -> 55,891 arrivals, 49,499 departures, final Q_total = 6392
[2026-03-23 11:31:01,701][gibbsq.engines.numpy_engine][INFO] - Replication 2/50  (seed=43)
[2026-03-23 11:31:08,468][gibbsq.engines.numpy_engine][INFO] -   -> 56,213 arrivals, 49,501 departures, final Q_total = 6712
[2026-03-23 11:31:08,469][gibbsq.engines.numpy_engine][INFO] - Replication 3/50  (seed=44)
[2026-03-23 11:31:16,049][gibbsq.engines.numpy_engine][INFO] -   -> 55,826 arrivals, 49,547 departures, final Q_total = 6279
[2026-03-23 11:31:16,049][gibbsq.engines.numpy_engine][INFO] - Replication 4/50  (seed=45)
[2026-03-23 11:31:23,235][gibbsq.engines.numpy_engine][INFO] -   -> 55,627 arrivals, 49,384 departures, final Q_total = 6243
[2026-03-23 11:31:23,236][gibbsq.engines.numpy_engine][INFO] - Replication 5/50  (seed=46)
[2026-03-23 11:31:30,316][gibbsq.engines.numpy_engine][INFO] -   -> 56,178 arrivals, 49,676 departures, final Q_total = 6502
[2026-03-23 11:31:30,317][gibbsq.engines.numpy_engine][INFO] - Replication 6/50  (seed=47)
[2026-03-23 11:31:37,926][gibbsq.engines.numpy_engine][INFO] -   -> 56,005 arrivals, 49,493 departures, final Q_total = 6512
[2026-03-23 11:31:37,926][gibbsq.engines.numpy_engine][INFO] - Replication 7/50  (seed=48)
[2026-03-23 11:31:45,508][gibbsq.engines.numpy_engine][INFO] -   -> 55,622 arrivals, 49,515 departures, final Q_total = 6107
[2026-03-23 11:31:45,508][gibbsq.engines.numpy_engine][INFO] - Replication 8/50  (seed=49)
[2026-03-23 11:31:53,640][gibbsq.engines.numpy_engine][INFO] -   -> 56,120 arrivals, 49,471 departures, final Q_total = 6649
[2026-03-23 11:31:53,641][gibbsq.engines.numpy_engine][INFO] - Replication 9/50  (seed=50)
[2026-03-23 11:32:01,982][gibbsq.engines.numpy_engine][INFO] -   -> 55,597 arrivals, 49,277 departures, final Q_total = 6320
[2026-03-23 11:32:01,983][gibbsq.engines.numpy_engine][INFO] - Replication 10/50  (seed=51)
[2026-03-23 11:32:10,611][gibbsq.engines.numpy_engine][INFO] -   -> 55,942 arrivals, 49,686 departures, final Q_total = 6256
[2026-03-23 11:32:10,611][gibbsq.engines.numpy_engine][INFO] - Replication 11/50  (seed=52)
[2026-03-23 11:32:21,538][gibbsq.engines.numpy_engine][INFO] -   -> 56,284 arrivals, 49,942 departures, final Q_total = 6342
[2026-03-23 11:32:21,539][gibbsq.engines.numpy_engine][INFO] - Replication 12/50  (seed=53)
[2026-03-23 11:32:30,975][gibbsq.engines.numpy_engine][INFO] -   -> 56,372 arrivals, 49,850 departures, final Q_total = 6522
[2026-03-23 11:32:30,976][gibbsq.engines.numpy_engine][INFO] - Replication 13/50  (seed=54)
[2026-03-23 11:32:38,976][gibbsq.engines.numpy_engine][INFO] -   -> 56,121 arrivals, 49,843 departures, final Q_total = 6278
[2026-03-23 11:32:38,977][gibbsq.engines.numpy_engine][INFO] - Replication 14/50  (seed=55)
[2026-03-23 11:32:46,581][gibbsq.engines.numpy_engine][INFO] -   -> 55,553 arrivals, 49,239 departures, final Q_total = 6314
[2026-03-23 11:32:46,582][gibbsq.engines.numpy_engine][INFO] - Replication 15/50  (seed=56)
[2026-03-23 11:32:55,115][gibbsq.engines.numpy_engine][INFO] -   -> 56,070 arrivals, 49,562 departures, final Q_total = 6508
[2026-03-23 11:32:55,116][gibbsq.engines.numpy_engine][INFO] - Replication 16/50  (seed=57)
[2026-03-23 11:33:02,008][gibbsq.engines.numpy_engine][INFO] -   -> 55,565 arrivals, 49,084 departures, final Q_total = 6481
[2026-03-23 11:33:02,009][gibbsq.engines.numpy_engine][INFO] - Replication 17/50  (seed=58)
[2026-03-23 11:33:09,851][gibbsq.engines.numpy_engine][INFO] -   -> 55,862 arrivals, 49,811 departures, final Q_total = 6051
[2026-03-23 11:33:09,852][gibbsq.engines.numpy_engine][INFO] - Replication 18/50  (seed=59)
[2026-03-23 11:33:19,478][gibbsq.engines.numpy_engine][INFO] -   -> 56,098 arrivals, 49,631 departures, final Q_total = 6467
[2026-03-23 11:33:19,479][gibbsq.engines.numpy_engine][INFO] - Replication 19/50  (seed=60)
[2026-03-23 11:33:26,351][gibbsq.engines.numpy_engine][INFO] -   -> 55,770 arrivals, 49,257 departures, final Q_total = 6513
[2026-03-23 11:33:26,352][gibbsq.engines.numpy_engine][INFO] - Replication 20/50  (seed=61)
[2026-03-23 11:33:33,055][gibbsq.engines.numpy_engine][INFO] -   -> 56,191 arrivals, 49,929 departures, final Q_total = 6262
[2026-03-23 11:33:33,055][gibbsq.engines.numpy_engine][INFO] - Replication 21/50  (seed=62)
[2026-03-23 11:33:40,391][gibbsq.engines.numpy_engine][INFO] -   -> 55,822 arrivals, 49,447 departures, final Q_total = 6375
[2026-03-23 11:33:40,392][gibbsq.engines.numpy_engine][INFO] - Replication 22/50  (seed=63)
[2026-03-23 11:33:47,092][gibbsq.engines.numpy_engine][INFO] -   -> 55,714 arrivals, 49,540 departures, final Q_total = 6174
[2026-03-23 11:33:47,092][gibbsq.engines.numpy_engine][INFO] - Replication 23/50  (seed=64)
[2026-03-23 11:33:53,418][gibbsq.engines.numpy_engine][INFO] -   -> 56,309 arrivals, 49,536 departures, final Q_total = 6773
[2026-03-23 11:33:53,419][gibbsq.engines.numpy_engine][INFO] - Replication 24/50  (seed=65)
[2026-03-23 11:33:59,940][gibbsq.engines.numpy_engine][INFO] -   -> 55,757 arrivals, 49,335 departures, final Q_total = 6422
[2026-03-23 11:33:59,940][gibbsq.engines.numpy_engine][INFO] - Replication 25/50  (seed=66)
[2026-03-23 11:34:06,496][gibbsq.engines.numpy_engine][INFO] -   -> 55,901 arrivals, 49,697 departures, final Q_total = 6204
[2026-03-23 11:34:06,497][gibbsq.engines.numpy_engine][INFO] - Replication 26/50  (seed=67)
[2026-03-23 11:34:13,036][gibbsq.engines.numpy_engine][INFO] -   -> 56,022 arrivals, 49,439 departures, final Q_total = 6583
[2026-03-23 11:34:13,036][gibbsq.engines.numpy_engine][INFO] - Replication 27/50  (seed=68)
[2026-03-23 11:34:19,304][gibbsq.engines.numpy_engine][INFO] -   -> 56,165 arrivals, 49,668 departures, final Q_total = 6497
[2026-03-23 11:34:19,304][gibbsq.engines.numpy_engine][INFO] - Replication 28/50  (seed=69)
[2026-03-23 11:34:25,624][gibbsq.engines.numpy_engine][INFO] -   -> 55,509 arrivals, 49,159 departures, final Q_total = 6350
[2026-03-23 11:34:25,624][gibbsq.engines.numpy_engine][INFO] - Replication 29/50  (seed=70)
[2026-03-23 11:34:32,016][gibbsq.engines.numpy_engine][INFO] -   -> 56,432 arrivals, 50,097 departures, final Q_total = 6335
[2026-03-23 11:34:32,016][gibbsq.engines.numpy_engine][INFO] - Replication 30/50  (seed=71)
[2026-03-23 11:34:38,566][gibbsq.engines.numpy_engine][INFO] -   -> 56,013 arrivals, 49,583 departures, final Q_total = 6430
[2026-03-23 11:34:38,567][gibbsq.engines.numpy_engine][INFO] - Replication 31/50  (seed=72)
[2026-03-23 11:34:44,610][gibbsq.engines.numpy_engine][INFO] -   -> 56,084 arrivals, 49,569 departures, final Q_total = 6515
[2026-03-23 11:34:44,611][gibbsq.engines.numpy_engine][INFO] - Replication 32/50  (seed=73)
[2026-03-23 11:34:51,158][gibbsq.engines.numpy_engine][INFO] -   -> 55,989 arrivals, 49,306 departures, final Q_total = 6683
[2026-03-23 11:34:51,159][gibbsq.engines.numpy_engine][INFO] - Replication 33/50  (seed=74)
[2026-03-23 11:34:57,694][gibbsq.engines.numpy_engine][INFO] -   -> 55,768 arrivals, 49,395 departures, final Q_total = 6373
[2026-03-23 11:34:57,694][gibbsq.engines.numpy_engine][INFO] - Replication 34/50  (seed=75)
[2026-03-23 11:35:04,066][gibbsq.engines.numpy_engine][INFO] -   -> 56,295 arrivals, 49,675 departures, final Q_total = 6620
[2026-03-23 11:35:04,066][gibbsq.engines.numpy_engine][INFO] - Replication 35/50  (seed=76)
[2026-03-23 11:35:10,654][gibbsq.engines.numpy_engine][INFO] -   -> 56,025 arrivals, 49,846 departures, final Q_total = 6179
[2026-03-23 11:35:10,655][gibbsq.engines.numpy_engine][INFO] - Replication 36/50  (seed=77)
[2026-03-23 11:35:17,178][gibbsq.engines.numpy_engine][INFO] -   -> 55,671 arrivals, 49,193 departures, final Q_total = 6478
[2026-03-23 11:35:17,179][gibbsq.engines.numpy_engine][INFO] - Replication 37/50  (seed=78)
[2026-03-23 11:35:23,971][gibbsq.engines.numpy_engine][INFO] -   -> 55,989 arrivals, 49,571 departures, final Q_total = 6418
[2026-03-23 11:35:23,971][gibbsq.engines.numpy_engine][INFO] - Replication 38/50  (seed=79)
[2026-03-23 11:35:30,355][gibbsq.engines.numpy_engine][INFO] -   -> 56,064 arrivals, 49,607 departures, final Q_total = 6457
[2026-03-23 11:35:30,356][gibbsq.engines.numpy_engine][INFO] - Replication 39/50  (seed=80)
[2026-03-23 11:35:36,623][gibbsq.engines.numpy_engine][INFO] -   -> 55,613 arrivals, 49,188 departures, final Q_total = 6425
[2026-03-23 11:35:36,623][gibbsq.engines.numpy_engine][INFO] - Replication 40/50  (seed=81)
[2026-03-23 11:35:43,512][gibbsq.engines.numpy_engine][INFO] -   -> 55,909 arrivals, 49,226 departures, final Q_total = 6683
[2026-03-23 11:35:43,512][gibbsq.engines.numpy_engine][INFO] - Replication 41/50  (seed=82)
[2026-03-23 11:35:49,848][gibbsq.engines.numpy_engine][INFO] -   -> 56,412 arrivals, 49,702 departures, final Q_total = 6710
[2026-03-23 11:35:49,848][gibbsq.engines.numpy_engine][INFO] - Replication 42/50  (seed=83)
[2026-03-23 11:35:56,288][gibbsq.engines.numpy_engine][INFO] -   -> 55,672 arrivals, 49,218 departures, final Q_total = 6454
[2026-03-23 11:35:56,288][gibbsq.engines.numpy_engine][INFO] - Replication 43/50  (seed=84)
[2026-03-23 11:36:03,085][gibbsq.engines.numpy_engine][INFO] -   -> 56,065 arrivals, 49,523 departures, final Q_total = 6542
[2026-03-23 11:36:03,085][gibbsq.engines.numpy_engine][INFO] - Replication 44/50  (seed=85)
[2026-03-23 11:36:09,386][gibbsq.engines.numpy_engine][INFO] -   -> 55,769 arrivals, 49,583 departures, final Q_total = 6186
[2026-03-23 11:36:09,387][gibbsq.engines.numpy_engine][INFO] - Replication 45/50  (seed=86)
[2026-03-23 11:36:15,750][gibbsq.engines.numpy_engine][INFO] -   -> 55,837 arrivals, 49,390 departures, final Q_total = 6447
[2026-03-23 11:36:15,751][gibbsq.engines.numpy_engine][INFO] - Replication 46/50  (seed=87)
[2026-03-23 11:36:22,928][gibbsq.engines.numpy_engine][INFO] -   -> 55,778 arrivals, 49,538 departures, final Q_total = 6240
[2026-03-23 11:36:22,928][gibbsq.engines.numpy_engine][INFO] - Replication 47/50  (seed=88)
[2026-03-23 11:36:29,288][gibbsq.engines.numpy_engine][INFO] -   -> 55,983 arrivals, 49,250 departures, final Q_total = 6733
[2026-03-23 11:36:29,289][gibbsq.engines.numpy_engine][INFO] - Replication 48/50  (seed=89)
[2026-03-23 11:36:35,597][gibbsq.engines.numpy_engine][INFO] -   -> 56,305 arrivals, 49,625 departures, final Q_total = 6680
[2026-03-23 11:36:35,597][gibbsq.engines.numpy_engine][INFO] - Replication 49/50  (seed=90)
[2026-03-23 11:36:41,975][gibbsq.engines.numpy_engine][INFO] -   -> 55,711 arrivals, 49,564 departures, final Q_total = 6147
[2026-03-23 11:36:41,975][gibbsq.engines.numpy_engine][INFO] - Replication 50/50  (seed=91)
[2026-03-23 11:36:47,979][gibbsq.engines.numpy_engine][INFO] -   -> 56,190 arrivals, 49,726 departures, final Q_total = 6464
[2026-03-23 11:36:48,021][__main__][INFO] -   E[Q_total] = 3860.8230 ± 14.9805
[2026-03-23 11:36:48,023][__main__][INFO] -
============================================================
[2026-03-23 11:36:48,024][__main__][INFO] -   Parity Analysis (Corrected Criteria)
[2026-03-23 11:36:48,025][__main__][INFO] - ============================================================
[2026-03-23 11:36:48,025][__main__][INFO] - N-GibbsQ (REINFORCE) not evaluated - skipping parity analysis
[2026-03-23 11:36:49,769][__main__][INFO] - Comparison plot saved to outputs\policy_comparison\run_20260323_104909\corrected_policy_comparison.png, outputs\policy_comparison\run_20260323_104909\corrected_policy_comparison.pdf

[5/10] Running Stability Sweep...
Running: sweep
==========================================================
 Starting Experiment: sweep
 Remaining Args (Hydra Overrides): +configs=fast +experiment=stability_sweep
==========================================================
[2026-03-23 11:36:55,351][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-23 11:36:55,372:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 11:36:55,372][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 11:36:55,373][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-23 11:36:55,399][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\stability_sweep\run_20260323_113655
[2026-03-23 11:36:55,400][gibbsq.utils.logging][INFO] - [Logging] WandB offline mode.
wandb: Tracking run with wandb version 0.23.1
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in outputs\stability_sweep\run_20260323_113655\wandb\offline-run-20260323_113655-zu3m8pls
[2026-03-23 11:36:59,307][gibbsq.utils.logging][INFO] - [Logging] WandB Run Linked: run_20260323_113655 (offline)
[2026-03-23 11:36:59,311][__main__][INFO] - System: N=10, cap=14.0000 | Backend: JAX
[2026-03-23 11:36:59,313][__main__][INFO] - Grid: 10 alpha x 7 rho x 50 reps
[2026-03-23 11:36:59,314][__main__][INFO] -
------------------------------------------------------------
  rho = 0.50  (lam = 7.0000)
------------------------------------------------------------
[2026-03-23 11:37:17,772][__main__][INFO] -   alpha=  0.01 | E[Q_total]=   59.11 | OK  (1/70)
[2026-03-23 11:37:33,915][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   16.54 | OK  (2/70)
[2026-03-23 11:37:50,283][__main__][INFO] -   alpha=  0.50 | E[Q_total]=    9.64 | OK  (3/70)
[2026-03-23 11:38:06,548][__main__][INFO] -   alpha=  1.00 | E[Q_total]=    7.87 | OK  (4/70)
[2026-03-23 11:38:22,754][__main__][INFO] -   alpha=  2.00 | E[Q_total]=    6.59 | OK  (5/70)
[2026-03-23 11:38:39,110][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    5.67 | OK  (6/70)
[2026-03-23 11:38:56,207][__main__][INFO] -   alpha= 10.00 | E[Q_total]=    5.61 | OK  (7/70)
[2026-03-23 11:39:13,362][__main__][INFO] -   alpha= 20.00 | E[Q_total]=    5.61 | OK  (8/70)
[2026-03-23 11:39:30,851][__main__][INFO] -   alpha= 50.00 | E[Q_total]=    5.61 | OK  (9/70)
[2026-03-23 11:39:48,709][__main__][INFO] -   alpha=100.00 | E[Q_total]=    5.60 | OK  (10/70)
[2026-03-23 11:39:48,710][__main__][INFO] -
------------------------------------------------------------
  rho = 0.70  (lam = 9.8000)
------------------------------------------------------------
[2026-03-23 11:40:13,985][__main__][INFO] -   alpha=  0.01 | E[Q_total]=  176.35 | OK  (11/70)
[2026-03-23 11:40:33,628][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   34.32 | OK  (12/70)
[2026-03-23 11:40:53,122][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   16.64 | OK  (13/70)
[2026-03-23 11:41:12,614][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   12.99 | OK  (14/70)
[2026-03-23 11:41:32,390][__main__][INFO] -   alpha=  2.00 | E[Q_total]=   10.59 | OK  (15/70)
[2026-03-23 11:41:51,718][__main__][INFO] -   alpha=  5.00 | E[Q_total]=    8.97 | OK  (16/70)
[2026-03-23 11:42:12,304][__main__][INFO] -   alpha= 10.00 | E[Q_total]=    8.82 | OK  (17/70)
[2026-03-23 11:42:32,644][__main__][INFO] -   alpha= 20.00 | E[Q_total]=    8.81 | OK  (18/70)
[2026-03-23 11:42:52,702][__main__][INFO] -   alpha= 50.00 | E[Q_total]=    8.83 | OK  (19/70)
[2026-03-23 11:43:13,151][__main__][INFO] -   alpha=100.00 | E[Q_total]=    8.82 | OK  (20/70)
[2026-03-23 11:43:13,153][__main__][INFO] -
------------------------------------------------------------
  rho = 0.80  (lam = 11.2000)
------------------------------------------------------------
[2026-03-23 11:43:37,013][__main__][INFO] -   alpha=  0.01 | E[Q_total]=  265.93 | OK  (21/70)
[2026-03-23 11:43:57,576][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   48.40 | OK  (22/70)
[2026-03-23 11:44:18,064][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   22.14 | OK  (23/70)
[2026-03-23 11:44:38,470][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   17.05 | OK  (24/70)
[2026-03-23 11:44:58,974][__main__][INFO] -   alpha=  2.00 | E[Q_total]=   13.93 | OK  (25/70)
[2026-03-23 11:45:20,596][__main__][INFO] -   alpha=  5.00 | E[Q_total]=   11.90 | OK  (26/70)
[2026-03-23 11:45:41,287][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   11.87 | OK  (27/70)
[2026-03-23 11:46:01,884][__main__][INFO] -   alpha= 20.00 | E[Q_total]=   11.80 | OK  (28/70)
[2026-03-23 11:46:22,396][__main__][INFO] -   alpha= 50.00 | E[Q_total]=   11.81 | OK  (29/70)
[2026-03-23 11:46:43,057][__main__][INFO] -   alpha=100.00 | E[Q_total]=   11.72 | OK  (30/70)
[2026-03-23 11:46:43,058][__main__][INFO] -
------------------------------------------------------------
  rho = 0.90  (lam = 12.6000)
------------------------------------------------------------
[2026-03-23 11:47:06,185][__main__][INFO] -   alpha=  0.01 | E[Q_total]=  399.37 | OK  (31/70)
[2026-03-23 11:47:27,925][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   70.52 | OK  (32/70)
[2026-03-23 11:47:50,189][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   32.01 | OK  (33/70)
[2026-03-23 11:48:11,952][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   25.03 | OK  (34/70)
[2026-03-23 11:48:34,748][__main__][INFO] -   alpha=  2.00 | E[Q_total]=   21.01 | OK  (35/70)
[2026-03-23 11:48:56,305][__main__][INFO] -   alpha=  5.00 | E[Q_total]=   18.83 | OK  (36/70)
[2026-03-23 11:49:17,916][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   18.49 | OK  (37/70)
[2026-03-23 11:49:40,804][__main__][INFO] -   alpha= 20.00 | E[Q_total]=   18.66 | OK  (38/70)
[2026-03-23 11:50:04,987][__main__][INFO] -   alpha= 50.00 | E[Q_total]=   18.68 | OK  (39/70)
[2026-03-23 11:50:34,179][__main__][INFO] -   alpha=100.00 | E[Q_total]=   18.86 | OK  (40/70)
[2026-03-23 11:50:34,179][__main__][INFO] -
------------------------------------------------------------
  rho = 0.95  (lam = 13.3000)
------------------------------------------------------------
[2026-03-23 11:51:05,542][__main__][INFO] -   alpha=  0.01 | E[Q_total]=  504.55 | OK  (41/70)
[2026-03-23 11:51:34,796][__main__][INFO] -   alpha=  0.10 | E[Q_total]=   91.54 | OK  (42/70)
[2026-03-23 11:52:03,982][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   43.95 | OK  (43/70)
[2026-03-23 11:52:41,686][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   37.58 | OK  (44/70)
[2026-03-23 11:53:11,531][__main__][INFO] -   alpha=  2.00 | E[Q_total]=   31.96 | OK  (45/70)
[2026-03-23 11:53:54,912][__main__][INFO] -   alpha=  5.00 | E[Q_total]=   29.49 | OK  (46/70)
[2026-03-23 11:54:22,490][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   29.70 | OK  (47/70)
[2026-03-23 11:54:51,694][__main__][INFO] -   alpha= 20.00 | E[Q_total]=   29.84 | OK  (48/70)
[2026-03-23 11:55:20,312][__main__][INFO] -   alpha= 50.00 | E[Q_total]=   29.86 | OK  (49/70)
[2026-03-23 11:55:48,765][__main__][INFO] -   alpha=100.00 | E[Q_total]=   29.15 | OK  (50/70)
[2026-03-23 11:55:48,766][__main__][INFO] -
------------------------------------------------------------
  rho = 0.98  (lam = 13.7200)
------------------------------------------------------------
[2026-03-23 11:56:24,407][__main__][INFO] -   alpha=  0.01 | E[Q_total]=  617.68 | OK  (51/70)
[2026-03-23 11:57:00,848][__main__][INFO] -   alpha=  0.10 | E[Q_total]=  132.72 | OK  (52/70)
[2026-03-23 11:57:36,248][__main__][INFO] -   alpha=  0.50 | E[Q_total]=   79.10 | OK  (53/70)
[2026-03-23 11:58:12,382][__main__][INFO] -   alpha=  1.00 | E[Q_total]=   66.79 | OK  (54/70)
[2026-03-23 11:58:49,102][__main__][INFO] -   alpha=  2.00 | E[Q_total]=   61.10 | OK  (55/70)
[2026-03-23 11:59:26,136][__main__][INFO] -   alpha=  5.00 | E[Q_total]=   57.69 | OK  (56/70)
[2026-03-23 12:00:03,084][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   62.70 | OK  (57/70)
[2026-03-23 12:00:39,162][__main__][INFO] -   alpha= 20.00 | E[Q_total]=   61.00 | OK  (58/70)
[2026-03-23 12:01:14,763][__main__][INFO] -   alpha= 50.00 | E[Q_total]=   60.71 | OK  (59/70)
[2026-03-23 12:01:50,041][__main__][INFO] -   alpha=100.00 | E[Q_total]=   59.06 | OK  (60/70)
[2026-03-23 12:01:50,042][__main__][INFO] -
------------------------------------------------------------
  rho = 0.99  (lam = 13.8600)
------------------------------------------------------------
[2026-03-23 12:02:21,119][__main__][INFO] -   alpha=  0.01 | E[Q_total]=  684.02 | OK  (61/70)
[2026-03-23 12:02:50,692][__main__][INFO] -   alpha=  0.10 | E[Q_total]=  172.08 | OK  (62/70)
[2026-03-23 12:03:20,971][__main__][INFO] -   alpha=  0.50 | E[Q_total]=  125.44 | OK  (63/70)
[2026-03-23 12:03:50,936][__main__][INFO] -   alpha=  1.00 | E[Q_total]=  121.09 | OK  (64/70)
[2026-03-23 12:04:20,553][__main__][INFO] -   alpha=  2.00 | E[Q_total]=  107.83 | OK  (65/70)
[2026-03-23 12:04:51,279][__main__][INFO] -   alpha=  5.00 | E[Q_total]=   91.89 | OK  (66/70)
[2026-03-23 12:05:21,821][__main__][INFO] -   alpha= 10.00 | E[Q_total]=   91.10 | OK  (67/70)
[2026-03-23 12:05:50,612][__main__][INFO] -   alpha= 20.00 | E[Q_total]=   89.53 | OK  (68/70)
[2026-03-23 12:06:20,706][__main__][INFO] -   alpha= 50.00 | E[Q_total]=   93.84 | OK  (69/70)
[2026-03-23 12:06:50,619][__main__][INFO] -   alpha=100.00 | E[Q_total]=  101.61 | OK  (70/70)
[2026-03-23 12:06:51,100][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 12:06:51,468][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 12:06:52,297][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 12:06:52,455][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 12:06:52,703][__main__][INFO] -
Saved plot: outputs\stability_sweep\run_20260323_113655\alpha_sweep.png, outputs\stability_sweep\run_20260323_113655\alpha_sweep.pdf
[2026-03-23 12:06:52,834][__main__][INFO] -
Summary: 0/70 configurations non-stationary.
wandb:
wandb: Run history:
wandb:             alpha ▁▁▁▁▁▁▁▁▂▂█▁▁▁▁▂▄▁▁▂▄█▁▁▁▂▄█▁▁▁▁▄█▁▁▁▂▂█
wandb:               lam ▁▁▁▁▁▄▄▄▄▄▄▄▅▅▅▅▅▅▅▇▇▇▇▇▇▇▇▇▇▇██████████
wandb:      mean_q_total ▂▁▁▁▁▁▁▃▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▆▁▁▁▇▂▂▂▂▂█▂▂▂▂▂
wandb:               rho ▁▁▁▁▁▄▄▄▄▄▄▄▅▅▅▅▅▇▇▇▇▇▇▇▇▇▇▇▇███████████
wandb: stationarity_rate ▇██▇▇▇▇▇█▇█▇▇██▅▇▆▇▇▇▇█▅▄█▇█▇▇▆▅▇▇▁▅▄▅▄▁
wandb:
wandb: Run summary:
wandb:             alpha 100
wandb:           backend JAX
wandb:     is_stationary True
wandb:               lam 13.86
wandb:      mean_q_total 101.61308
wandb:               rho 0.99
wandb: stationarity_rate 0.78
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync outputs\stability_sweep\run_20260323_113655\wandb\offline-run-20260323_113655-zu3m8pls
wandb: Find logs at: outputs\stability_sweep\run_20260323_113655\wandb\offline-run-20260323_113655-zu3m8pls\logs

[6/10] Running Scaling Stress Tests...
Running: stress
==========================================================
 Starting Experiment: stress
 Remaining Args (Hydra Overrides): +configs=fast ++jax.enabled=True
==========================================================
[2026-03-23 12:06:58,213][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-23 12:06:58,244:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 12:06:58,244][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 12:06:58,245][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-23 12:06:58,264][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\scientific_stress\run_20260323_120658
[2026-03-23 12:06:58,265][__main__][INFO] - ============================================================
[2026-03-23 12:06:58,265][__main__][INFO] -   GibbsQ Stress Test (JAX Accelerator Active)
[2026-03-23 12:06:58,266][__main__][INFO] - ============================================================
[2026-03-23 12:06:58,266][__main__][INFO] -
[TEST 1] Massive-N Scaling Analysis
[2026-03-23 12:06:58,386][__main__][INFO] -   Simulating N=8 experts (rho=0.8)...
[2026-03-23 12:07:02,003][__main__][INFO] -     -> Average Gini Imbalance: 0.0193
[2026-03-23 12:07:02,190][__main__][INFO] -   Simulating N=32 experts (rho=0.8)...
[2026-03-23 12:07:14,162][__main__][INFO] -     -> Average Gini Imbalance: 0.0230
[2026-03-23 12:07:14,320][__main__][INFO] -   Simulating N=128 experts (rho=0.8)...
[2026-03-23 12:08:37,415][__main__][INFO] -     -> Average Gini Imbalance: 0.0242
[2026-03-23 12:08:37,582][__main__][INFO] -   Simulating N=512 experts (rho=0.8)...
[2026-03-23 12:24:08,509][__main__][INFO] -     -> Average Gini Imbalance: 0.0245
[2026-03-23 12:24:08,697][__main__][INFO] -   Simulating N=1024 experts (rho=0.8)...
[2026-03-23 13:16:04,117][__main__][INFO] -     -> Average Gini Imbalance: 0.0245
[2026-03-23 13:16:04,370][__main__][INFO] -
[TEST 2] Critical Load Analysis (rho up to 0.999)
[2026-03-23 13:16:04,409][__main__][INFO] -   Simulating rho=0.900 (T=10000.0)...
[2026-03-23 13:16:40,198][__main__][INFO] -     -> Gelman-Rubin R-hat across replicas (post MSER-5 burn-in): 1.0019
[2026-03-23 13:16:40,368][__main__][INFO] -     -> Avg E[Q_total]: 21.96 | Stationarity: 50/50
[2026-03-23 13:16:40,370][__main__][INFO] -   Simulating rho=0.950 (T=19999.999999999978)...
[2026-03-23 13:17:48,349][__main__][INFO] -     -> Gelman-Rubin R-hat across replicas (post MSER-5 burn-in): 1.0042
[2026-03-23 13:17:48,684][__main__][INFO] -     -> Avg E[Q_total]: 33.23 | Stationarity: 47/50
[2026-03-23 13:17:48,686][__main__][INFO] -   Simulating rho=0.990 (T=99999.9999999999)...
[2026-03-23 13:23:38,550][__main__][INFO] -     -> Gelman-Rubin R-hat across replicas (post MSER-5 burn-in): 1.0915
[2026-03-23 13:23:39,243][__main__][INFO] -     -> Avg E[Q_total]: 102.99 | Stationarity: 48/50
[2026-03-23 13:23:39,248][__main__][WARNING] -   [!] rho=0.9990: sim_time capped at 100,000s (linear mixing time ~ 1000000s). E[Q] near criticality may be underestimated. Report only rho<=0.999 and add mixing-time caveat.
[2026-03-23 13:23:39,248][__main__][INFO] -   Simulating rho=0.999 (T=100000.0)...
[2026-03-23 13:29:33,676][__main__][INFO] -     -> Gelman-Rubin R-hat across replicas (post MSER-5 burn-in): 45.7987
[2026-03-23 13:29:34,308][__main__][INFO] -     -> Avg E[Q_total]: 604.94 | Stationarity: 44/50
[2026-03-23 13:29:34,309][__main__][INFO] -
[TEST 3] Extreme Heterogeneity Resilience (100x Speed Gap)
[2026-03-23 13:29:34,342][__main__][INFO] -   Simulating heterogenous setup: mu=[10.   0.1  0.1  0.1]
[2026-03-23 13:29:38,518][__main__][INFO] -     -> Mean Queue per Expert: [0.96139825 0.96579276 0.96474407 0.96429463]
[2026-03-23 13:29:38,518][__main__][INFO] -     -> Gini: 0.0009
[2026-03-23 13:29:39,421][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 13:29:39,544][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 13:29:39,760][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 13:29:39,810][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 13:29:40,003][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 13:29:40,067][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 13:29:41,323][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 13:29:41,374][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 13:29:41,602][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 13:29:41,696][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 13:29:41,960][__main__][INFO] - Stress dashboard saved to outputs\scientific_stress\run_20260323_120658\stress_dashboard.png, outputs\scientific_stress\run_20260323_120658\stress_dashboard.pdf
[2026-03-23 13:29:41,960][__main__][INFO] -
Stress test complete.



PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ> python scripts/execution/experiment_runner.py reinforce_train --config-name fast
==========================================================
 Starting Experiment: reinforce_train
 Remaining Args (Hydra Overrides): --config-name fast
==========================================================
[2026-03-23 14:18:40,376][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-23 14:18:40,404:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 14:18:40,404][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 14:18:40,405][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-23 14:18:40,423][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\fast\reinforce_training\run_20260323_141840
[2026-03-23 14:18:44,195][__main__][INFO] - Computing JSQ baseline (before bootstrapping)...
[2026-03-23 14:18:45,620][__main__][INFO] -   JSQ Mean Queue (Target): 1.1299
[2026-03-23 14:18:45,620][__main__][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-03-23 14:18:45,622][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-23 14:18:47,231][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
[2026-03-23 14:18:48,525][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6868 | Acc: 35.82%
[2026-03-23 14:18:49,061][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5876 | Acc: 96.12%
[2026-03-23 14:18:49,605][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5874 | Acc: 96.12%
[2026-03-23 14:18:49,606][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-23 14:18:51,085][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
[2026-03-23 14:18:51,813][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 3684.2561
[2026-03-23 14:18:52,483][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 1410.1620
[2026-03-23 14:18:53,140][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 727.9290
[2026-03-23 14:18:53,141][__main__][INFO] - --- Bootstrapping Complete (Actor-Critic Warmed) ---
[2026-03-23 14:18:53,149][__main__][INFO] - ============================================================
[2026-03-23 14:18:53,149][__main__][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-03-23 14:18:53,150][__main__][INFO] - ============================================================
[2026-03-23 14:18:53,150][__main__][INFO] -   Epochs: 30, Batch size: 16
[2026-03-23 14:18:53,151][__main__][INFO] -   Simulation time: 1000.0
[2026-03-23 14:18:53,151][__main__][INFO] - ------------------------------------------------------------
[2026-03-23 14:19:14,828][__main__][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: -0.0510 | mean_logp: -0.6498 | corr: -0.0766
[2026-03-23 14:19:14,828][__main__][INFO] -     [Grad Check] P-Grad Norm: 0.3144 | V-Grad Norm: 2182.7156
[2026-03-23 14:19:14,829][__main__][INFO] - Epoch    0 | mean_reward: -228.3% (EMA: -228.3%) | Loss: -0.0510 | V-Loss: 644.8207
[2026-03-23 14:19:14,829][__main__][INFO] -    -> Signaling | EV:  0.783 [EMA], Corr: -0.0766 [EMA]
[2026-03-23 14:20:59,672][__main__][INFO] - Epoch    5 | mean_reward: -181.9% (EMA: -195.4%) | Loss: -0.0360 | V-Loss: 954.5846
[2026-03-23 14:20:59,672][__main__][INFO] -    -> Signaling | EV:  0.739 [EMA], Corr: -0.0561 [EMA]
[2026-03-23 14:22:37,573][__main__][INFO] -   [Checkpoint] Saved epoch 10 model to policy_net_epoch_010.eqx
[2026-03-23 14:22:37,576][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\fast\latest_reinforce_weights.txt
[2026-03-23 14:23:03,483][__main__][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: -0.0397 | mean_logp: -0.6048 | corr: -0.0461
[2026-03-23 14:23:03,483][__main__][INFO] -     [Grad Check] P-Grad Norm: 0.1399 | V-Grad Norm: 5227.9224
[2026-03-23 14:23:03,484][__main__][INFO] - Epoch   10 | mean_reward: -186.0% (EMA: -173.9%) | Loss: -0.0397 | V-Loss: 1425.4834
[2026-03-23 14:23:03,484][__main__][INFO] -    -> Signaling | EV:  0.735 [EMA], Corr: -0.0457 [EMA]
[2026-03-23 14:25:37,110][__main__][INFO] - Epoch   15 | mean_reward: -151.6% (EMA: -163.7%) | Loss: -0.0344 | V-Loss: 2109.7336
[2026-03-23 14:25:37,111][__main__][INFO] -    -> Signaling | EV:  0.723 [EMA], Corr: -0.0436 [EMA]
[2026-03-23 14:27:32,538][__main__][INFO] -   [Checkpoint] Saved epoch 20 model to policy_net_epoch_020.eqx
[2026-03-23 14:27:32,541][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\fast\latest_reinforce_weights.txt
[2026-03-23 14:28:04,083][__main__][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: -0.0237 | mean_logp: -0.5832 | corr: -0.0292
[2026-03-23 14:28:04,084][__main__][INFO] -     [Grad Check] P-Grad Norm: 0.2517 | V-Grad Norm: 13680.8750
[2026-03-23 14:28:04,085][__main__][INFO] - Epoch   20 | mean_reward: -375.0% (EMA: -224.9%) | Loss: -0.0237 | V-Loss: 3406.4434
[2026-03-23 14:28:04,085][__main__][INFO] -    -> Signaling | EV:  0.786 [EMA], Corr: -0.0484 [EMA]
[2026-03-23 14:30:48,742][__main__][INFO] - Epoch   25 | mean_reward: -340.2% (EMA: -328.8%) | Loss: -0.0428 | V-Loss: 5049.6128
[2026-03-23 14:30:48,742][__main__][INFO] -    -> Signaling | EV:  0.872 [EMA], Corr: -0.0538 [EMA]
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\src\gibbsq\analysis\plotting.py:681: UserWarning: Glyph 8711 (\N{NABLA}) missing from font(s) Times New Roman.
  fig.tight_layout(rect=[0, 0, 1, 0.95])
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\src\gibbsq\utils\chart_exporter.py:108: UserWarning: Glyph 8711 (\N{NABLA}) missing from font(s) Times New Roman.
  fig.savefig(
[2026-03-23 14:33:07,527][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\fast\latest_reinforce_weights.txt
[2026-03-23 14:33:07,527][__main__][INFO] - -------------------------------------------------------
[2026-03-23 14:33:07,528][__main__][INFO] - -------------------------------------------------------
[2026-03-23 14:33:07,528][__main__][INFO] - Running Final Deterministic Evaluation (N=3)...
[2026-03-23 14:33:08,824][__main__][INFO] - Deterministic Policy Score: 110.36% ± 21.63%
[2026-03-23 14:33:08,831][__main__][INFO] - JSQ Target: 100.0% | Random Floor: 0.0% (Performance Index Scale)
[2026-03-23 14:33:08,836][__main__][INFO] - -------------------------------------------------------
[2026-03-23 14:33:08,844][__main__][INFO] - Training Complete! Final Loss: -0.0329
[2026-03-23 14:33:08,852][__main__][INFO] - Final Reward: -312.69
[2026-03-23 14:33:08,856][__main__][INFO] - Policy weights: outputs\fast\reinforce_training\run_20260323_141840\n_gibbsq_reinforce_weights.eqx
[2026-03-23 14:33:08,857][__main__][INFO] - Value weights: outputs\fast\reinforce_training\run_20260323_141840\value_network_weights.eqx
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ>


PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ> python scripts/execution/experiment_runner.py stats --config-name fast
==========================================================
 Starting Experiment: stats
 Remaining Args (Hydra Overrides): --config-name fast
==========================================================
[2026-03-23 14:34:14,767][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-23 14:34:14,800:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 14:34:14,800][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 14:34:14,801][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-23 14:34:14,826][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\fast\stats_benchmark\run_20260323_143414
[2026-03-23 14:34:14,827][__main__][INFO] - ============================================================
[2026-03-23 14:34:14,827][__main__][INFO] -   Phase VII: Statistical Summary
[2026-03-23 14:34:14,828][__main__][INFO] - ============================================================
[2026-03-23 14:34:15,130][__main__][INFO] - Initiating statistical comparison (n=10 seeds).
[2026-03-23 14:34:15,183][__main__][INFO] - Environment: N=2, rho=0.40
[2026-03-23 14:34:17,317][__main__][INFO] - Running 10 GibbsQ SSA simulations...
[2026-03-23 14:34:19,069][__main__][INFO] - Running 10 Neural SSA simulations...
[2026-03-23 14:34:21,187][__main__][INFO] -
============================================================
[2026-03-23 14:34:21,188][__main__][INFO] -   STATISTICAL SUMMARY
[2026-03-23 14:34:21,189][__main__][INFO] - ============================================================
[2026-03-23 14:34:21,189][__main__][INFO] - GibbsQ E[Q]:   1.1323 ± 0.0904
[2026-03-23 14:34:21,190][__main__][INFO] - N-GibbsQ E[Q]:   1.1100 ± 0.1127
[2026-03-23 14:34:21,190][__main__][INFO] - Rel. Improve:  1.97%
[2026-03-23 14:34:21,190][__main__][INFO] - ----------------------------------------
[2026-03-23 14:34:21,191][__main__][INFO] - P-Value:       6.31e-01 (NOT SIGNIFICANT)
[2026-03-23 14:34:21,192][__main__][INFO] - Effect Size:   -0.22 (Cohen's d)
[2026-03-23 14:34:21,192][__main__][INFO] - 95% CI (Diff): [-0.1184, 0.0737]
[2026-03-23 14:34:21,192][__main__][INFO] - ============================================================
[2026-03-23 14:34:21,977][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 14:34:22,063][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 14:34:22,162][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 14:34:22,941][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 14:34:23,029][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ>
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ> python scripts/execution/experiment_runner.py generalize --config-name fast
==========================================================
 Starting Experiment: generalize
 Remaining Args (Hydra Overrides): --config-name fast
==========================================================
[2026-03-23 14:41:25,647][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-23 14:41:25,674:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 14:41:25,674][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 14:41:25,675][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-23 14:41:25,696][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\fast\generalization_sweep\run_20260323_144125
[2026-03-23 14:41:25,697][__main__][INFO] - ============================================================
[2026-03-23 14:41:25,697][__main__][INFO] -   Phase VIII: Generalization & Stress Heatmap
[2026-03-23 14:41:25,698][__main__][INFO] - ============================================================
[2026-03-23 14:41:25,960][__main__][INFO] - Initiating Generalization Sweep (Scales=[0.5, 2.0], rho=[0.5, 0.85])
[2026-03-23 14:41:28,813][__main__][INFO] - Evaluating N-GibbsQ improvement ratio (GibbsQ / Neural) on 5x5 Grid...
[2026-03-23 14:41:35,122][__main__][INFO] -    Scale=  0.5x | rho=0.50 | Improvement=1.06x
[2026-03-23 14:41:39,026][__main__][INFO] -    Scale=  0.5x | rho=0.85 | Improvement=1.07x
[2026-03-23 14:41:45,419][__main__][INFO] -    Scale=  2.0x | rho=0.50 | Improvement=0.98x
[2026-03-23 14:41:54,806][__main__][INFO] -    Scale=  2.0x | rho=0.85 | Improvement=0.87x
[2026-03-23 14:41:57,442][__main__][INFO] - Generalization analysis complete. Heatmap saved to outputs\fast\generalization_sweep\run_20260323_144125\generalization_heatmap.png, outputs\fast\generalization_sweep\run_20260323_144125\generalization_heatmap.pdf
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ>


PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ> python scripts/execution/experiment_runner.py critical --config-name fast
==========================================================
 Starting Experiment: critical
 Remaining Args (Hydra Overrides): --config-name fast
==========================================================
[2026-03-23 14:44:15,883][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-23 14:44:15,911:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 14:44:15,911][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 14:44:15,912][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-23 14:44:15,939][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\fast\critical_load\run_20260323_144415
[2026-03-23 14:44:15,940][__main__][INFO] - ============================================================
[2026-03-23 14:44:15,940][__main__][INFO] -   Phase VIII: The Critical Stability Boundary
[2026-03-23 14:44:15,941][__main__][INFO] - ============================================================
[2026-03-23 14:44:18,464][__main__][INFO] - System Capacity: 2.50
[2026-03-23 14:44:18,465][__main__][INFO] - Targeting Load Boundary: [0.95]
[2026-03-23 14:44:18,492][__main__][INFO] - Evaluating Boundary rho=0.950 (Arrival=2.375)...
[2026-03-23 14:44:39,576][__main__][INFO] -    => N-GibbsQ E[Q]: 18.38 | GibbsQ E[Q]: 20.92
[2026-03-23 14:44:40,276][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 14:44:40,421][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 14:44:40,618][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 14:44:41,947][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 14:44:42,200][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 14:44:42,474][__main__][INFO] - Critical load test complete. Curve saved to outputs\fast\critical_load\run_20260323_144415\critical_load_curve.png, outputs\fast\critical_load\run_20260323_144415\critical_load_curve.pdf
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ>


PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ> python scripts/execution/experiment_runner.py ablation --config-name fast
==========================================================
 Starting Experiment: ablation
 Remaining Args (Hydra Overrides): --config-name fast
==========================================================
[2026-03-23 14:47:45,754][gibbsq.utils.device][INFO] - [JAX] Using default 32-bit precision (FP32) for performance.
INFO:2026-03-23 14:47:45,796:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 14:47:45,796][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-23 14:47:45,797][gibbsq.utils.device][INFO] - [JAX] Auto-selected platform: CPU (1 devices)
[2026-03-23 14:47:45,822][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\fast\ablation_ssa\run_20260323_144745
[2026-03-23 14:47:45,822][__main__][INFO] - ============================================================
[2026-03-23 14:47:45,826][__main__][INFO] -   SSA-Based Ablation Study
[2026-03-23 14:47:45,827][__main__][INFO] - ============================================================
[2026-03-23 14:47:45,829][__main__][INFO] - ------------------------------------------------------------
[2026-03-23 14:47:45,831][__main__][INFO] - Training variant: Full Model
[2026-03-23 14:47:45,833][__main__][INFO] -   preprocessing=log1p, init_type=zero_final
[2026-03-23 14:47:49,936][experiments.training.train_reinforce][INFO] - Computing JSQ baseline (before bootstrapping)...
[2026-03-23 14:47:51,847][experiments.training.train_reinforce][INFO] -   JSQ Mean Queue (Target): 1.1299
[2026-03-23 14:47:51,848][experiments.training.train_reinforce][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-03-23 14:47:51,851][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-23 14:47:53,426][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
[2026-03-23 14:47:54,611][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6868 | Acc: 35.82%
[2026-03-23 14:47:55,221][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5876 | Acc: 96.12%
[2026-03-23 14:47:55,875][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5874 | Acc: 96.12%
[2026-03-23 14:47:55,876][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-23 14:47:57,781][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
[2026-03-23 14:47:58,574][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 3684.2561
[2026-03-23 14:47:59,185][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 1410.1620
[2026-03-23 14:47:59,763][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 727.9290
[2026-03-23 14:47:59,764][experiments.training.train_reinforce][INFO] - --- Bootstrapping Complete (Actor-Critic Warmed) ---
[2026-03-23 14:47:59,769][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-23 14:47:59,770][experiments.training.train_reinforce][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-03-23 14:47:59,770][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-23 14:47:59,770][experiments.training.train_reinforce][INFO] -   Epochs: 30, Batch size: 16
[2026-03-23 14:47:59,771][experiments.training.train_reinforce][INFO] -   Simulation time: 1000.0
[2026-03-23 14:47:59,771][experiments.training.train_reinforce][INFO] - ------------------------------------------------------------
[2026-03-23 14:48:24,818][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: -0.0510 | mean_logp: -0.6498 | corr: -0.0766
[2026-03-23 14:48:24,819][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.3144 | V-Grad Norm: 2182.7156
[2026-03-23 14:48:24,820][experiments.training.train_reinforce][INFO] - Epoch    0 | mean_reward: -228.3% (EMA: -228.3%) | Loss: -0.0510 | V-Loss: 644.8207
[2026-03-23 14:48:24,820][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.783 [EMA], Corr: -0.0766 [EMA]
[2026-03-23 14:50:29,507][experiments.training.train_reinforce][INFO] - Epoch    5 | mean_reward: -181.9% (EMA: -195.4%) | Loss: -0.0360 | V-Loss: 954.5846
[2026-03-23 14:50:29,508][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.739 [EMA], Corr: -0.0561 [EMA]
[2026-03-23 14:52:23,661][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 10 model to policy_net_epoch_010.eqx
[2026-03-23 14:52:23,663][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\fast\ablation_ssa\latest_reinforce_weights.txt
[2026-03-23 14:52:52,767][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: -0.0397 | mean_logp: -0.6048 | corr: -0.0461
[2026-03-23 14:52:52,767][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.1399 | V-Grad Norm: 5227.9224
[2026-03-23 14:52:52,768][experiments.training.train_reinforce][INFO] - Epoch   10 | mean_reward: -186.0% (EMA: -173.9%) | Loss: -0.0397 | V-Loss: 1425.4834
[2026-03-23 14:52:52,768][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.735 [EMA], Corr: -0.0457 [EMA]
[2026-03-23 14:55:15,954][experiments.training.train_reinforce][INFO] - Epoch   15 | mean_reward: -151.6% (EMA: -163.7%) | Loss: -0.0344 | V-Loss: 2109.7336
[2026-03-23 14:55:15,955][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.723 [EMA], Corr: -0.0436 [EMA]
[2026-03-23 14:57:29,272][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 20 model to policy_net_epoch_020.eqx
[2026-03-23 14:57:29,274][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\fast\ablation_ssa\latest_reinforce_weights.txt
[2026-03-23 14:58:10,280][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: -0.0237 | mean_logp: -0.5832 | corr: -0.0292
[2026-03-23 14:58:10,281][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.2517 | V-Grad Norm: 13680.8750
[2026-03-23 14:58:10,282][experiments.training.train_reinforce][INFO] - Epoch   20 | mean_reward: -375.0% (EMA: -224.9%) | Loss: -0.0237 | V-Loss: 3406.4434
[2026-03-23 14:58:10,282][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.786 [EMA], Corr: -0.0484 [EMA]
[2026-03-23 15:01:14,548][experiments.training.train_reinforce][INFO] - Epoch   25 | mean_reward: -340.2% (EMA: -328.8%) | Loss: -0.0428 | V-Loss: 5049.6128
[2026-03-23 15:01:14,549][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.872 [EMA], Corr: -0.0538 [EMA]
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\src\gibbsq\analysis\plotting.py:657: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_c.legend(loc="lower right", fontsize=7)
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\src\gibbsq\analysis\plotting.py:677: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_d.legend(loc="upper right", fontsize=7)
[2026-03-23 15:03:32,160][__main__][INFO] - Saved variant artifacts in outputs\fast\ablation_ssa\run_20260323_144745\variant_1_full_model
[2026-03-23 15:03:32,219][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)
[2026-03-23 15:03:32,690][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-03-23 15:03:32,691][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-23 15:03:33,171][gibbsq.engines.numpy_engine][INFO] -   -> 1,001 arrivals, 1,000 departures, final Q_total = 1
[2026-03-23 15:03:33,172][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-23 15:03:33,996][gibbsq.engines.numpy_engine][INFO] -   -> 981 arrivals, 975 departures, final Q_total = 6
[2026-03-23 15:03:33,998][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-23 15:03:34,593][gibbsq.engines.numpy_engine][INFO] -   -> 1,040 arrivals, 1,040 departures, final Q_total = 0
[2026-03-23 15:03:34,594][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-03-23 15:03:35,188][gibbsq.engines.numpy_engine][INFO] -   -> 1,027 arrivals, 1,026 departures, final Q_total = 1
[2026-03-23 15:03:35,189][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-03-23 15:03:35,670][gibbsq.engines.numpy_engine][INFO] -   -> 908 arrivals, 907 departures, final Q_total = 1
[2026-03-23 15:03:35,671][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-23 15:03:36,480][gibbsq.engines.numpy_engine][INFO] -   -> 989 arrivals, 989 departures, final Q_total = 0
[2026-03-23 15:03:36,481][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-23 15:03:37,135][gibbsq.engines.numpy_engine][INFO] -   -> 1,020 arrivals, 1,018 departures, final Q_total = 2
[2026-03-23 15:03:37,136][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-03-23 15:03:37,641][gibbsq.engines.numpy_engine][INFO] -   -> 1,041 arrivals, 1,039 departures, final Q_total = 2
[2026-03-23 15:03:37,643][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-23 15:03:38,198][gibbsq.engines.numpy_engine][INFO] -   -> 1,060 arrivals, 1,060 departures, final Q_total = 0
[2026-03-23 15:03:38,204][__main__][INFO] -   SSA E[Q_total] = 1.1759 +/- 0.0392
[2026-03-23 15:03:38,206][__main__][INFO] - ------------------------------------------------------------
[2026-03-23 15:03:38,207][__main__][INFO] - Training variant: Ablated: No Log-Norm
[2026-03-23 15:03:38,207][__main__][INFO] -   preprocessing=none, init_type=zero_final
[2026-03-23 15:03:38,228][experiments.training.train_reinforce][INFO] - Computing JSQ baseline (before bootstrapping)...
[2026-03-23 15:03:41,374][experiments.training.train_reinforce][INFO] -   JSQ Mean Queue (Target): 1.1299
[2026-03-23 15:03:41,375][experiments.training.train_reinforce][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-03-23 15:03:41,376][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-23 15:03:43,451][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
[2026-03-23 15:03:44,806][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6823 | Acc: 64.18%
[2026-03-23 15:03:47,620][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5835 | Acc: 98.58%
[2026-03-23 15:03:50,510][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5835 | Acc: 95.05%
[2026-03-23 15:03:50,511][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-23 15:03:52,760][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
[2026-03-23 15:03:53,552][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 3669.3113
[2026-03-23 15:03:55,792][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 1053.5160
[2026-03-23 15:03:57,997][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 509.4660
[2026-03-23 15:03:57,998][experiments.training.train_reinforce][INFO] - --- Bootstrapping Complete (Actor-Critic Warmed) ---
[2026-03-23 15:03:58,010][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-23 15:03:58,011][experiments.training.train_reinforce][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-03-23 15:03:58,012][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-23 15:03:58,012][experiments.training.train_reinforce][INFO] -   Epochs: 30, Batch size: 16
[2026-03-23 15:03:58,014][experiments.training.train_reinforce][INFO] -   Simulation time: 1000.0
[2026-03-23 15:03:58,014][experiments.training.train_reinforce][INFO] - ------------------------------------------------------------
[2026-03-23 15:04:36,704][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: -0.0608 | mean_logp: -0.6522 | corr: -0.0947
[2026-03-23 15:04:36,705][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.2436 | V-Grad Norm: 3240.2947
[2026-03-23 15:04:36,705][experiments.training.train_reinforce][INFO] - Epoch    0 | mean_reward: -231.9% (EMA: -231.9%) | Loss: -0.0608 | V-Loss: 897.3005
[2026-03-23 15:04:36,707][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.677 [EMA], Corr: -0.0947 [EMA]
[2026-03-23 15:07:27,154][experiments.training.train_reinforce][INFO] - Epoch    5 | mean_reward: -177.5% (EMA: -194.2%) | Loss: -0.0477 | V-Loss: 1038.6250
[2026-03-23 15:07:27,154][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.639 [EMA], Corr: -0.0732 [EMA]
[2026-03-23 15:09:31,554][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 10 model to policy_net_epoch_010.eqx
[2026-03-23 15:09:31,558][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\fast\ablation_ssa\latest_reinforce_weights.txt
[2026-03-23 15:10:02,591][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: -0.0460 | mean_logp: -0.6049 | corr: -0.0530
[2026-03-23 15:10:02,593][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.1158 | V-Grad Norm: 4833.5254
[2026-03-23 15:10:02,593][experiments.training.train_reinforce][INFO] - Epoch   10 | mean_reward: -180.9% (EMA: -168.7%) | Loss: -0.0460 | V-Loss: 1252.0916
[2026-03-23 15:10:02,594][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.675 [EMA], Corr: -0.0557 [EMA]
[2026-03-23 15:12:30,766][experiments.training.train_reinforce][INFO] - Epoch   15 | mean_reward: -149.4% (EMA: -160.7%) | Loss: -0.0357 | V-Loss: 1681.3452
[2026-03-23 15:12:30,767][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.703 [EMA], Corr: -0.0448 [EMA]
[2026-03-23 15:14:57,930][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 20 model to policy_net_epoch_020.eqx
[2026-03-23 15:14:57,933][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\fast\ablation_ssa\latest_reinforce_weights.txt
[2026-03-23 15:15:34,385][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: -0.0266 | mean_logp: -0.5590 | corr: -0.0317
[2026-03-23 15:15:34,386][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.2319 | V-Grad Norm: 11415.0127
[2026-03-23 15:15:34,386][experiments.training.train_reinforce][INFO] - Epoch   20 | mean_reward: -372.3% (EMA: -222.4%) | Loss: -0.0266 | V-Loss: 2708.5017
[2026-03-23 15:15:34,387][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.781 [EMA], Corr: -0.0465 [EMA]
[2026-03-23 15:18:53,275][experiments.training.train_reinforce][INFO] - Epoch   25 | mean_reward: -330.5% (EMA: -322.3%) | Loss: -0.0474 | V-Loss: 3992.7341
[2026-03-23 15:18:53,276][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.870 [EMA], Corr: -0.0530 [EMA]
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\src\gibbsq\analysis\plotting.py:657: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_c.legend(loc="lower right", fontsize=7)
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\src\gibbsq\analysis\plotting.py:677: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_d.legend(loc="upper right", fontsize=7)
[2026-03-23 15:21:19,315][__main__][INFO] - Saved variant artifacts in outputs\fast\ablation_ssa\run_20260323_144745\variant_2_ablated_no_log-norm
[2026-03-23 15:21:19,395][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)
[2026-03-23 15:21:20,509][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-03-23 15:21:20,512][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-23 15:21:21,830][gibbsq.engines.numpy_engine][INFO] -   -> 996 arrivals, 993 departures, final Q_total = 3
[2026-03-23 15:21:21,833][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-23 15:21:22,824][gibbsq.engines.numpy_engine][INFO] -   -> 993 arrivals, 991 departures, final Q_total = 2
[2026-03-23 15:21:22,825][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-23 15:21:23,497][gibbsq.engines.numpy_engine][INFO] -   -> 1,050 arrivals, 1,049 departures, final Q_total = 1
[2026-03-23 15:21:23,498][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-03-23 15:21:24,378][gibbsq.engines.numpy_engine][INFO] -   -> 1,041 arrivals, 1,039 departures, final Q_total = 2
[2026-03-23 15:21:24,380][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-03-23 15:21:25,839][gibbsq.engines.numpy_engine][INFO] -   -> 904 arrivals, 894 departures, final Q_total = 10
[2026-03-23 15:21:25,840][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-23 15:21:27,565][gibbsq.engines.numpy_engine][INFO] -   -> 1,000 arrivals, 997 departures, final Q_total = 3
[2026-03-23 15:21:27,566][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-23 15:21:29,011][gibbsq.engines.numpy_engine][INFO] -   -> 1,027 arrivals, 1,026 departures, final Q_total = 1
[2026-03-23 15:21:29,038][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-03-23 15:21:29,689][gibbsq.engines.numpy_engine][INFO] -   -> 1,027 arrivals, 1,027 departures, final Q_total = 0
[2026-03-23 15:21:29,690][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-23 15:21:30,287][gibbsq.engines.numpy_engine][INFO] -   -> 1,049 arrivals, 1,047 departures, final Q_total = 2
[2026-03-23 15:21:30,292][__main__][INFO] -   SSA E[Q_total] = 1.1836 +/- 0.0331
[2026-03-23 15:21:30,294][__main__][INFO] - ------------------------------------------------------------
[2026-03-23 15:21:30,294][__main__][INFO] - Training variant: Ablated: No Zero-Init
[2026-03-23 15:21:30,295][__main__][INFO] -   preprocessing=log1p, init_type=standard
[2026-03-23 15:21:30,313][experiments.training.train_reinforce][INFO] - Computing JSQ baseline (before bootstrapping)...
[2026-03-23 15:21:32,146][experiments.training.train_reinforce][INFO] -   JSQ Mean Queue (Target): 1.1299
[2026-03-23 15:21:32,147][experiments.training.train_reinforce][INFO] -   Random Mean Queue (Analytical): 1.5000
[2026-03-23 15:21:32,148][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-23 15:21:34,786][gibbsq.core.pretraining][INFO] - --- Bootstrapping Actor (Behavior Cloning) ---
[2026-03-23 15:21:36,644][gibbsq.core.pretraining][INFO] -   Step    0 | Loss: 0.6823 | Acc: 52.24%
[2026-03-23 15:21:39,287][gibbsq.core.pretraining][INFO] -   Step  100 | Loss: 0.5871 | Acc: 96.12%
[2026-03-23 15:21:41,324][gibbsq.core.pretraining][INFO] -   Step  200 | Loss: 0.5850 | Acc: 98.51%
[2026-03-23 15:21:41,325][gibbsq.core.pretraining][INFO] - --- Collecting Robust Expert Data (Steady-State + Augmentation) ---
[2026-03-23 15:21:43,317][gibbsq.core.pretraining][INFO] - --- Bootstrapping Critic (Value Warming) ---
[2026-03-23 15:21:44,923][gibbsq.core.pretraining][INFO] -   Step    0 | MSE Loss: 3625.7993
[2026-03-23 15:21:46,646][gibbsq.core.pretraining][INFO] -   Step  100 | MSE Loss: 962.7968
[2026-03-23 15:21:48,536][gibbsq.core.pretraining][INFO] -   Step  200 | MSE Loss: 417.2241
[2026-03-23 15:21:48,537][experiments.training.train_reinforce][INFO] - --- Bootstrapping Complete (Actor-Critic Warmed) ---
[2026-03-23 15:21:48,547][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-23 15:21:48,547][experiments.training.train_reinforce][INFO] -   REINFORCE Training (SSA-Based Policy Gradient)
[2026-03-23 15:21:48,548][experiments.training.train_reinforce][INFO] - ============================================================
[2026-03-23 15:21:48,548][experiments.training.train_reinforce][INFO] -   Epochs: 30, Batch size: 16
[2026-03-23 15:21:48,548][experiments.training.train_reinforce][INFO] -   Simulation time: 1000.0
[2026-03-23 15:21:48,548][experiments.training.train_reinforce][INFO] - ------------------------------------------------------------
[2026-03-23 15:22:20,914][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: -0.0587 | mean_logp: -0.6471 | corr: -0.0935
[2026-03-23 15:22:20,916][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.3162 | V-Grad Norm: 3535.8086
[2026-03-23 15:22:20,917][experiments.training.train_reinforce][INFO] - Epoch    0 | mean_reward: -224.2% (EMA: -224.2%) | Loss: -0.0587 | V-Loss: 1013.1010
[2026-03-23 15:22:20,917][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.584 [EMA], Corr: -0.0935 [EMA]
[2026-03-23 15:26:14,480][experiments.training.train_reinforce][INFO] - Epoch    5 | mean_reward: -176.7% (EMA: -191.3%) | Loss: -0.0366 | V-Loss: 1057.6851
[2026-03-23 15:26:14,481][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.563 [EMA], Corr: -0.0607 [EMA]
[2026-03-23 15:28:34,832][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 10 model to policy_net_epoch_010.eqx
[2026-03-23 15:28:34,834][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\fast\ablation_ssa\latest_reinforce_weights.txt
[2026-03-23 15:29:03,044][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: -0.0000 | mean_loss: -0.0488 | mean_logp: -0.5926 | corr: -0.0558
[2026-03-23 15:29:03,045][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.1343 | V-Grad Norm: 4676.3403
[2026-03-23 15:29:03,047][experiments.training.train_reinforce][INFO] - Epoch   10 | mean_reward: -177.9% (EMA: -166.9%) | Loss: -0.0488 | V-Loss: 1160.7469
[2026-03-23 15:29:03,048][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.641 [EMA], Corr: -0.0502 [EMA]
[2026-03-23 15:31:47,832][experiments.training.train_reinforce][INFO] - Epoch   15 | mean_reward: -142.7% (EMA: -155.1%) | Loss: -0.0451 | V-Loss: 1527.8706
[2026-03-23 15:31:47,833][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.694 [EMA], Corr: -0.0541 [EMA]
[2026-03-23 15:33:49,611][experiments.training.train_reinforce][INFO] -   [Checkpoint] Saved epoch 20 model to policy_net_epoch_020.eqx
[2026-03-23 15:33:49,615][gibbsq.utils.model_io][INFO] - [Pointer] Updated latest_reinforce_weights.txt at outputs\fast\ablation_ssa\latest_reinforce_weights.txt
[2026-03-23 15:34:25,804][experiments.training.train_reinforce][INFO] -     [Sign Check] mean_adv: 0.0000 | mean_loss: -0.0298 | mean_logp: -0.5576 | corr: -0.0356
[2026-03-23 15:34:25,820][experiments.training.train_reinforce][INFO] -     [Grad Check] P-Grad Norm: 0.2182 | V-Grad Norm: 11000.0664
[2026-03-23 15:34:25,846][experiments.training.train_reinforce][INFO] - Epoch   20 | mean_reward: -366.8% (EMA: -216.1%) | Loss: -0.0298 | V-Loss: 2517.7808
[2026-03-23 15:34:25,851][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.778 [EMA], Corr: -0.0565 [EMA]
[2026-03-23 15:37:17,178][experiments.training.train_reinforce][INFO] - Epoch   25 | mean_reward: -327.0% (EMA: -315.7%) | Loss: -0.0496 | V-Loss: 3817.8540
[2026-03-23 15:37:17,179][experiments.training.train_reinforce][INFO] -    -> Signaling | EV:  0.871 [EMA], Corr: -0.0595 [EMA]
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\src\gibbsq\analysis\plotting.py:657: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_c.legend(loc="lower right", fontsize=7)
C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\src\gibbsq\analysis\plotting.py:677: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  ax_d.legend(loc="upper right", fontsize=7)
[2026-03-23 15:39:52,705][__main__][INFO] - Saved variant artifacts in outputs\fast\ablation_ssa\run_20260323_144745\variant_3_ablated_no_zero-init
[2026-03-23 15:39:52,756][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)
[2026-03-23 15:39:53,238][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-03-23 15:39:53,238][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-23 15:39:54,001][gibbsq.engines.numpy_engine][INFO] -   -> 1,001 arrivals, 1,000 departures, final Q_total = 1
[2026-03-23 15:39:54,002][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-23 15:39:54,777][gibbsq.engines.numpy_engine][INFO] -   -> 990 arrivals, 990 departures, final Q_total = 0
[2026-03-23 15:39:54,778][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-23 15:39:55,501][gibbsq.engines.numpy_engine][INFO] -   -> 1,047 arrivals, 1,045 departures, final Q_total = 2
[2026-03-23 15:39:55,502][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-03-23 15:39:56,177][gibbsq.engines.numpy_engine][INFO] -   -> 1,034 arrivals, 1,033 departures, final Q_total = 1
[2026-03-23 15:39:56,177][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-03-23 15:39:56,579][gibbsq.engines.numpy_engine][INFO] -   -> 909 arrivals, 907 departures, final Q_total = 2
[2026-03-23 15:39:56,580][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-23 15:39:57,000][gibbsq.engines.numpy_engine][INFO] -   -> 994 arrivals, 993 departures, final Q_total = 1
[2026-03-23 15:39:57,001][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-23 15:39:57,492][gibbsq.engines.numpy_engine][INFO] -   -> 1,020 arrivals, 1,019 departures, final Q_total = 1
[2026-03-23 15:39:57,493][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-03-23 15:39:57,976][gibbsq.engines.numpy_engine][INFO] -   -> 1,038 arrivals, 1,038 departures, final Q_total = 0
[2026-03-23 15:39:57,977][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-23 15:39:58,454][gibbsq.engines.numpy_engine][INFO] -   -> 1,054 arrivals, 1,051 departures, final Q_total = 3
[2026-03-23 15:39:58,457][__main__][INFO] -   SSA E[Q_total] = 1.1806 +/- 0.0360
[2026-03-23 15:39:58,457][gibbsq.engines.numpy_engine][INFO] - Replication 1/10  (seed=42)
[2026-03-23 15:39:58,584][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-03-23 15:39:58,585][gibbsq.engines.numpy_engine][INFO] - Replication 2/10  (seed=43)
[2026-03-23 15:39:58,734][gibbsq.engines.numpy_engine][INFO] -   -> 1,010 arrivals, 1,010 departures, final Q_total = 0
[2026-03-23 15:39:58,734][gibbsq.engines.numpy_engine][INFO] - Replication 3/10  (seed=44)
[2026-03-23 15:39:58,888][gibbsq.engines.numpy_engine][INFO] -   -> 991 arrivals, 990 departures, final Q_total = 1
[2026-03-23 15:39:58,888][gibbsq.engines.numpy_engine][INFO] - Replication 4/10  (seed=45)
[2026-03-23 15:39:59,031][gibbsq.engines.numpy_engine][INFO] -   -> 1,040 arrivals, 1,040 departures, final Q_total = 0
[2026-03-23 15:39:59,031][gibbsq.engines.numpy_engine][INFO] - Replication 5/10  (seed=46)
[2026-03-23 15:39:59,185][gibbsq.engines.numpy_engine][INFO] -   -> 1,027 arrivals, 1,026 departures, final Q_total = 1
[2026-03-23 15:39:59,186][gibbsq.engines.numpy_engine][INFO] - Replication 6/10  (seed=47)
[2026-03-23 15:39:59,346][gibbsq.engines.numpy_engine][INFO] -   -> 911 arrivals, 911 departures, final Q_total = 0
[2026-03-23 15:39:59,347][gibbsq.engines.numpy_engine][INFO] - Replication 7/10  (seed=48)
[2026-03-23 15:39:59,524][gibbsq.engines.numpy_engine][INFO] -   -> 993 arrivals, 991 departures, final Q_total = 2
[2026-03-23 15:39:59,524][gibbsq.engines.numpy_engine][INFO] - Replication 8/10  (seed=49)
[2026-03-23 15:39:59,869][gibbsq.engines.numpy_engine][INFO] -   -> 1,029 arrivals, 1,027 departures, final Q_total = 2
[2026-03-23 15:39:59,870][gibbsq.engines.numpy_engine][INFO] - Replication 9/10  (seed=50)
[2026-03-23 15:40:00,107][gibbsq.engines.numpy_engine][INFO] -   -> 1,020 arrivals, 1,016 departures, final Q_total = 4
[2026-03-23 15:40:00,107][gibbsq.engines.numpy_engine][INFO] - Replication 10/10  (seed=51)
[2026-03-23 15:40:00,312][gibbsq.engines.numpy_engine][INFO] -   -> 1,058 arrivals, 1,058 departures, final Q_total = 0
[2026-03-23 15:40:01,077][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 15:40:01,207][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 15:40:01,354][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 15:40:02,082][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
[2026-03-23 15:40:02,164][matplotlib.mathtext][INFO] - Substituting symbol E from STIXNonUnicode
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ>



