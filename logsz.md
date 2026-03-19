PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ> python scripts/execution/experiment_runner.py reinforce_check --config-name debug
==========================================================
 Starting Experiment: reinforce_check
 Remaining Args (Hydra Overrides): --config-name debug
==========================================================
[2026-03-19 21:16:24,140][gibbsq.utils.logging][INFO] - [IO] Run directory: C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ\outputs\small\gradient_check\run_20260319_211624
[2026-03-19 21:16:24,143][__main__][INFO] - ============================================================
[2026-03-19 21:16:24,149][__main__][INFO] -   REINFORCE Gradient Estimator Validation
[2026-03-19 21:16:24,150][__main__][INFO] - ============================================================
INFO:2026-03-19 21:16:24,221:jax._src.xla_bridge:752: Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-19 21:16:24,221][jax._src.xla_bridge][INFO] - Unable to initialize backend 'tpu': UNIMPLEMENTED: LoadPjrtPlugin is not implemented on windows yet.
[2026-03-19 21:16:30,013][__main__][INFO] - Computing REINFORCE gradient estimate...
[2026-03-19 21:16:37,711][__main__][INFO] - Computing finite-difference gradient estimate...
[2026-03-19 21:16:53,917][__main__][INFO] -   [OK] Param  10/500 (idx   185): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.0000 | CosSim= 0.0000
[2026-03-19 21:17:08,100][__main__][INFO] -   [OK] Param  20/500 (idx 11191): RF= -0.000161 | FD=  0.000000 | diff=  0.000161 | z= 1.30 | RelErr= 0.4632 | CosSim= 0.9756
[2026-03-19 21:17:22,369][__main__][INFO] -   [!!] Param  30/500 (idx  3319): RF= -0.002312 | FD= -0.000912 | diff=  0.001400 | z= 5.91 | RelErr= 0.4426 | CosSim= 0.9647
[2026-03-19 21:17:35,988][__main__][INFO] -   [!!] Param  40/500 (idx 10958): RF=  0.045993 | FD=  0.086434 | diff=  0.040440 | z= 7.33 | RelErr= 0.4457 | CosSim= 0.9534
[2026-03-19 21:17:49,546][__main__][INFO] -   [OK] Param  50/500 (idx  9902): RF= -0.001017 | FD= -0.003965 | diff=  0.002948 | z= 1.64 | RelErr= 0.4662 | CosSim= 0.9630
[2026-03-19 21:18:03,519][__main__][INFO] -   [OK] Param  60/500 (idx 14805): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.4669 | CosSim= 0.9628
[2026-03-19 21:18:17,114][__main__][INFO] -   [!!] Param  70/500 (idx  5707): RF= -0.019518 | FD= -0.002500 | diff=  0.017018 | z=12.37 | RelErr= 0.4343 | CosSim= 0.9605
[2026-03-19 21:18:29,853][__main__][INFO] -   [OK] Param  80/500 (idx  1541): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.4041 | CosSim= 0.9584
[2026-03-19 21:18:43,070][__main__][INFO] -   [OK] Param  90/500 (idx 10434): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3714 | CosSim= 0.9545
[2026-03-19 21:18:57,139][__main__][INFO] -   [OK] Param 100/500 (idx 11447): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3718 | CosSim= 0.9543
[2026-03-19 21:19:10,940][__main__][INFO] -   [OK] Param 110/500 (idx  4596): RF= -0.003832 | FD= -0.003320 | diff=  0.000512 | z= 1.24 | RelErr= 0.3719 | CosSim= 0.9543
[2026-03-19 21:19:26,851][__main__][INFO] -   [!!] Param 120/500 (idx 10956): RF=  0.049225 | FD=  0.078367 | diff=  0.029142 | z= 6.83 | RelErr= 0.3720 | CosSim= 0.9529
[2026-03-19 21:19:39,731][__main__][INFO] -   [OK] Param 130/500 (idx  8566): RF= -0.190511 | FD= -0.167533 | diff=  0.022978 | z= 1.36 | RelErr= 0.3608 | CosSim= 0.9561
[2026-03-19 21:19:52,397][__main__][INFO] -   [OK] Param 140/500 (idx  3968): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3607 | CosSim= 0.9561
[2026-03-19 21:20:05,564][__main__][INFO] -   [!!] Param 150/500 (idx  7426): RF= -0.002197 | FD= -0.031080 | diff=  0.028883 | z= 5.68 | RelErr= 0.3643 | CosSim= 0.9545
[2026-03-19 21:20:17,555][__main__][INFO] -   [!!] Param 160/500 (idx  7166): RF=  0.047125 | FD=  0.067855 | diff=  0.020730 | z= 6.53 | RelErr= 0.3574 | CosSim= 0.9560
[2026-03-19 21:20:30,188][__main__][INFO] -   [OK] Param 170/500 (idx 11461): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3671 | CosSim= 0.9534
[2026-03-19 21:20:44,142][__main__][INFO] -   [OK] Param 180/500 (idx 10261): RF=  0.003330 | FD=  0.003320 | diff=  0.000009 | z= 0.00 | RelErr= 0.3643 | CosSim= 0.9538
[2026-03-19 21:20:57,139][__main__][INFO] -   [OK] Param 190/500 (idx 16700): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3680 | CosSim= 0.9521
[2026-03-19 21:21:11,074][__main__][INFO] -   [!!] Param 200/500 (idx  1554): RF= -0.045706 | FD= -0.092891 | diff=  0.047185 | z= 5.26 | RelErr= 0.3806 | CosSim= 0.9436
[2026-03-19 21:21:24,746][__main__][INFO] -   [!!] Param 210/500 (idx  5646): RF= -0.042691 | FD= -0.089443 | diff=  0.046752 | z= 3.61 | RelErr= 0.3804 | CosSim= 0.9426
[2026-03-19 21:21:38,632][__main__][INFO] -   [OK] Param 220/500 (idx  9741): RF=  0.001704 | FD=  0.000000 | diff=  0.001704 | z= 1.11 | RelErr= 0.3918 | CosSim= 0.9433
[2026-03-19 21:21:52,000][__main__][INFO] -   [!!] Param 230/500 (idx 12641): RF= -0.049508 | FD= -0.023754 | diff=  0.025754 | z= 7.56 | RelErr= 0.3834 | CosSim= 0.9478
[2026-03-19 21:22:07,202][__main__][INFO] -   [OK] Param 240/500 (idx  4407): RF= -0.000177 | FD=  0.000000 | diff=  0.000177 | z= 1.30 | RelErr= 0.3824 | CosSim= 0.9479
[2026-03-19 21:22:23,269][__main__][INFO] -   [!!] Param 250/500 (idx  9631): RF=  0.019178 | FD=  0.005590 | diff=  0.013588 | z=10.36 | RelErr= 0.3820 | CosSim= 0.9483
[2026-03-19 21:22:36,650][__main__][INFO] -   [OK] Param 260/500 (idx  1389): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3782 | CosSim= 0.9488
[2026-03-19 21:22:49,872][__main__][INFO] -   [!!] Param 270/500 (idx 10337): RF=  0.000747 | FD=  0.003588 | diff=  0.002841 | z= 3.70 | RelErr= 0.3782 | CosSim= 0.9488
[2026-03-19 21:23:03,510][__main__][INFO] -   [OK] Param 280/500 (idx 15956): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3758 | CosSim= 0.9492
[2026-03-19 21:23:17,045][__main__][INFO] -   [OK] Param 290/500 (idx 17069): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3544 | CosSim= 0.9669
[2026-03-19 21:23:31,355][__main__][INFO] -   [!!] Param 300/500 (idx  5810): RF= -0.021176 | FD= -0.038764 | diff=  0.017588 | z= 7.71 | RelErr= 0.3413 | CosSim= 0.9689
[2026-03-19 21:23:45,067][__main__][INFO] -   [OK] Param 310/500 (idx  8294): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3405 | CosSim= 0.9693
[2026-03-19 21:24:00,934][__main__][INFO] -   [!!] Param 320/500 (idx  4971): RF= -0.060847 | FD= -0.322229 | diff=  0.261382 | z= 4.98 | RelErr= 0.3918 | CosSim= 0.9455
[2026-03-19 21:24:17,192][__main__][INFO] -   [OK] Param 330/500 (idx  3058): RF= -0.001314 | FD= -0.003588 | diff=  0.002274 | z= 1.72 | RelErr= 0.3918 | CosSim= 0.9455
[2026-03-19 21:24:33,185][__main__][INFO] -   [!!] Param 340/500 (idx 12488): RF=  0.160943 | FD=  0.111021 | diff=  0.049922 | z= 6.34 | RelErr= 0.3932 | CosSim= 0.9451
[2026-03-19 21:24:48,415][__main__][INFO] -   [OK] Param 350/500 (idx 16600): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3885 | CosSim= 0.9464
[2026-03-19 21:25:03,868][__main__][INFO] -   [!!] Param 360/500 (idx  3367): RF= -0.020461 | FD= -0.084799 | diff=  0.064338 | z= 3.92 | RelErr= 0.3898 | CosSim= 0.9442
[2026-03-19 21:25:19,475][__main__][INFO] -   [OK] Param 370/500 (idx  5092): RF=  0.002773 | FD=  0.003588 | diff=  0.000814 | z= 1.38 | RelErr= 0.3612 | CosSim= 0.9536
[2026-03-19 21:25:34,970][__main__][INFO] -   [OK] Param 380/500 (idx  7140): RF=  0.142521 | FD=  0.135479 | diff=  0.007043 | z= 0.45 | RelErr= 0.3610 | CosSim= 0.9527
[2026-03-19 21:25:47,718][__main__][INFO] -   [OK] Param 390/500 (idx   242): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3558 | CosSim= 0.9542
[2026-03-19 21:26:02,805][__main__][INFO] -   [OK] Param 400/500 (idx 13194): RF=  0.162332 | FD=  0.150965 | diff=  0.011367 | z= 0.56 | RelErr= 0.3529 | CosSim= 0.9540
[2026-03-19 21:26:33,822][__main__][INFO] -   [!!] Param 410/500 (idx  8481): RF= -0.047111 | FD= -0.092854 | diff=  0.045742 | z= 9.70 | RelErr= 0.3537 | CosSim= 0.9534
[2026-03-19 21:26:46,696][__main__][INFO] -   [OK] Param 420/500 (idx  4472): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3529 | CosSim= 0.9536
[2026-03-19 21:27:01,668][__main__][INFO] -   [OK] Param 430/500 (idx  1665): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3526 | CosSim= 0.9536
[2026-03-19 21:27:17,022][__main__][INFO] -   [OK] Param 440/500 (idx  1156): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3549 | CosSim= 0.9532
[2026-03-19 21:27:29,152][__main__][INFO] -   [OK] Param 450/500 (idx  2363): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3448 | CosSim= 0.9566
[2026-03-19 21:27:41,029][__main__][INFO] -   [OK] Param 460/500 (idx 14140): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3450 | CosSim= 0.9565
[2026-03-19 21:27:53,882][__main__][INFO] -   [OK] Param 470/500 (idx 12052): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3445 | CosSim= 0.9565
[2026-03-19 21:28:11,369][__main__][INFO] -   [OK] Param 480/500 (idx 15206): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3447 | CosSim= 0.9564
[2026-03-19 21:28:26,549][__main__][INFO] -   [OK] Param 490/500 (idx  8340): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3475 | CosSim= 0.9562
[2026-03-19 21:28:39,970][__main__][INFO] -   [OK] Param 500/500 (idx  3744): RF= -0.000000 | FD=  0.000000 | diff=  0.000000 | z=N/A  | RelErr= 0.3494 | CosSim= 0.9555
[2026-03-19 21:28:39,988][__main__][INFO] - Relative error: 0.3494
[2026-03-19 21:28:39,989][__main__][INFO] - Cosine similarity: 0.9555
[2026-03-19 21:28:39,990][__main__][INFO] - Bias estimate (L2): 0.539416
[2026-03-19 21:28:39,990][__main__][INFO] - Relative bias: 0.3494
[2026-03-19 21:28:39,991][__main__][INFO] - Variance estimate: 0.294586
[2026-03-19 21:28:39,991][__main__][INFO] - Passed: True
[2026-03-19 21:28:39,996][__main__][INFO] - Results saved to outputs\small\gradient_check\run_20260319_211624\gradient_check_result.json
[2026-03-19 21:28:40,001][__main__][INFO] - GRADIENT CHECK PASSED - REINFORCE estimator is valid
PS C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\MoEQ>