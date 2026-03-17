# Unverified Signals

## Fault-Surface Checklist Questions (generated to widen failure hunt)
1. Does each Phase IV entrypoint resolve config_name and override semantics consistently across PowerShell, Bash, and direct Python invocation?
2. Is every artifact path derived from config/output_dir rather than hardcoded literals?
3. Are pointer files atomically written and validated before consumer read?
4. Can stale pointers from prior runs contaminate corrected_policy results?
5. Do all four tracks use consistent random seed derivation and reproducibility contracts?
6. Is policy weight dimensionality checked against both num_servers and hidden_size before evaluation?
7. Are gradient-check thresholds sourced from config and used consistently in pass/fail logic?
8. Do finite-difference and REINFORCE objectives measure the same return definition (discounting + horizon handling)?
9. Is burn-in applied consistently when computing queue, gini, and sojourn metrics?
10. Can WandB/logging failures alter experiment control flow or silently skip result writes?
11. Is Hydra fallback path (no argv) behavior equivalent to CLI path for all tracks?
12. Are exceptions in neural evaluation only logged (warning) while still producing explicit failure status?
13. Are domain-randomization phases guaranteed to cover target rho ranges defined in config?
14. Are run ids and output directories guaranteed unique to avoid race/overwrite in batch runs?

## Remaining Unverified Signals
- None (all identified code-path anomalies in this pass were confirmed or moved to UNKNOWN).
