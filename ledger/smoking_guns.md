# Smoking Gun Registry

## Confirmed Findings

SMOKING GUN #1
  Experiment    : sweep (also impacts stress/generalize/critical)
  File          : experiments/sweeps/stability_sweep.py
  Function      : main
  Lines         : 58-77
  Observed      : rho_vals is read from config and used to compute lam=rho*cap without any rho<1 or lam<cap guard.
  Source        : Source 1: experiments/sweeps/stability_sweep.py:58-77. Source 2: src/gibbsq/core/config.py:471-475 validates only cfg.system.arrival_rate against capacity, not sweep rho lists.
  Impact        : A user-supplied rho >= 1 would violate the strict capacity condition (Λ>λ) in docs/gibbsq.md and invalidate stability claims.
  Severity      : HIGH
  Severity basis: Directly bypasses the theorem precondition for multiple experiments when overrides/custom configs are used.
  Status        : CONFIRMED

SMOKING GUN #2
  Experiment    : sweep (JAX branch)
  File          : src/gibbsq/engines/jax_engine.py
  Function      : _validate_inputs / run_replications_jax call chain
  Lines         : 23-58, 413-470
  Observed      : JAX simulation input validation checks arrival/service/sample bounds but does not enforce alpha>0.
  Source        : Source 1: src/gibbsq/engines/jax_engine.py:23-58 omits alpha positivity checks. Source 2: experiments/sweeps/stability_sweep.py:76-109 passes per-cell alpha_values directly into run_replications_jax.
  Impact        : Negative/zero alpha from sweep overrides can be executed in JAX path, violating the documented model condition alpha>0.
  Severity      : HIGH
  Severity basis: Violates core policy/theory assumption and creates backend inconsistency (NumPy policy constructors reject alpha<=0, JAX path does not).
  Status        : CONFIRMED

## Summary Table
| # | Experiment | File | Severity | Status |
|---|---|---|---|---|
| 1 | sweep (+stress/generalize/critical) | experiments/sweeps/stability_sweep.py | HIGH | CONFIRMED |
| 2 | sweep (JAX branch) | src/gibbsq/engines/jax_engine.py | HIGH | CONFIRMED |

## Cross-Experiment Findings
- SG#1 is cross-experiment: the same unchecked rho-driven λ reconstruction pattern appears in sweep/stress/generalize/critical style loops.

## Total confirmed : 2
## Total critical  : 0
## Total high      : 2
## Total medium    : 0
