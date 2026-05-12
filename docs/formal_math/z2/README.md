# `z2` Formal Math Package

This directory contains the current proof-facing math notes for the Calibrated
UAS stability program. It is intentionally organized as a **chronological
research record** rather than a single polished manuscript: some files document
earlier obstruction analyses, while later files record the newer direct CTMC
proof route and its validation.

## Current Reading Order

If you want the shortest path through the current theory, read the files in this
order:

1. [00_overview.md](./00_overview.md)
2. [01_reflected_fluid_model.md](./01_reflected_fluid_model.md)
3. [02_boundary_equilibrium.md](./02_boundary_equilibrium.md)
4. [05_projected_gradient_structure.md](./05_projected_gradient_structure.md)
5. [06_global_convergence_reflected_ode.md](./06_global_convergence_reflected_ode.md)
6. [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md)
7. [10_direct_ctmc_quadratic_proof_attempt.md](./10_direct_ctmc_quadratic_proof_attempt.md)
8. [11_audit_of_direct_ctmc_quadratic_proof.md](./11_audit_of_direct_ctmc_quadratic_proof.md)

Use the remaining files as status notes, references, and review audit trails.

## What Each File Does

- [00_overview.md](./00_overview.md)
  High-level map of the deterministic and stochastic proof program.

- [01_reflected_fluid_model.md](./01_reflected_fluid_model.md)
  Defines the reflected deterministic surrogate and proves that any equilibrium
  must lie on the boundary.

- [02_boundary_equilibrium.md](./02_boundary_equilibrium.md)
  Derives the exact scalar characterization of the unique boundary equilibrium.

- [03_theorem_status.md](./03_theorem_status.md)
  Status note for the theorem program.
  Important: this file reflects the pre-promotion state of the project before
  the later direct CTMC route is fully integrated into all status notes.

- [04_references.md](./04_references.md)
  Queueing, reflected-dynamics, and convex-flow references.

- [05_projected_gradient_structure.md](./05_projected_gradient_structure.md)
  Shows that the reflected ODE is a constrained convex gradient system.

- [06_global_convergence_reflected_ode.md](./06_global_convergence_reflected_ode.md)
  Proves global asymptotic stability of the reflected ODE.

- [07_ctmc_scaling_gap.md](./07_ctmc_scaling_gap.md)
  Explains why the reflected ODE did not automatically certify the original CTMC
  under the original fluid-limit discussion.

- [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md)
  Derives the exact CTMC generator for the potential `H` and isolates the
  boundary mismatch.

- [09_review_of_suggestions.md](./09_review_of_suggestions.md)
  Audits the professor review in `suggestions.md` and keeps only the parts that
  are mathematically justified.

- [10_direct_ctmc_quadratic_proof_attempt.md](./10_direct_ctmc_quadratic_proof_attempt.md)
  New direct Foster-Lyapunov proof attempt for the fixed-parameter CTMC using a
  weighted quadratic Lyapunov function and an exact softmax-minimum bound.

- [11_audit_of_direct_ctmc_quadratic_proof.md](./11_audit_of_direct_ctmc_quadratic_proof.md)
  Internal audit of the proof attempt in file `10`.

- [suggestions.md](./suggestions.md)
  External review / research guidance record. This is not a theorem file.

## Current Scientific Position

The directory now contains two distinct theorem layers:

1. **Deterministic reflected-ODE theory**
   This is established in files `01`, `02`, `05`, and `06`.

2. **Direct CTMC proof route**
   This is developed in files `10` and `11`, and numerically validated by the
   dedicated experiment runner
   [direct_ctmc_validation.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/experiments/verification/direct_ctmc_validation.py).

The older obstruction notes in files `07` and `08` remain important because
they explain why the earlier stochastic route was insufficient. They should not
be deleted; they document the failure mode that the newer proof attempt is meant
to bypass.

## Validation Artifacts

The new direct CTMC route has a dedicated validation capsule and a dedicated
test file.

### Unit Test

Test file:
- [test_direct_ctmc_validation.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/tests/test_direct_ctmc_validation.py)

Recorded result:

```text
pytest tests\test_direct_ctmc_validation.py -q
..                                                                       [100%]
2 passed in 16.59s
```

What the test covers:
- positivity and finiteness of the new theorem constants on a toy subcritical
  system
- exhaustive verification on a small state grid that the exact quadratic
  generator drift satisfies the claimed bound

### Benchmark Audit

Audit summary:
- [direct_ctmc_validation_summary.md](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/direct_ctmc_validation/direct_ctmc_validation/final_20260512_235101/metadata/direct_ctmc_validation_summary.md)
- [direct_ctmc_audit.jsonl](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/direct_ctmc_validation/direct_ctmc_validation/final_20260512_235101/metrics/direct_ctmc_audit.jsonl)

Key benchmark audit outcomes:
- `uas_special_case`: `epsilon = 0.200000000000`, sampled bound passed
- `calibrated_default`: `epsilon = 0.212805842869`, sampled bound passed
- `grid_b0p5_g0p25_c0p25`: `epsilon = 0.242337433796`, sampled bound passed
- `grid_b0p5_g0p5_c0p25`: `epsilon = 0.242337433796`, sampled bound passed
- `grid_b0p7_g0p25_c0p25`: `epsilon = 0.225592288568`, sampled bound passed

This is the main reason files `10` and `11` are now central: the benchmark
default calibrated policy passed the dedicated audit under the new route.

### Quick Empirical Rerun

Quick rerun summary:
- [direct_ctmc_validation_summary.md](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/direct_ctmc_validation_quick/direct_ctmc_validation/final_20260512_235410/metadata/direct_ctmc_validation_summary.md)

Key short-run comparison:
- `UAS`: `11.489221`
- `Calibrated UAS (empirical default)`: `10.039646`
- candidate `(0.5, 0.25, 0.25)`: `10.341040`
- candidate `(0.5, 0.5, 0.25)`: `10.307986`
- candidate `(0.7, 0.25, 0.25)`: `10.223725`

So the benchmark default calibrated policy remains the strongest performer among
the audited closed-form candidates in that quick rerun.

## Recommended Next Maintenance Step

Before using `z2` as the final manuscript source, the status notes should be
promoted and synchronized:

- update [03_theorem_status.md](./03_theorem_status.md) so it explicitly reflects
  the newer direct CTMC route
- update [00_overview.md](./00_overview.md) if you want the overview to present
  file `10` as the active stochastic theorem path rather than only as a later
  development
==

I ran the targeted SMVR comparison at higher load, not the full expensive sweep.

Command used:
```powershell
python -m experiments.testing.smvr_probe --config-name final_experiment `
  wandb.mode=disabled `
  system.arrival_rate=13.3 `
  +probe.smvr.search_mode=quick `
  +probe.smvr.max_candidates=8 `
  +probe.smvr.top_k=3 `
  +probe.smvr.pilot_replications=2 `
  +probe.smvr.pilot_sim_time=800 `
  +probe.smvr.final_replications=4 `
  +probe.smvr.final_sim_time=3000 `
  +probe.smvr.min_unique_candidates=4
```

Results are in:
- [smvr_probe_summary.md](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/final/smvr_probe/final_20260512_210420/artifacts/smvr_probe_summary.md)
- [paired_deltas.json](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/final/smvr_probe/final_20260512_210420/artifacts/paired_deltas.json)
- [final_metrics.jsonl](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/final/smvr_probe/final_20260512_210420/metrics/final_metrics.jsonl)

Conclusion: this architecture did **not** beat `Calibrated UAS`.

At `lambda = 13.3` (`rho = 0.95` on total service `14.0`):
- `Calibrated UAS`: `26.4913`
- `JSSQ`: `28.6455`
- `UAS`: `29.5137`

Best SMVR candidate:
- `smvr_a(18)_g(0.5)_l(0)_t(0)`: `26.5068`
- delta vs `Calibrated UAS`: `+0.0155`
- paired 95% CI: `[-0.0606, +0.0917]`

That means there is **no evidence of improvement** over `Calibrated UAS`.

More importantly, the best **genuine correction** candidate was worse:
- `smvr_a(18)_g(1)_l(0.2)_t(0)`: `27.1426`
- delta vs `Calibrated UAS`: `+0.6514`
- paired 95% CI: `[+0.3876, +0.9152]`

So the targeted run gives a clean decision: **stop the SMVR line as an empirical direction**. The value-correction part did not help. The only near-tie came from a zero-correction retune, which is just a calibrated-UAS variant, not a new architecture.


====
This README is written to remain robust even before that promotion, by making
the chronology explicit.



