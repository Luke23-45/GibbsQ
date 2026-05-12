# `z2` Formal Math Package

This directory is the canonical formal-math package for the current
Calibrated-UAS thesis direction.

It is no longer organized as an open-ended research dump. Its files now fall
into clear roles:

- **theorem core** for the deterministic reflected-ODE program,
- **correction / obstruction notes** for the failed stochastic shortcut,
- **conditional stochastic certification route** for the direct CTMC argument,
- **supporting control files** for thesis claim discipline,
- **archive / external-guidance material** that should not control the final
  thesis status on its own.

The package is meant to be safe to read as the mathematical foundation of the
project without importing older overclaims from `docs/notes/x1/x5.md`.

## Recommended Reading Order

For the cleanest path through the package, read:

1. [00_overview.md](./00_overview.md)
2. [03_theorem_status.md](./03_theorem_status.md)
3. [01_reflected_fluid_model.md](./01_reflected_fluid_model.md)
4. [02_boundary_equilibrium.md](./02_boundary_equilibrium.md)
5. [05_projected_gradient_structure.md](./05_projected_gradient_structure.md)
6. [06_global_convergence_reflected_ode.md](./06_global_convergence_reflected_ode.md)
7. [07_ctmc_scaling_gap.md](./07_ctmc_scaling_gap.md)
8. [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md)
9. [09_review_of_suggestions.md](./09_review_of_suggestions.md)
10. [10_direct_ctmc_quadratic_proof_attempt.md](./10_direct_ctmc_quadratic_proof_attempt.md)
11. [11_audit_of_direct_ctmc_quadratic_proof.md](./11_audit_of_direct_ctmc_quadratic_proof.md)
12. [12_thesis_hypotheses.md](./12_thesis_hypotheses.md)
13. [13_thesis_evidence_requirements.md](./13_thesis_evidence_requirements.md)

Use [04_references.md](./04_references.md) as the external-reference map and
[suggestions.md](./suggestions.md) only as archival guidance.

## File Roles

### Theorem core

- [01_reflected_fluid_model.md](./01_reflected_fluid_model.md)
  Correct reflected deterministic model and boundary complementarity setup.
- [02_boundary_equilibrium.md](./02_boundary_equilibrium.md)
  Exact scalar characterization of the unique boundary equilibrium.
- [05_projected_gradient_structure.md](./05_projected_gradient_structure.md)
  Hidden convex potential, convexity, coercivity, and constrained minimizer.
- [06_global_convergence_reflected_ode.md](./06_global_convergence_reflected_ode.md)
  Global asymptotic stability of the reflected ODE.

These files are the mathematical backbone of the thesis and are treated as
established theorem material.

### Correction and obstruction notes

- [07_ctmc_scaling_gap.md](./07_ctmc_scaling_gap.md)
  Explains why the reflected ODE does not automatically arise as the classical
  fluid limit of the fixed-parameter CTMC.
- [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md)
  Gives the exact CTMC generator for the potential \(H\) and isolates the
  boundary mismatch.
- [09_review_of_suggestions.md](./09_review_of_suggestions.md)
  Audits `suggestions.md` and keeps only the mathematically justified parts.

These files are active parts of the thesis story, but as limitation and
correction material rather than theorem-completion material.

### Conditional stochastic route

- [10_direct_ctmc_quadratic_proof_attempt.md](./10_direct_ctmc_quadratic_proof_attempt.md)
  Proof-facing direct Foster-Lyapunov route for the fixed-parameter CTMC.
- [11_audit_of_direct_ctmc_quadratic_proof.md](./11_audit_of_direct_ctmc_quadratic_proof.md)
  Internal audit of the argument in file `10`.

These files are central to the stochastic program, but they are
**promotion-sensitive**. They should be treated as an audited certification
route unless and until they are explicitly promoted to finished theorem status.

### Control files for thesis discipline

- [00_overview.md](./00_overview.md)
  High-level mathematical map of the package.
- [03_theorem_status.md](./03_theorem_status.md)
  Authoritative statement of what is proved, conditional, and merely
  validated.
- [12_thesis_hypotheses.md](./12_thesis_hypotheses.md)
  Claim ladder for the thesis.
- [13_thesis_evidence_requirements.md](./13_thesis_evidence_requirements.md)
  Evidence thresholds and forbidden overclaims.

### Support and archive files

- [04_references.md](./04_references.md)
  External theorem and background references.
- [suggestions.md](./suggestions.md)
  External guidance / archival note. This file is not authoritative for final
  thesis status.

## Current Scientific Position

The package supports the following hierarchy.

### Established theorem core

The deterministic reflected-ODE program is established:

- the correct deterministic object is the reflected ODE on the orthant,
- the equilibrium lies on the boundary and is uniquely characterized,
- the reflected ODE is a constrained convex gradient system,
- every reflected trajectory converges to the unique equilibrium.

### Established correction result

The old shortcut from the smooth reflected ODE to the CTMC is not valid as
stated:

- the classical fluid-scaling route is delicate for fixed \(\alpha\),
- the exact CTMC generator for \(H\) contains a genuine boundary mismatch,
- the earlier stochastic shortcut must therefore be treated as invalid.

### Conditional stochastic route

The direct weighted-quadratic CTMC argument in files `10` and `11` is supported
by internal audit and targeted validation, but it should still be presented as
**conditional / promotion-sensitive** unless explicit theorem-level sign-off is
recorded.

### Supporting applicability result

Because Calibrated UAS is softmax-based and smooth, it is compatible with
differentiable policy-learning workflows in a way that hard dispatch rules such
as JSQ and JSSQ are not. This is a supporting applicability result, not the
main hero of the package.

## Validation Artifacts

The direct CTMC route has a dedicated validation capsule and test support.

### Unit test

Test file:
- [test_direct_ctmc_validation.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/tests/test_direct_ctmc_validation.py)

Recorded result:

```text
pytest tests\test_direct_ctmc_validation.py -q
..                                                                       [100%]
2 passed in 16.59s
```

What the test covers:

- positivity and finiteness of the theorem constants on a toy subcritical
  system
- exhaustive verification on a small state grid that the exact quadratic
  generator drift satisfies the claimed bound

### Benchmark audit

Audit summary:
- [direct_ctmc_validation_summary.md](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/direct_ctmc_validation/direct_ctmc_validation/final_20260512_235101/metadata/direct_ctmc_validation_summary.md)
- [direct_ctmc_audit.jsonl](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/direct_ctmc_validation/direct_ctmc_validation/final_20260512_235101/metrics/direct_ctmc_audit.jsonl)

Key audit outcomes:

- `uas_special_case`: `epsilon = 0.200000000000`, sampled bound passed
- `calibrated_default`: `epsilon = 0.212805842869`, sampled bound passed
- `grid_b0p5_g0p25_c0p25`: `epsilon = 0.242337433796`, sampled bound passed
- `grid_b0p5_g0p5_c0p25`: `epsilon = 0.242337433796`, sampled bound passed
- `grid_b0p7_g0p25_c0p25`: `epsilon = 0.225592288568`, sampled bound passed

These checks support the conditional stochastic route. They do not by
themselves replace theorem-level sign-off.

### Quick empirical rerun

Quick rerun summary:
- [direct_ctmc_validation_summary.md](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/direct_ctmc_validation_quick/direct_ctmc_validation/final_20260512_235410/metadata/direct_ctmc_validation_summary.md)

Key short-run comparison:

- `UAS`: `11.489221`
- `Calibrated UAS (empirical default)`: `10.039646`
- candidate `(0.5, 0.25, 0.25)`: `10.341040`
- candidate `(0.5, 0.5, 0.25)`: `10.307986`
- candidate `(0.7, 0.25, 0.25)`: `10.223725`

This rerun supports the benchmark relevance of the default calibrated point, but
it is still an empirical benchmark fact rather than a theorem statement.
