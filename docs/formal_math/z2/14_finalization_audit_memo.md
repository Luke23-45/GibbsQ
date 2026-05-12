# `z2` Finalization Audit Memo

This memo records the audit classification of each file in the `z2` package
after the framing and consistency audit.

The labels below describe the kind of action required during this pass. They do
not mean that the updated files are still awaiting another round by default.

Use the labels exactly as follows:

- `correct as is`
- `needs framing update`
- `needs theorem-language tightening`
- `archive/support only`

## Audit Summary

### Theorem core

- `01_reflected_fluid_model.md` - `correct as is`
- `02_boundary_equilibrium.md` - `correct as is`
- `05_projected_gradient_structure.md` - `correct as is`
- `06_global_convergence_reflected_ode.md` - `correct as is`

No proof gap was identified during the finalization pass. These files remain the
active deterministic theorem backbone.

### Framing and control layer

- `README.md` - `needs framing update`
- `00_overview.md` - `needs framing update`
- `03_theorem_status.md` - `needs framing update`
- `12_thesis_hypotheses.md` - `correct as is`
- `13_thesis_evidence_requirements.md` - `correct as is`

These files were the primary sources of stale transitional language and are now
aligned to the current claim ladder. Their labels describe what this audit pass
had to do, not unresolved follow-up by default.

### Correction and obstruction layer

- `07_ctmc_scaling_gap.md` - `needs framing update`
- `08_ctmc_generator_analysis.md` - `correct as is`
- `09_review_of_suggestions.md` - `needs framing update`

These files remain active because they document real mathematical corrections,
not dead exploratory work.

### Conditional stochastic route

- `10_direct_ctmc_quadratic_proof_attempt.md` - `needs theorem-language tightening`
- `11_audit_of_direct_ctmc_quadratic_proof.md` - `needs theorem-language tightening`

These files remain central to the stochastic program, but they are
promotion-sensitive and must not silently override the package-level status
controls. Their labels again describe the type of tightening applied in this
pass.

### Support and archive

- `04_references.md` - `needs framing update`
- `suggestions.md` - `archive/support only`

`suggestions.md` remains useful as an archival guidance record, but it is not
authoritative for final thesis status.

## Final Package-Level Reading Rule

Read the package with the following hierarchy:

1. `README`, `00`, `03`, `12`, `13`, `14` control status and claim discipline
2. `01`, `02`, `05`, `06` are established theorem core
3. `07`, `08`, `09` are established correction / obstruction notes
4. `10`, `11` are audited but promotion-sensitive stochastic route files
5. `04` and `suggestions.md` are support files, not active theorem-status files

## Final Risk Control

The package is safe to use as the thesis math foundation only if the following
discipline is preserved:

- deterministic theorem claims come from `01`, `02`, `05`, `06`
- obstruction claims come from `07`, `08`, `09`
- the direct CTMC route is not automatically treated as fully promoted unless
  that promotion is explicitly recorded
- differentiable/trainable-policy material remains supporting applicability only
- exploratory policy work stays outside the formal-math backbone
