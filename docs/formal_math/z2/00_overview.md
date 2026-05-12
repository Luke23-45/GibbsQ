# Calibrated UAS: Formal Math Overview

This note gives the high-level mathematical map for the `z2` package.

Its purpose is not to restate every theorem. Its purpose is to identify the
correct backbone of the thesis and to separate:

- what is already proved,
- what is a correction or limitation result,
- what is an audited but still promotion-sensitive stochastic route,
- and what is merely supporting context.

## Package Backbone

The package now has three active mathematical layers.

### Layer 1. Deterministic theorem core

The reflected-ODE theory is the main established contribution.

Its key points are:

- the correct deterministic object is a reflected ODE on the nonnegative
  orthant,
- its equilibrium is a boundary equilibrium rather than a strictly positive
  interior fixed point,
- that equilibrium can be characterized by an exact scalar consistency
  equation,
- the reflected dynamics form a constrained convex gradient system,
- every reflected trajectory converges to the unique equilibrium.

This layer is carried by:

- [01_reflected_fluid_model.md](./01_reflected_fluid_model.md)
- [02_boundary_equilibrium.md](./02_boundary_equilibrium.md)
- [05_projected_gradient_structure.md](./05_projected_gradient_structure.md)
- [06_global_convergence_reflected_ode.md](./06_global_convergence_reflected_ode.md)

### Layer 2. Stochastic correction and obstruction

The project also establishes why the older stochastic shortcut was not valid.

Its key points are:

- the fixed-parameter CTMC does not automatically inherit the smooth reflected
  ODE as its classical fluid limit under standard queue-length scaling,
- the exact CTMC generator for the potential \(H\) contains a genuine boundary
  mismatch,
- the old `x5`-style lift from deterministic convergence to CTMC stability was
  therefore too strong.

This layer is carried by:

- [07_ctmc_scaling_gap.md](./07_ctmc_scaling_gap.md)
- [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md)
- [09_review_of_suggestions.md](./09_review_of_suggestions.md)

This is not failure material to hide. It is part of the corrected scientific
foundation.

### Layer 3. Conditional direct CTMC route

The package also contains a direct Foster-Lyapunov route for the fixed-parameter
CTMC based on a weighted quadratic Lyapunov function and an exact
softmax-minimum bound.

This route is carried by:

- [10_direct_ctmc_quadratic_proof_attempt.md](./10_direct_ctmc_quadratic_proof_attempt.md)
- [11_audit_of_direct_ctmc_quadratic_proof.md](./11_audit_of_direct_ctmc_quadratic_proof.md)

These files are central to the stochastic program, but they should currently be
read as an **audited certification route** unless explicitly promoted to final
theorem status.

## Policy Under Study

For service rates \(\mu_i>0\), queue state \(q\in\mathbb R_+^N\), and
parameters \(\alpha>0\), \(\beta>0\), \(\gamma\in\mathbb R\), \(c\ge 0\), the
Calibrated-UAS routing probabilities are

\[
p_i(q)
=
\frac{\mu_i^\gamma \exp\!\left(-\alpha (q_i+c)/\mu_i^\beta\right)}
{\sum_{j=1}^N \mu_j^\gamma \exp\!\left(-\alpha (q_j+c)/\mu_j^\beta\right)}.
\]

Throughout the package,

\[
\Lambda := \sum_{i=1}^N \mu_i,
\qquad
\lambda \in (0,\Lambda).
\]

## Thesis-Level Position

The thesis should treat the package as supporting the following claim ladder.

### Established theorem material

- the reflected deterministic model is explicit and correct
- the boundary equilibrium is exact and unique
- the convex-potential structure is explicit
- the reflected ODE is globally asymptotically stable

### Established correction material

- the old smooth-ODE-to-CTMC shortcut is invalid as previously stated
- the boundary mismatch in the exact CTMC generator is real

### Conditional stochastic material

- the direct weighted-quadratic CTMC route appears to close the fixed-parameter
  theorem and has passed internal audit and targeted validation
- however, it should remain promotion-sensitive until final theorem-level
  sign-off is recorded

### Supporting applicability material

- because the routing law is softmax-based and continuous, it is compatible with
  differentiable policy-learning workflows
- this is useful supporting evidence, but not the main hero of the package

## File-Role Map

- [03_theorem_status.md](./03_theorem_status.md)
  authoritative status note for proved, conditional, and supporting claims
- [04_references.md](./04_references.md)
  external-reference map only
- [12_thesis_hypotheses.md](./12_thesis_hypotheses.md)
  thesis claim ladder
- [13_thesis_evidence_requirements.md](./13_thesis_evidence_requirements.md)
  evidence thresholds and forbidden overclaims
- [suggestions.md](./suggestions.md)
  archival guidance only; not an authoritative theorem-status file
