# Calibrated UAS: Corrected Deterministic Theory

This folder replaces the over-strong claims in `docs/notes/x1/x5.md` with a
proof-facing package that keeps only statements that can be justified from the
current model.

The key correction is structural:

- the relevant deterministic object is a **reflected ODE** on the nonnegative
  orthant
- its equilibrium is a **boundary equilibrium**, not an interior positive fixed
  point
- the equilibrium can be characterized exactly by a scalar consistency equation
- the reflected ODE has a hidden **convex projected-gradient structure**
- every reflected-ODE trajectory converges to the unique boundary equilibrium
- the remaining stochastic gap is the link from that deterministic ODE to a
  **classical fluid-limit theorem** for the CTMC

That separation matters. It lets the project move forward without claiming a
queueing stability theorem that has not yet been proved.

## File Map

- [01_reflected_fluid_model.md](./01_reflected_fluid_model.md)
  Defines the corrected reflected model and proves that any equilibrium must lie
  on the boundary.

- [02_boundary_equilibrium.md](./02_boundary_equilibrium.md)
  Derives the exact scalar characterization of the boundary equilibrium and
  proves uniqueness.

- [03_theorem_status.md](./03_theorem_status.md)
  States precisely what is proved, what remains open, and what the next theorem
  should be.

- [05_projected_gradient_structure.md](./05_projected_gradient_structure.md)
  Identifies the hidden convex potential behind the reflected ODE and proves
  convexity, coercivity, and uniqueness of the constrained minimizer.

- [06_global_convergence_reflected_ode.md](./06_global_convergence_reflected_ode.md)
  Proves that every reflected-ODE trajectory converges to the unique boundary
  equilibrium.

- [07_ctmc_scaling_gap.md](./07_ctmc_scaling_gap.md)
  Explains the remaining stochastic issue: the reflected ODE analyzed here is
  not yet identified as the classical Dai fluid limit of the CTMC under the
  usual queue-length scaling.

- [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md)
  Derives the exact CTMC generator acting on the same potential \(H\) and
  isolates the boundary mismatch that obstructs an immediate Foster-Lyapunov
  proof.

- [09_review_of_suggestions.md](./09_review_of_suggestions.md)
  Audits the professor review in `suggestions.md`, keeps the correct critique,
  and rejects the invalid shortcut that would otherwise overstate the stochastic
  theorem.

## Policy Under Study

For service rates \(\mu_i > 0\), queue state \(q \in \mathbb R_+^N\), and
parameters \(\alpha > 0\), \(\beta > 0\), \(\gamma \in \mathbb R\), \(c \ge 0\),
the calibrated UAS routing probabilities are

\[
p_i(q)
=
\frac{\mu_i^\gamma \exp\!\left(-\alpha (q_i+c)/\mu_i^\beta\right)}
{\sum_{j=1}^N \mu_j^\gamma \exp\!\left(-\alpha (q_j+c)/\mu_j^\beta\right)}.
\]

Throughout these notes,

\[
\Lambda := \sum_{i=1}^N \mu_i,
\qquad
\lambda \in (0,\Lambda).
\]

## Research Position

These notes do **not** claim positive Harris recurrence of the full calibrated
family.

They do show that:

1. the corrected reflected deterministic model is explicit,
2. the benchmark numerical diagnostic already computed in the repo is
   consistent with that corrected model,
3. the equilibrium can be characterized exactly by a one-dimensional fixed-point
   equation,
4. the reflected ODE is a constrained convex gradient system,
5. every reflected-ODE trajectory converges to the same equilibrium,
6. the remaining theorem gap is now sharply identified.
