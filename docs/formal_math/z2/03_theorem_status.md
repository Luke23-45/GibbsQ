# Theorem Status And Remaining Gap

This note states, as cleanly as possible, what is already established and what
is still missing for a full queueing stability theorem for Calibrated UAS.

## 1. What Is Proved In `z2`

The notes in this folder prove the following facts.

**Proved fact A.**
The correct deterministic object is a reflected ODE on \(\mathbb R_+^N\), not
the unconstrained ODE \(\dot q = \lambda p(q)-\mu\).

**Proved fact B.**
When \(\lambda < \Lambda\), the reflected ODE has no strictly positive interior
equilibrium.

**Proved fact C.**
Any equilibrium must satisfy the complementarity conditions

\[
q_i^* \ge 0,
\qquad
\lambda p_i(q^*) \le \mu_i,
\qquad
q_i^*(\mu_i-\lambda p_i(q^*))=0.
\]

**Proved fact D.**
The equilibrium is determined by a scalar parameter \(K>0\) and has the
explicit coordinate form

\[
q_i^*
=
\max\!\left\{
0,
\frac{\mu_i^\beta}{\alpha}\log\!\left(\frac{\theta_i}{K}\right)
\right\},
\qquad
\theta_i=\mu_i^{\gamma-1}e^{-\alpha c/\mu_i^\beta}.
\]

**Proved fact E.**
The scalar consistency equation

\[
\lambda
=
\sum_{i=1}^N
\min\!\left\{\mu_i,\frac{\mu_i\theta_i}{K}\right\}
\]

has a unique solution \(K^*>0\), so the equilibrium is unique.

**Proved fact F.**
For the benchmark parameters used in the repo, that equilibrium coincides with
the stored numerical attractor from the deterministic diagnostic.

**Proved fact G.**
The reflected ODE has an explicit convex potential \(H\) with
\(\lambda p(q)-\mu=-D\nabla H(q)\), where
\(D=\operatorname{diag}(\mu_i^\beta)\).

**Proved fact H.**
The constrained potential \(H+I_{\mathbb R_+^N}\) is convex and coercive, has a
unique minimizer, and that minimizer is exactly the equilibrium \(q^*\).

**Proved fact I.**
Every reflected-ODE trajectory converges to \(q^*\). So the deterministic
reflected ODE is globally asymptotically stable.

## 2. What Is Not Yet Proved

The decisive remaining theorem is now stochastic rather than deterministic.

**Open stochastic theorem.**
The calibrated-UAS CTMC is positive Harris recurrent for every
\(\lambda < \Lambda\).

What blocks that step is not the deterministic ODE analysis anymore. It is the
fact that the smooth reflected ODE studied in `z2` is not yet identified as the
classical Dai fluid limit of the CTMC under standard queue-length scaling. See
[07_ctmc_scaling_gap.md](./07_ctmc_scaling_gap.md) and
[09_review_of_suggestions.md](./09_review_of_suggestions.md).

## 3. Why `x5.md` Was Too Strong

The note `docs/notes/x1/x5.md` moved too quickly from a numerical attractor to a
global theorem. The specific overclaims were:

- it treated the equilibrium as if it lived in the strictly positive orthant
- it ignored the reflection/complementarity terms at the boundary
- it started with an M-matrix Jacobian route that collapses once the row-sum
  structure is computed
- it transformed the dynamics to \(w\)-coordinates without closing the boundary
  argument for the reflected system
- it implicitly treated the smooth reflected ODE as if it were already the
  standard queueing fluid limit of the CTMC

Those are not small details. They are the difference between a correct theorem
and an incorrect one.

## 4. The Correct Next Theorem To Target

The mathematically correct next target is now:

**Candidate theorem.**
For fixed \(\alpha>0\), \(\beta>0\), \(\gamma \in \mathbb R\), \(c \ge 0\), and
\(\lambda < \Lambda\), the Calibrated-UAS CTMC is positive Harris recurrent,
with proof based on a correctly scaled stochastic limit or on a direct
Foster-Lyapunov argument.

## 5. What A Correct Proof Program Must Do

Any end-to-end theorem proof now has to handle four tasks explicitly.

### Task 1. Identify the correct large-scale stochastic regime

One must decide whether the right theorem should use standard fluid scaling,
parameter-rescaled fluid scaling, or a direct stochastic Lyapunov route.

### Task 2. Preserve the queueing policy under that regime

Because the softmax exponent contains the raw queue lengths, ordinary fluid
scaling changes the effective routing map. That issue must be handled honestly.

### Task 3. Build the stochastic stability argument

If the route is fluid-limit based, the scaled limit must be characterized
correctly. If the route is direct, a new Foster-Lyapunov function must be found.

### Task 4. Lift the deterministic insight without overclaiming

The convex potential \(H\) and the unique deterministic attractor are valuable
guides, but they do not automatically prove CTMC stability.

## 6. Practical Research Conclusion

The project should now stop running architecture searches and focus on the
stochastic theorem gap.

The current state of knowledge is:

- empirically, Calibrated UAS remains the best closed-form policy found in the
  repo
- mathematically, the deterministic reflected ODE is now clean and globally
  convergent
- the exact CTMC generator obstruction has been written down explicitly
- the professor review correctly validated that obstruction but did **not**
  close the CTMC theorem
- the only missing high-value result is a correct bridge from that deterministic
  picture to the original CTMC

That is the next direction.
