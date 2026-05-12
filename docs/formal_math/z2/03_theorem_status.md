# Theorem Status And Claim Ladder

This note is the authoritative status file for the `z2` package.

Its job is to state, without sales language, which parts of the project are:

- fully proved,
- conditional / promotion-sensitive,
- empirically validated,
- or archival only.

## 1. Fully Proved Deterministic Results

The following statements are established within `z2`.

**Proved fact A.**
The correct deterministic object associated with Calibrated UAS is a reflected
ODE on \(\mathbb R_+^N\), not the unconstrained equation
\(\dot q=\lambda p(q)-\mu\).

**Proved fact B.**
When \(\lambda<\Lambda\), the reflected ODE has no strictly positive interior
equilibrium.

**Proved fact C.**
Any reflected equilibrium must satisfy the complementarity conditions

\[
q_i^* \ge 0,
\qquad
\lambda p_i(q^*) \le \mu_i,
\qquad
q_i^*(\mu_i-\lambda p_i(q^*))=0.
\]

**Proved fact D.**
The equilibrium is uniquely determined by a scalar parameter \(K^*>0\) and has
the explicit coordinate form

\[
q_i^*
=
\max\!\left\{
0,
\frac{\mu_i^\beta}{\alpha}\log\!\left(\frac{\theta_i}{K^*}\right)
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
The reflected ODE has an explicit convex potential \(H\) with
\(\lambda p(q)-\mu=-D\nabla H(q)\), where
\(D=\operatorname{diag}(\mu_i^\beta)\).

**Proved fact G.**
The constrained potential \(H+I_{\mathbb R_+^N}\) is convex and coercive, has a
unique minimizer, and that minimizer coincides with the unique equilibrium
\(q^*\).

**Proved fact H.**
Every reflected-ODE trajectory converges to \(q^*\). The deterministic
reflected ODE is therefore globally asymptotically stable.

**Proved fact I.**
For the benchmark parameter point used in the repo, the exact equilibrium
formula agrees with the stored deterministic numerical attractor.

These results are carried by files `01`, `02`, `05`, and `06`.

## 2. Fully Established Correction Results

The package also establishes two negative-but-important mathematical facts.

**Correction fact I.**
The smooth reflected ODE analyzed in `z2` is not automatically identified with
the classical fixed-parameter CTMC fluid limit under standard queue-length
scaling.

**Correction fact J.**
For the potential \(H\), the exact CTMC generator contains a genuine boundary
mismatch relative to the reflected-ODE Lyapunov picture.

These results mean that the older shortcut from deterministic convergence to
stochastic stability was too strong.

These correction results are carried by files `07`, `08`, and `09`.

## 3. Conditional Stochastic Route

The package contains a direct fixed-parameter CTMC certification route in files
`10` and `11`.

Its status is:

**Conditional fact K.**
The weighted-quadratic Foster-Lyapunov route for the fixed-parameter
Calibrated-UAS CTMC has passed internal audit and targeted numerical
validation, and it is a viable candidate route to positive Harris recurrence
under the natural load condition \(\lambda<\Lambda\).

At the current package level, this route should be treated as:

- **audited certification material**, or
- **promotion-sensitive theorem material**

unless and until explicit theorem-level sign-off is recorded.

This note therefore does **not** automatically elevate file `10` to the same
status as the deterministic theorem core.

## 4. What Is Not A Main-Line Theorem Claim

The following may appear elsewhere in the repo, but they are not theorem-core
claims for `z2`:

- exploratory policy directions such as SMVR
- failed or superseded certification routes such as the older SCUAS line
- neural-policy performance narratives
- broad claims that the whole calibrated family is already beyond theorem risk

Those items may still exist as archive, empirical context, or applicability
material, but they are not the foundation of this package.

## 5. Supporting Applicability Claim

The package also supports one limited applicability claim:

**Supporting fact L.**
Because Calibrated UAS is softmax-based and continuous, it provides a
differentiable routing law that is compatible with gradient-based policy
learning in a way that hard dispatch rules such as JSQ and JSSQ are not.

This is useful supporting material, but it is not the main hero of the thesis
and must not be used as a substitute for the theorem core.

## 6. Practical Reading Rule

When writing from `z2`, use the following discipline:

- files `01`, `02`, `05`, `06`: theorem core
- files `07`, `08`, `09`: correction / obstruction core
- files `10`, `11`: conditional stochastic route
- files `12`, `13`: claim-control layer
- file `suggestions.md`: archive only

That separation is what keeps the thesis mathematically honest.
