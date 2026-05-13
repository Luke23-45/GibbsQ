# Exact CTMC Generator For The Potential \(H\)

This note pushes the project one step further on the stochastic side. It
computes the exact CTMC generator acting on the same potential \(H\) used in the
deterministic reflected-ODE analysis and shows precisely where the two drift
pictures diverge.

## 1. The Queueing Generator

Let \(Q\in\mathbb Z_+^N\). Under Reflected UAS, the continuous-time Markov
chain has generator

\[
(\mathcal L f)(Q)
=
\lambda\sum_{i=1}^N p_i(Q)\bigl[f(Q+e_i)-f(Q)\bigr]
+
\sum_{i=1}^N \mu_i \mathbf 1_{\{Q_i>0\}}\bigl[f(Q-e_i)-f(Q)\bigr].
\]

We apply this generator to the same potential

\[
H(Q)
:=
\sum_{i=1}^N \mu_i^{1-\beta} Q_i
+
\frac{\lambda}{\alpha}\log W(Q),
\]

with

\[
W(Q)=\sum_{j=1}^N \mu_j^\gamma
\exp\!\left(-\alpha (Q_j+c)/\mu_j^\beta\right).
\]

Define

\[
a_i := \frac{\alpha}{\mu_i^\beta}.
\]

## 2. Exact One-Step Increment Formulas

Because only the \(i\)-th coordinate changes in \(Q\pm e_i\), the normalizing
sum \(W\) changes explicitly.

### Arrival increment

\[
W(Q+e_i)
=
W(Q) + w_i(Q)\bigl(e^{-a_i}-1\bigr)
=
W(Q)\Bigl[1-p_i(Q)\bigl(1-e^{-a_i}\bigr)\Bigr].
\]

Therefore

\[
H(Q+e_i)-H(Q)
=
\mu_i^{1-\beta}
+
\frac{\lambda}{\alpha}
\log\!\Bigl[1-p_i(Q)\bigl(1-e^{-a_i}\bigr)\Bigr].
\]

### Service increment

If \(Q_i>0\), then

\[
W(Q-e_i)
=
W(Q) + w_i(Q)\bigl(e^{a_i}-1\bigr)
=
W(Q)\Bigl[1+p_i(Q)\bigl(e^{a_i}-1\bigr)\Bigr].
\]

Hence

\[
H(Q-e_i)-H(Q)
=
-\mu_i^{1-\beta}
+
\frac{\lambda}{\alpha}
\log\!\Bigl[1+p_i(Q)\bigl(e^{a_i}-1\bigr)\Bigr].
\]

These formulas are exact.

## 3. Exact Generator Identity

Substituting the increments into the generator yields

\[
(\mathcal L H)(Q)
=
\lambda\sum_{i=1}^N p_i(Q)
\left[
\mu_i^{1-\beta}
+
\frac{\lambda}{\alpha}
\log\!\Bigl(1-p_i(Q)(1-e^{-a_i})\Bigr)
\right]
\]
\[
+\;
\sum_{i=1}^N \mu_i \mathbf 1_{\{Q_i>0\}}
\left[
-\mu_i^{1-\beta}
+
\frac{\lambda}{\alpha}
\log\!\Bigl(1+p_i(Q)(e^{a_i}-1)\Bigr)
\right].
\]

This identity is exact and requires no scaling or approximation.

## 4. First-Order Decomposition Around The Deterministic Drift

From
[05_projected_gradient_structure.md](./05_projected_gradient_structure.md),

\[
\partial_i H(Q)=\frac{\mu_i-\lambda p_i(Q)}{\mu_i^\beta}.
\]

Apply the one-dimensional Taylor formula along the \(i\)-th coordinate:

\[
H(Q+e_i)-H(Q)=\partial_i H(Q)+r_i^+(Q),
\]

\[
H(Q-e_i)-H(Q)=-\partial_i H(Q)+r_i^-(Q)
\qquad (Q_i>0).
\]

Because the Hessian of \(H\) is globally bounded on the orthant, there exists a
finite constant

\[
B_H := \sup_{q\in\mathbb R_+^N}\max_{1\le i\le N}\partial_{ii}^2 H(q)
\le
\frac{\lambda\alpha}{\mu_{\min}^{2\beta}},
\qquad
\mu_{\min}:=\min_i \mu_i,
\]

such that

\[
|r_i^+(Q)|\le \frac{B_H}{2},
\qquad
|r_i^-(Q)|\le \frac{B_H}{2}.
\]

Therefore

\[
(\mathcal L H)(Q)
=
\sum_{i=1}^N \bigl(\lambda p_i(Q)-\mu_i\mathbf 1_{\{Q_i>0\}}\bigr)\partial_i H(Q)
+
R(Q),
\]

with remainder bound

\[
|R(Q)|\le \frac{\lambda+\Lambda}{2}B_H.
\]

Now separate positive and zero coordinates.

For \(Q_i>0\),

\[
\lambda p_i(Q)-\mu_i = -\mu_i^\beta \partial_i H(Q),
\]

so

\[
\bigl(\lambda p_i(Q)-\mu_i\bigr)\partial_i H(Q)
=
-\mu_i^\beta \bigl(\partial_i H(Q)\bigr)^2.
\]

For \(Q_i=0\), the service term is absent and only the arrival term remains:

\[
\bigl(\lambda p_i(Q)-0\bigr)\partial_i H(Q)
=
\lambda p_i(Q)\partial_i H(Q).
\]

Thus:

\[
(\mathcal L H)(Q)
=
-\sum_{i:\,Q_i>0}\mu_i^\beta \bigl(\partial_i H(Q)\bigr)^2
+
\sum_{i:\,Q_i=0}\lambda p_i(Q)\partial_i H(Q)
+
R(Q).
\]

This is the key stochastic decomposition.

## 5. The Boundary Mismatch

Compare the last identity with the reflected-ODE Lyapunov derivative in
[06_global_convergence_reflected_ode.md](./06_global_convergence_reflected_ode.md).

For the deterministic reflected ODE:

- if \(q_i=0\) and \(\partial_i H(q)\ge 0\), the \(i\)-th coordinate is clipped
  and contributes `0` to \(\dot H\)
- if \(q_i=0\) and \(\partial_i H(q)<0\), the \(i\)-th coordinate moves inward
  and contributes a negative square term

For the CTMC generator:

- if \(Q_i=0\) and \(\partial_i H(Q)\ge 0\), the \(i\)-th term contributes the
  **positive** quantity \(\lambda p_i(Q)\partial_i H(Q)\)
- if \(Q_i=0\) and \(\partial_i H(Q)<0\), the \(i\)-th term contributes a
  negative quantity

So the deterministic reflection mechanism and the lattice generator agree on
strictly positive coordinates, but they disagree on boundary coordinates with
\(\partial_i H\ge 0\).

That is the precise obstruction behind the current theorem gap.

## 6. What This Does And Does Not Show

This note does **not** prove that \(H\) fails as a Foster-Lyapunov function. The
negative interior square terms may still dominate the positive boundary terms
outside a compact set.

What it does prove is narrower and more important:

- the deterministic reflected-ODE descent of \(H\) is not the exact stochastic
  generator drift
- any direct Foster-Lyapunov proof must control the extra boundary term

\[
\sum_{i:\,Q_i=0}\lambda p_i(Q)\partial_i H(Q),
\]

and the bounded second-order remainder \(R(Q)\)

## 7. Immediate Research Consequence

The project now has a sharper decision tree.

### Route A. Try to close the direct generator drift

One can try to prove that the negative interior term dominates the positive
boundary term plus \(R(Q)\) outside a compact set.

### Route B. Change the stochastic scaling argument

One can try to derive a large-scale limit in which the boundary clipping in the
deterministic ODE emerges naturally from the stochastic model.

### Route C. Use a different CTMC Lyapunov function

One can keep the deterministic theory as guidance but choose a lattice Lyapunov
function whose generator handles boundary states more naturally.

This is the correct point to continue from. The obstruction is now explicit
rather than guessed.
