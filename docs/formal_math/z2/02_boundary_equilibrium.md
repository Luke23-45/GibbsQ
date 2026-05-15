# Boundary Equilibrium: Exact Scalar Characterization

This note derives an exact formula for the equilibrium of the reflected fluid
model. The main result is that the equilibrium reduces to a one-dimensional
fixed-point equation.

## 1. Threshold Constants

For each server \(i\), define

\[
\theta_i
:=
\mu_i^{\gamma-1}\exp\!\left(-\alpha c/\mu_i^\beta\right).
\]

These constants encode the queue-free routing weight relative to the service
rate.

## 2. Equilibrium Equations

At equilibrium \(q^*\), the reflected conditions from
[01_reflected_fluid_model.md](./01_reflected_fluid_model.md) are

\[
q_i^* \ge 0,
\qquad
\lambda p_i(q^*) \le \mu_i,
\qquad
q_i^* \bigl(\mu_i-\lambda p_i(q^*)\bigr)=0.
\]

Let

\[
w_i^* := \mu_i^\gamma
\exp\!\left(-\alpha (q_i^*+c)/\mu_i^\beta\right),
\qquad
W^* := \sum_{j=1}^N w_j^*,
\qquad
K := \frac{W^*}{\lambda}.
\]

Since \(p_i(q^*) = w_i^*/W^*\), the equilibrium complementarity conditions become

- if \(q_i^*>0\), then \(w_i^* = \mu_i K\),
- if \(q_i^*=0\), then \(w_i^* \le \mu_i K\).

## 3. Closed-Form Coordinate Formula

From the definition of \(w_i^*\),

\[
\mu_i^\gamma \exp\!\left(-\alpha (q_i^*+c)/\mu_i^\beta\right)
=
\min\!\left\{\mu_i K,\ \mu_i^\gamma e^{-\alpha c/\mu_i^\beta}\right\}.
\]

Dividing by \(\mu_i\) and using \(\theta_i\) gives

\[
\mu_i^{\gamma-1}\exp\!\left(-\alpha (q_i^*+c)/\mu_i^\beta\right)
=
\min\!\left\{K,\ \theta_i\right\}.
\]

Therefore,

\[
q_i^*
=
\max\!\left\{
0,
\frac{\mu_i^\beta}{\alpha}\log\!\left(\frac{\theta_i}{K}\right)
\right\}.
\]

This already shows that the active set is determined by the scalar \(K\):

\[
q_i^*>0
\quad\Longleftrightarrow\quad
K<\theta_i.
\]

## 4. Scalar Fixed-Point Equation

Substituting the coordinate formula back into the normalization equation
\(W^*=\lambda K\) yields

\[
\lambda K
=
\sum_{i=1}^N
\mu_i
\min\!\left\{K,\theta_i\right\}.
\]

Dividing by \(K>0\) gives the equivalent scalar equation

\[
\lambda
=
G(K)
:=
\sum_{i=1}^N
\min\!\left\{\mu_i,\ \frac{\mu_i\theta_i}{K}\right\}.
\]

This is the exact equilibrium condition.

## 5. Existence And Uniqueness

**Proposition 2.**
Assume \(\lambda < \Lambda := \sum_i \mu_i\). Then the scalar equation
\(\lambda = G(K)\) has a unique solution \(K^* > 0\).

**Proof.**
For each \(i\), the function

\[
K \mapsto \min\!\left\{\mu_i,\frac{\mu_i\theta_i}{K}\right\}
\]

is continuous and nonincreasing on \((0,\infty)\). Hence \(G\) is continuous
and nonincreasing on \((0,\infty)\).

Moreover,

\[
\lim_{K\downarrow 0} G(K)
=
\sum_{i=1}^N \mu_i
=
\Lambda
>
\lambda,
\]

because \(\mu_i\theta_i/K \to \infty\) for each \(i\), and

\[
\lim_{K\uparrow\infty} G(K)
=
0
<
\lambda,
\]

because \(\mu_i\theta_i/K \to 0\) for each \(i\).

By continuity, at least one solution \(K^*>0\) exists.

To prove uniqueness, let

\[
\theta_{\min}:=\min_{1\le i\le N}\theta_i.
\]

For \(0<K\le \theta_{\min}\), every term equals \(\mu_i\), so
\(G(K)=\Lambda\). Hence any solution of \(G(K)=\lambda<\Lambda\) must satisfy
\(K>\theta_{\min}\).

Now take \(K_2>K_1>\theta_{\min}\). Since \(K_1>\theta_{\min}\), at least one
index \(i\) satisfies \(\theta_i<K_1\), and for every such index
\[
\frac{\mu_i\theta_i}{K_2}<\frac{\mu_i\theta_i}{K_1}<\mu_i.
\]
That term strictly decreases from \(K_1\) to \(K_2\), while every other term is
nonincreasing. Therefore \(G(K_2)<G(K_1)\) on
\((\theta_{\min},\infty)\). Since every solution lies in this interval,
\(G(K)=\lambda\) has at most one solution. Together with existence, this gives
uniqueness. `QED`

## 6. Exact Equilibrium Formula

Combining the previous steps gives the equilibrium explicitly.

**Theorem 1 (proved equilibrium characterization).**
Assume \(\lambda < \Lambda\). Let \(K^* > 0\) be the unique solution of

\[
\lambda
=
\sum_{i=1}^N
\min\!\left\{\mu_i,\ \frac{\mu_i\theta_i}{K}\right\}.
\]

Then the reflected fluid equilibrium is uniquely determined by

\[
q_i^*
=
\max\!\left\{
0,
\frac{\mu_i^\beta}{\alpha}\log\!\left(\frac{\theta_i}{K^*}\right)
\right\},
\qquad i=1,\dots,N.
\]

The associated active set is

\[
A^* = \{i : K^* < \theta_i\}.
\]

For every \(i \in A^*\),

\[
\lambda p_i(q^*) = \mu_i.
\]

For every \(i \notin A^*\),

\[
\lambda p_i(q^*) \le \mu_i.
\]

## 7. Benchmark Check Against The Existing Diagnostic

For the benchmark parameters used in the repo,

\[
(\alpha,\beta,\gamma,c)=(20,0.85,0.5,0.5),
\]

\[
\mu=(0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3),
\qquad
\lambda=11.2,
\]

solving the scalar equation yields

\[
K^* \approx 2.963229688552612\times 10^{-4}.
\]

Plugging that value into the equilibrium formula gives

\[
q^*
\approx
(0,0,0,0,0,0.05904346,0.11688593,0.17325504,0.22833526,0.28227270).
\]

This matches the numerical attractor recorded in the current
[boundary_equilibrium_verification_20260515_091832.csv](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/data/boundary_equilibrium_verification_20260515_091832.csv)
artifact to machine precision.

So the current numerical evidence is fully consistent with the corrected
boundary-equilibrium formula.
