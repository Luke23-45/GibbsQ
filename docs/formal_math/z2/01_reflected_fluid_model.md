# Reflected Fluid Model

This note records the corrected reflected deterministic model for Reflected UAS
and proves the first structural fact needed for any stability theorem: the
equilibrium cannot be an interior fixed point when \(\lambda < \Lambda\).

For continuity with the queueing literature, these notes still occasionally use
the phrase "fluid model" for this reflected ODE surrogate. The separate issue of
whether it is the **classical** fluid limit of the CTMC under standard scaling
is addressed later in [07_ctmc_scaling_gap.md](./07_ctmc_scaling_gap.md).

## 1. Routing Map

Fix \(N \ge 1\), service rates \(\mu_i > 0\), and parameters
\(\alpha > 0\), \(\beta > 0\), \(\gamma \in \mathbb R\), \(c \ge 0\).
For \(q \in \mathbb R_+^N\), define

\[
w_i(q) := \mu_i^\gamma \exp\!\left(-\alpha (q_i+c)/\mu_i^\beta\right),
\qquad
W(q) := \sum_{j=1}^N w_j(q),
\]

and

\[
p_i(q) := \frac{w_i(q)}{W(q)}.
\]

Because every \(w_i(q)\) is strictly positive, the routing map satisfies

\[
p_i(q) > 0,
\qquad
\sum_{i=1}^N p_i(q)=1,
\qquad
p \in C^\infty(\mathbb R_+^N).
\]

## 2. Correct Fluid Model

The unconstrained drift

\[
b_i(q) := \lambda p_i(q) - \mu_i
\]

is not itself the correct fluid model on \(\mathbb R_+^N\), because it can point
outside the orthant when some coordinates are zero.

The corrected object is the reflected fluid model

\[
q_i(t)
=
q_i(0)
+
\int_0^t b_i(q(s))\,ds
+
y_i(t),
\qquad i=1,\dots,N,
\]

subject to

\[
q_i(t)\ge 0,
\qquad
y_i(\cdot)\ \text{nondecreasing},
\qquad
y_i(0)=0,
\]

and the complementarity condition

\[
\int_0^\infty q_i(s)\,dy_i(s)=0.
\]

The last condition means that the regulator \(y_i\) can increase only while the
coordinate \(q_i\) is pinned at zero.

## 3. Interior Dynamics

On any time interval where \(q_i(t)>0\) for every \(i\), the regulators are
constant, so the reflected system reduces to the ordinary differential equation

\[
\dot q_i(t) = \lambda p_i(q(t)) - \mu_i.
\]

Therefore the naive ODE is locally valid on active faces, but it is not the
global fluid model.

## 4. Total Drift Identity

Summing the reflected equations gives

\[
\sum_{i=1}^N q_i(t)
=
\sum_{i=1}^N q_i(0)
+
\int_0^t \left(\lambda \sum_{i=1}^N p_i(q(s)) - \sum_{i=1}^N \mu_i\right)\,ds
+
\sum_{i=1}^N y_i(t).
\]

Since \(\sum_i p_i(q)=1\), this becomes

\[
\sum_{i=1}^N q_i(t)
=
\sum_{i=1}^N q_i(0)
+
(\lambda-\Lambda)t
+
\sum_{i=1}^N y_i(t),
\qquad
\Lambda := \sum_{i=1}^N \mu_i.
\]

In differential form, on intervals where derivatives exist,

\[
\frac{d}{dt}\sum_{i=1}^N q_i(t)
=
\lambda-\Lambda + \sum_{i=1}^N \dot y_i(t).
\]

## 5. Interior Equilibrium Is Impossible

**Proposition 1.**
Assume \(\lambda < \Lambda\). The reflected fluid model has no equilibrium
\(q^* \in (0,\infty)^N\) with all coordinates strictly positive.

**Proof.**
If \(q^* \in (0,\infty)^N\), then every regulator is locally constant at
equilibrium, so the equilibrium equations would be

\[
0 = \lambda p_i(q^*) - \mu_i,
\qquad i=1,\dots,N.
\]

Summing over \(i\) yields

\[
0
=
\lambda \sum_{i=1}^N p_i(q^*) - \sum_{i=1}^N \mu_i
=
\lambda-\Lambda.
\]

This contradicts \(\lambda < \Lambda\). Therefore no equilibrium can lie in the
strictly positive orthant. `QED`

**Corollary 1.**
Any fluid equilibrium must lie on a boundary face of \(\mathbb R_+^N\).

This is exactly what the existing numerical diagnostic in
[direction1_fluid.json](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/direction12_probe/run_20260512_194636/artifacts/direction1_fluid.json)
shows: the first five coordinates are zero at the attracting state.

## 6. Equilibrium Conditions With Reflection

Let \(q^*\) be an equilibrium of the reflected fluid model. Then there exist
constants \(r_i^* \ge 0\) representing the regulator rates such that

\[
0 = \lambda p_i(q^*) - \mu_i + r_i^*,
\qquad i=1,\dots,N,
\]

with complementarity

\[
q_i^* \ge 0,
\qquad
r_i^* \ge 0,
\qquad
q_i^* r_i^* = 0.
\]

Equivalently,

- if \(q_i^*>0\), then \(r_i^*=0\) and \(\lambda p_i(q^*)=\mu_i\),
- if \(q_i^*=0\), then \(r_i^*=\mu_i-\lambda p_i(q^*)\ge 0\), so
  \(\lambda p_i(q^*)\le \mu_i\).

These are the correct equilibrium equations for the queueing fluid model.

## 7. Interior Weight Dynamics On An Active Face

For later use, define the transformed coordinates

\[
w_i(t) := \mu_i^\gamma \exp\!\left(-\alpha (q_i(t)+c)/\mu_i^\beta\right).
\]

Whenever \(q_i(t)>0\) on an interval, differentiation gives

\[
\dot q_i(t)
=
-\frac{\mu_i^\beta}{\alpha}\frac{\dot w_i(t)}{w_i(t)}.
\]

Substituting the interior ODE \(\dot q_i = \lambda w_i/W - \mu_i\) yields

\[
\dot w_i(t)
=
-\frac{\alpha}{\mu_i^\beta} w_i(t)
\left(
\lambda \frac{w_i(t)}{W(t)} - \mu_i
\right).
\]

Equivalently,

\[
\dot w_i(t)
=
\frac{\alpha}{\mu_i^\beta} w_i(t)
\left(
\mu_i - \lambda \frac{w_i(t)}{W(t)}
\right).
\]

This transformed dynamics is correct on active faces, but it is not by itself a
complete global model because the reflected boundary still has to be handled
separately.
