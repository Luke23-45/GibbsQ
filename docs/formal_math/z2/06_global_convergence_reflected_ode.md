# Global Convergence Of The Reflected ODE

This note completes the deterministic part of the theory. It proves that the
reflected reflected-UAS ODE converges globally to the unique boundary
equilibrium characterized earlier.

## 1. Projected Drift Form

Let

\[
F(q):=\lambda p(q)-\mu = -D\nabla H(q),
\]

with \(D\) and \(H\) defined in
[05_projected_gradient_structure.md](./05_projected_gradient_structure.md).

For the orthant \(C=\mathbb R_+^N\), the reflected dynamics can be written
coordinatewise as

\[
\dot q_i =
\Gamma_i(q)
:=
\begin{cases}
F_i(q), & q_i>0,\\
\max\{F_i(q),0\}, & q_i=0.
\end{cases}
\]

This is the standard orthant reflection of the unconstrained drift.

Because \(p(\cdot)\) is smooth on \(C\), \(F\) is globally Lipschitz on every
compact subset of \(C\). Standard Skorokhod-map results for orthant reflection
therefore give a unique global absolutely continuous solution for each initial
state \(q(0)\in C\). The convergence argument below applies to any such
solution.

## 2. Lyapunov Descent

Let \(q(\cdot)\) be a reflected trajectory. For almost every \(t\),

\[
\frac{d}{dt}H(q(t))
=
\sum_{i=1}^N \partial_i H(q(t))\,\dot q_i(t).
\]

Using \(F_i(q)=-\mu_i^\beta \partial_i H(q)\), the projected drift gives

\[
\dot q_i(t)
=
\begin{cases}
-\mu_i^\beta \partial_i H(q(t)), &
q_i(t)>0,\\
-\mu_i^\beta \partial_i H(q(t)), &
q_i(t)=0 \text{ and } \partial_i H(q(t))<0,\\
0, &
q_i(t)=0 \text{ and } \partial_i H(q(t))\ge 0.
\end{cases}
\]

Therefore

\[
\frac{d}{dt}H(q(t))
=
-\sum_{i:\,\dot q_i(t)\ne 0}
\mu_i^\beta \bigl(\partial_i H(q(t))\bigr)^2
\le 0.
\]

So \(H\) is a Lyapunov function for the reflected ODE.

## 3. When Is The Descent Strict?

The derivative vanishes at time \(t\) if and only if every coordinate satisfies
one of the two conditions:

- \(q_i(t)>0\) and \(\partial_i H(q(t))=0\),
- \(q_i(t)=0\) and \(\partial_i H(q(t))\ge 0\).

By Proposition 3 in
[05_projected_gradient_structure.md](./05_projected_gradient_structure.md),
these are exactly the complementarity conditions

\[
q_i(t)\ge 0,
\qquad
\lambda p_i(q(t))\le \mu_i,
\qquad
q_i(t)(\mu_i-\lambda p_i(q(t)))=0.
\]

Hence:

**Lemma 1.**
\(\frac{d}{dt}H(q(t))=0\) if and only if \(q(t)\) is an equilibrium of the
reflected ODE.

Because the equilibrium is unique, the only stationary point of \(H\) on the
orthant is \(q^*\).

## 4. Boundedness Of Every Trajectory

Fix any initial state \(q(0)\in C\). Since \(H(q(t))\) is nonincreasing,

\[
H(q(t))\le H(q(0))
\qquad\text{for all }t\ge 0.
\]

By coercivity of \(H\) from
[05_projected_gradient_structure.md](./05_projected_gradient_structure.md), the
sublevel set

\[
\{q\in C: H(q)\le H(q(0))\}
\]

is compact. Therefore every trajectory is bounded and precompact.

## 5. Every Limit Point Is An Equilibrium

Let \(z\) be any accumulation point of the trajectory \(q(t)\). We claim that
\(z\) must be an equilibrium.

Assume the contrary. Then at least one coordinate \(i\) violates the
equilibrium complementarity conditions. There are only two possibilities.

### Case 1. \(z_i>0\) and \(\partial_i H(z)\ne 0\)

By continuity of \(q\mapsto \partial_i H(q)\), there exist \(\delta>0\),
\(\eta>0\), and a neighborhood \(U\) of \(z\) such that for every \(q\in U\),

\[
q_i\ge \delta,
\qquad
|\partial_i H(q)|\ge \eta.
\]

Hence for every \(q\in U\),

\[
\frac{d}{dt}H(q(t))\le -\mu_i^\beta \eta^2
\]

whenever \(q(t)=q\).

### Case 2. \(z_i=0\) and \(\partial_i H(z)<0\)

Again by continuity, there exist \(\eta>0\) and a neighborhood \(U\) of \(z\)
such that for every \(q\in U\),

\[
\partial_i H(q)\le -\eta.
\]

If \(q_i>0\), then the \(i\)-th contribution to \(\dot H\) is
\(-\mu_i^\beta (\partial_i H(q))^2\le -\mu_i^\beta\eta^2\). If \(q_i=0\), then
the reflection rule still gives
\(\dot q_i=-\mu_i^\beta \partial_i H(q)\ge \mu_i^\beta\eta>0\), so the same
negative contribution appears. Thus for every \(q\in U\),

\[
\frac{d}{dt}H(q(t))\le -\mu_i^\beta \eta^2
\]

whenever \(q(t)=q\).

### Consequence

In either case, there is a neighborhood \(U\) of \(z\) and a constant
\(\varepsilon>0\) such that

\[
\frac{d}{dt}H(q(t))\le -\varepsilon
\]

whenever \(q(t)\in U\).

Now choose a smaller neighborhood \(V\) with compact closure
\(\overline V\subset U\). Since \(z\) is an accumulation point, there exists a
sequence \(t_n\to\infty\) with \(q(t_n)\in V\).

The projected drift is globally bounded because \(0\le \lambda p_i(q)\le\lambda\)
for every \(i\), hence

\[
|\dot q_i(t)|\le \lambda+\mu_i
\qquad\text{for almost every }t.
\]

Therefore there is a finite constant \(M>0\) such that
\(\|\dot q(t)\|_2\le M\) for almost every \(t\). Let

\[
\rho := \operatorname{dist}(\overline V,U^c)>0.
\]

Whenever the trajectory hits \(V\), it must remain inside \(U\) for at least
\(\rho/M\) units of time before it can exit \(U\). Hence each visit to \(V\)
decreases \(H\) by at least \(\varepsilon\rho/M\).

Because \(q(t_n)\in V\) for infinitely many \(n\), this would force
\(H(q(t))\) to decrease by an unbounded total amount along an infinite
subsequence. That is impossible because \(H\) is bounded below by \(H(q^*)\).
This contradiction proves the claim.

So every accumulation point is an equilibrium, hence every accumulation point is
the unique point \(q^*\).

## 6. Global Convergence Theorem

Because the trajectory is precompact and all of its accumulation points equal
\(q^*\), the full trajectory converges to \(q^*\).

**Theorem 2 (proved deterministic convergence theorem).**
Fix \(\alpha>0\), \(\beta>0\), \(\gamma\in\mathbb R\), \(c\ge 0\), service rates
\(\mu_i>0\), and \(\lambda<\Lambda=\sum_i\mu_i\). Then the reflected
reflected-UAS ODE on \(\mathbb R_+^N\) has a unique equilibrium \(q^*\), and
every reflected trajectory satisfies

\[
\lim_{t\to\infty} q(t)=q^*.
\]

So the deterministic reflected ODE is globally asymptotically stable.

## 7. What This Theorem Does And Does Not Give

This theorem closes the deterministic part of the project. It proves global
convergence of the reflected ODE surrogate that we have analyzed throughout
`z2`.

It does **not** by itself imply positive Harris recurrence of the original
Reflected-UAS CTMC. That extra step requires identifying the same ODE as the
correct large-scale fluid limit of the stochastic queueing process under an
appropriate scaling regime. The scaling issue is discussed separately in
[07_ctmc_scaling_gap.md](./07_ctmc_scaling_gap.md).
