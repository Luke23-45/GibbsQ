# Direct CTMC Stability Proof Attempt Via A Weighted Quadratic Lyapunov Function

This note gives a direct Foster-Lyapunov proof attempt for the fixed-parameter
Calibrated-UAS CTMC. Unlike the `H`-based route, this argument uses a weighted
quadratic Lyapunov function and an exact softmax minimum bound.

The note is intentionally separated from the deterministic theorem core.
Within the current package, it should be read as a proof-facing stochastic
certification route whose promotion to finished theorem status depends on the
audit outcome and explicit sign-off recorded elsewhere in `z2`.

## 1. CTMC Model

Let \(Q(t)\in\mathbb Z_+^N\) be the queue-length CTMC with:

- Poisson arrivals of rate \(\lambda>0\),
- exponential service at server \(i\) with rate \(\mu_i>0\),
- routing probabilities

\[
p_i(Q)
=
\frac{\mu_i^\gamma \exp\!\left(-\alpha (Q_i+c)/\mu_i^\beta\right)}
{\sum_{j=1}^N \mu_j^\gamma \exp\!\left(-\alpha (Q_j+c)/\mu_j^\beta\right)},
\qquad i=1,\dots,N,
\]

where

\[
\alpha>0,\qquad \beta>0,\qquad \gamma\in\mathbb R,\qquad c\ge 0.
\]

Define

\[
\Lambda := \sum_{i=1}^N \mu_i.
\]

Because \(p_i(Q)>0\) for every \(i,Q\), the chain is irreducible on
\(\mathbb Z_+^N\). Because the total jump rate satisfies

\[
\lambda + \sum_{i=1}^N \mu_i \mathbf 1_{\{Q_i>0\}}
\le \lambda+\Lambda,
\]

the chain is non-explosive.

## 2. Lyapunov Function

Consider

\[
V(Q):=\frac12\sum_{i=1}^N \frac{Q_i^2}{\mu_i^\beta}.
\]

This function is norm-like on \(\mathbb Z_+^N\): if \(|Q|_1\to\infty\), then
\(V(Q)\to\infty\).

## 3. Exact Generator Identity

Let \(\mathcal L\) denote the CTMC generator. For an arrival to coordinate
\(i\),

\[
V(Q+e_i)-V(Q)=\frac{Q_i+1/2}{\mu_i^\beta}.
\]

For a service completion at coordinate \(i\) with \(Q_i>0\),

\[
V(Q-e_i)-V(Q)=-\frac{Q_i-1/2}{\mu_i^\beta}.
\]

Therefore

\[
(\mathcal L V)(Q)
=
\lambda\sum_{i=1}^N p_i(Q)\frac{Q_i+1/2}{\mu_i^\beta}
-
\sum_{i=1}^N \mu_i \mathbf 1_{\{Q_i>0\}}\frac{Q_i-1/2}{\mu_i^\beta}.
\]

Equivalently,

\[
(\mathcal L V)(Q)
=
\lambda\sum_{i=1}^N p_i(Q)\frac{Q_i}{\mu_i^\beta}
-
\sum_{i=1}^N \mu_i^{1-\beta}Q_i
+
R_0(Q),
\]

where

\[
R_0(Q)
:=
\frac{\lambda}{2}\sum_{i=1}^N p_i(Q)\mu_i^{-\beta}
+
\frac12\sum_{i=1}^N \mu_i^{1-\beta}\mathbf 1_{\{Q_i>0\}}.
\]

Since \(\sum_i p_i(Q)=1\),

\[
R_0(Q)\le
\frac{\lambda}{2}\max_{1\le i\le N}\mu_i^{-\beta}
+
\frac12\sum_{i=1}^N \mu_i^{1-\beta}
=: C_0.
\]

So the only nontrivial term is the arrival contribution
\(\sum_i p_i(Q)Q_i/\mu_i^\beta\).

## 4. Exact Softmax Reformulation

Define the shifted energies

\[
s_i(Q):=\frac{Q_i}{\mu_i^\beta}+\kappa_i,
\qquad
\kappa_i:=\frac{c}{\mu_i^\beta}-\frac{\gamma}{\alpha}\log \mu_i.
\]

Then

\[
p_i(Q)
=
\frac{e^{-\alpha s_i(Q)}}{\sum_{j=1}^N e^{-\alpha s_j(Q)}}.
\]

So Calibrated UAS is exactly an ordinary softmax over the energies \(s_i(Q)\).

## 5. Softmax Minimum Bound

For any vector \(x\in\mathbb R^N\), define

\[
\pi_i(x):=\frac{e^{-\alpha x_i}}{\sum_{j=1}^N e^{-\alpha x_j}}.
\]

The standard entropy variational identity gives

\[
\sum_{i=1}^N \pi_i(x)x_i
+
\frac{1}{\alpha}\sum_{i=1}^N \pi_i(x)\log \pi_i(x)
=
-\frac{1}{\alpha}\log\sum_{j=1}^N e^{-\alpha x_j}.
\]

Since entropy is bounded above by \(\log N\),

\[
\sum_{i=1}^N \pi_i(x)x_i
\le
\min_{1\le i\le N} x_i + \frac{\log N}{\alpha}.
\]

Applying this with \(x_i=s_i(Q)\) yields

\[
\sum_{i=1}^N p_i(Q)s_i(Q)
\le
\min_{1\le i\le N} s_i(Q) + \frac{\log N}{\alpha}.
\]

Now define

\[
m(Q):=\min_{1\le i\le N}\frac{Q_i}{\mu_i^\beta}.
\]

Because

\[
\min_i s_i(Q)\le m(Q)+\max_i \kappa_i
\]

and

\[
\sum_{i=1}^N p_i(Q)\kappa_i \ge \min_i \kappa_i,
\]

we obtain

\[
\sum_{i=1}^N p_i(Q)\frac{Q_i}{\mu_i^\beta}
=
\sum_{i=1}^N p_i(Q)s_i(Q)-\sum_{i=1}^N p_i(Q)\kappa_i
\le
m(Q) + C_1,
\]

where

\[
C_1 := \frac{\log N}{\alpha} + \max_i \kappa_i - \min_i \kappa_i.
\]

This is the key bound.

## 6. Drift Decomposition

Substitute the bound into the generator formula:

\[
(\mathcal L V)(Q)
\le
\lambda m(Q)
-
\sum_{i=1}^N \mu_i^{1-\beta}Q_i
+
\lambda C_1 + C_0.
\]

Now decompose each coordinate as

\[
Q_i=\mu_i^\beta m(Q)+\Delta_i(Q),
\qquad
\Delta_i(Q)\ge 0.
\]

Then

\[
\sum_{i=1}^N \mu_i^{1-\beta}Q_i
=
\sum_{i=1}^N \mu_i^{1-\beta}\bigl(\mu_i^\beta m(Q)+\Delta_i(Q)\bigr)
=
\Lambda m(Q)
+
\sum_{i=1}^N \mu_i^{1-\beta}\Delta_i(Q).
\]

Hence

\[
(\mathcal L V)(Q)
\le
-(\Lambda-\lambda)m(Q)
-
\sum_{i=1}^N \mu_i^{1-\beta}\Delta_i(Q)
+
R,
\]

where

\[
R:=\lambda C_1 + C_0.
\]

## 7. Conversion To A Linear Norm Drift

Because

\[
|Q|_1
=
\sum_{i=1}^N Q_i
=
\left(\sum_{i=1}^N \mu_i^\beta\right)m(Q)
+
\sum_{i=1}^N \Delta_i(Q),
\]

we may choose

\[
\varepsilon
:=
\min\!\left\{
\frac{\Lambda-\lambda}{\sum_{i=1}^N \mu_i^\beta},
\min_{1\le i\le N}\mu_i^{1-\beta}
\right\}
>0.
\]

Then

\[
\varepsilon |Q|_1
\le
(\Lambda-\lambda)m(Q)
+
\sum_{i=1}^N \mu_i^{1-\beta}\Delta_i(Q).
\]

Therefore

\[
(\mathcal L V)(Q)\le -\varepsilon |Q|_1 + R.
\]

This is the required Foster-Lyapunov inequality.

## 8. Conditional Theorem Statement

**Theorem 3 (direct CTMC stability for Calibrated UAS).**
Fix

\[
\alpha>0,\qquad \beta>0,\qquad \gamma\in\mathbb R,\qquad c\ge 0,
\qquad
\mu_i>0,
\qquad
\lambda<\Lambda:=\sum_{i=1}^N \mu_i.
\]

Then the Calibrated-UAS queue-length CTMC is non-explosive, irreducible, and
positive Harris recurrent.

**Proof.**
Non-explosion and irreducibility were established in Section 1. The Lyapunov
function \(V\) is norm-like, and Sections 3-7 proved

\[
(\mathcal L V)(Q)\le -\varepsilon |Q|_1 + R
\]

for some constants \(\varepsilon>0\) and \(R<\infty\). Hence

\[
(\mathcal L V)(Q)\le -1
\]

outside the finite set

\[
\left\{Q\in\mathbb Z_+^N:\ \varepsilon |Q|_1\le R+1\right\}.
\]

By the continuous-time Foster-Lyapunov criterion, the CTMC is positive Harris
recurrent. `QED`

Within the current package, this theorem statement should be read together with
the promotion-sensitive status note in
[03_theorem_status.md](./03_theorem_status.md) and the audit note in
[11_audit_of_direct_ctmc_quadratic_proof.md](./11_audit_of_direct_ctmc_quadratic_proof.md).

## 9. Why This Bypasses The Earlier Failures

This proof does **not** use:

- the Jensen bound that led to the old SCUAS sufficient condition,
- the deterministic reflected-ODE Lyapunov function \(H\),
- any fluid-limit scaling argument.

Instead, it uses the fact that Calibrated UAS is exactly a softmax over the
shifted energies \(Q_i/\mu_i^\beta+\kappa_i\). That gives a direct minimum-type
bound on the arrival term, and that bound matches the weighted quadratic drift
structure well enough to close the CTMC theorem under the natural load
condition \(\lambda<\Lambda\).
