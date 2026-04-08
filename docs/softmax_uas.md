# Unified Archimedean Softmax (UAS)

This note is a proof-facing derivation for the UAS policy used in the GibbsQ manuscript. It records the exact routing formula, its weighted variational interpretation, the Archimedean drift identity, and the Foster-Lyapunov closure that yields positive Harris recurrence.

## 1. Definition

Consider a system of \(N\) servers with service rates \(\mu_i > 0\) and queue lengths \(Q_i \ge 0\). For fixed \(\alpha > 0\), define the UAS potential

\[
\Phi_i(Q,\mu)=\frac{Q_i+1}{\mu_i}-\frac{1}{\alpha}\log \mu_i.
\]

The UAS routing law is

\[
p_i(Q,\mu)=
\frac{e^{-\alpha \Phi_i(Q,\mu)}}
{\sum_{j=1}^N e^{-\alpha \Phi_j(Q,\mu)}}
=
\frac{\mu_i \exp\!\left(-\alpha \frac{Q_i+1}{\mu_i}\right)}
{\sum_{j=1}^N \mu_j \exp\!\left(-\alpha \frac{Q_j+1}{\mu_j}\right)}.
\]

This is a smooth, capacity-aware, entropy-regularized routing policy for heterogeneous servers.

Because each \(\mu_i > 0\), the map
\[
(Q_i,\mu_i)\mapsto \mu_i \exp\!\left(-\alpha \frac{Q_i+1}{\mu_i}\right)
\]
is \(C^\infty\) on the interior of the positive-service domain, so the normalized routing probabilities are smooth there as well.

## 2. Variational Form

Let
\[
\Lambda=\sum_{i=1}^N \mu_i,
\qquad
r_i=\frac{\mu_i}{\Lambda},
\qquad
x_i(Q,\mu)=\frac{Q_i+1}{\mu_i}.
\]

Define the energy term

\[
E_i^{\mathrm{UAS}}(Q)=x_i(Q,\mu)-\frac{1}{\alpha}\log \mu_i
\]

and the equivalent prior-weighted entropy-regularized variational objective over the simplex \(\Delta_{N-1}\):

\[
\mathcal G_{\mathrm{UAS}}(p)
=
\sum_{i=1}^N p_i x_i(Q,\mu)
+
\frac{1}{\alpha}\mathrm{KL}(p\|r)
-\frac{1}{\alpha}\log\Lambda.
\]

**Proposition (UAS weighted variational form).**
The unique minimizer of \(\mathcal G_{\mathrm{UAS}}(p)\) over \(\Delta_{N-1}\) is

\[
p_i(Q,\mu)=
\frac{\mu_i \exp\!\left(-\alpha \frac{Q_i+1}{\mu_i}\right)}
{\sum_{j=1}^N \mu_j \exp\!\left(-\alpha \frac{Q_j+1}{\mu_j}\right)}.
\]

At the minimizer,
\[
\sum_i p_i x_i(Q,\mu)+\frac{1}{\alpha}\mathrm{KL}(p\|r)
=
-\frac{1}{\alpha}\log\sum_i r_i e^{-\alpha x_i(Q,\mu)}.
\]

So UAS is a genuine entropy-regularized routing law, but with a heterogeneous-server prior \(r_i=\mu_i/\Lambda\) that differs from the raw-softmax baseline.

## 3. Structural Interpretation

The UAS law combines three effects in one score:

- queue pressure through \(Q_i\)
- service heterogeneity through \(\mu_i\)
- entropy regularization through the softmax form

At low load it favors faster servers. Under congestion, the queue-dependent term shifts mass away from heavily loaded servers. The transition is smooth because routing remains exponential rather than threshold-based.

## 4. Empty-State Behavior

If all queues are empty, then

\[
p_i \propto \mu_i e^{-\alpha/\mu_i}.
\]

Thus UAS favors faster servers at the idle state, but it is not exactly proportional routing unless \(\alpha\) is small or the service rates are close.

## 5. Pairwise Crossover Law

For two servers \(f\) and \(s\), the equality \(p_f = p_s\) is equivalent to

\[
\log \mu_f-\alpha\frac{Q_f+1}{\mu_f}
=
\log \mu_s-\alpha\frac{Q_s+1}{\mu_s}.
\]

Rearranging gives the exact crossover boundary

\[
\frac{Q_f+1}{\mu_f}-\frac{Q_s+1}{\mu_s}
=
\frac{\log \mu_f-\log \mu_s}{\alpha}.
\]

Equivalently,

\[
\frac{Q_f}{\mu_f}-\frac{Q_s}{\mu_s}
=
\frac{\log \mu_f-\log \mu_s}{\alpha}
+
\left(\frac{1}{\mu_s}-\frac{1}{\mu_f}\right).
\]

This identifies the precise queueing balance at which routing mass transfers between faster and slower servers.

## 6. Lyapunov Candidate And Drift Structure

For the heterogeneous theorem path, use the Archimedean Lyapunov candidate

\[
V(Q)=\frac12 \sum_{i=1}^N \frac{Q_i^2}{\mu_i}.
\]

The generator drift for UAS routing takes the form

\[
\mathcal{L}V(Q)
=
\lambda \sum_i p_i(Q,\mu)\frac{Q_i + 1/2}{\mu_i}
-
\sum_i Q_i
+
\frac12 \sum_i \mathbf{1}_{Q_i > 0}.
\]

This identity is the starting point of the UAS recurrence proof.

### Lemma (Archimedean drift identity)

For UAS routing

\[
p_i(Q,\mu)=
\frac{\mu_i \exp\!\left(-\alpha \frac{Q_i+1}{\mu_i}\right)}
{\sum_{j=1}^N \mu_j \exp\!\left(-\alpha \frac{Q_j+1}{\mu_j}\right)},
\]

the generator action on

\[
V(Q)=\frac12 \sum_{i=1}^N \frac{Q_i^2}{\mu_i}
\]

is exactly

\[
\mathcal{L}V(Q)
=
\lambda \sum_i p_i(Q,\mu)\frac{Q_i + 1/2}{\mu_i}
-
\sum_i Q_i
+
\frac12 \sum_i \mathbf{1}_{Q_i > 0}.
\]

### Lemma (Arrival-term bound)

Assume

\[
\Lambda := \sum_{i=1}^N \mu_i > \lambda.
\]

Define

\[
\sum_i p_i(Q,\mu)\frac{Q_i + 1/2}{\mu_i}
\le
\frac{|Q|_1+N}{\Lambda}.
\]

**Proof.**
From the weighted variational identity,

\[
\sum_i p_i x_i(Q,\mu)
+
\frac{1}{\alpha}\mathrm{KL}(p\|r)
=
-\frac{1}{\alpha}\log\sum_i r_i e^{-\alpha x_i(Q,\mu)}.
\]

Since \(\mathrm{KL}(p\|r)\ge 0\),
\[
\sum_i p_i x_i(Q,\mu)
\le
-\frac{1}{\alpha}\log\sum_i r_i e^{-\alpha x_i(Q,\mu)}.
\]

By Jensen's inequality applied to the convex map \(u\mapsto e^{-\alpha u}\),

\[
\sum_i r_i e^{-\alpha x_i(Q,\mu)}
\ge
\exp\left(-\alpha\sum_i r_i x_i(Q,\mu)\right).
\]

Therefore

\[
-\frac{1}{\alpha}\log\sum_i r_i e^{-\alpha x_i(Q,\mu)}
\le
\sum_i r_i x_i(Q,\mu)
=
\frac{1}{\Lambda}\sum_i \mu_i\frac{Q_i+1}{\mu_i}
=
\frac{|Q|_1+N}{\Lambda}.
\]

Hence

\[
\sum_i p_i(Q,\mu)\frac{Q_i+1}{\mu_i}\le \frac{|Q|_1+N}{\Lambda}.
\]

Finally,

\[
\sum_i p_i(Q,\mu)\frac{Q_i + 1/2}{\mu_i}
=
\sum_i p_i(Q,\mu)\frac{Q_i+1}{\mu_i}
-
\frac12\sum_i \frac{p_i(Q,\mu)}{\mu_i}
\le
\sum_i p_i(Q,\mu)\frac{Q_i+1}{\mu_i},
\]

which proves the stated bound.

### Theorem (Foster-Lyapunov closure for UAS)

Assume

\[
\Lambda := \sum_{i=1}^N \mu_i > \lambda.
\]

Define

\[
\epsilon = \frac{\Lambda - \lambda}{\Lambda}
\]

and

\[
R = \frac{\lambda N}{\Lambda} + \frac{N}{2}.
\]

Then the drift identity implies

\[
\mathcal{L}V(Q) \le -\epsilon |Q|_1 + R.
\]

Consequently there exists a compact set

\[
C = \left\{Q \in \mathbb{Z}_+^N : |Q|_1 \le \frac{R+1}{\epsilon}\right\}
\]

such that for all \(Q \notin C\),

\[
\mathcal{L}V(Q) \le -1.
\]

At that point, the continuous-time Foster-Lyapunov criterion yields positive Harris recurrence.

## 7. Theorem Position In The Paper

The manuscript should not present UAS as a corollary of raw softmax. Instead, it should present UAS as:

- a separate variational policy derived from \(\mathcal G_{\mathrm{UAS}}\)
- a separate Lyapunov argument using the weighted potential \(V(Q)\)
- a separate recurrence theorem proved through the weighted-KL/Jensen arrival bound above

That keeps the paper mathematically honest and fully aligned with the heterogeneous routing implementation used in the repo.

## 8. Paper Usage

This note supports the manuscript in five ways:

- it provides the exact UAS routing formula
- it gives the entropy-regularized variational interpretation
- it identifies the Lyapunov candidate for the heterogeneous theorem path
- it gives the exact arrival-term inequality used to close the theorem
- it records the final recurrence constants and compact-set statement

## 9. Standard Reference Points

The standard definitions used around this note are consistent with the usual literature:

- the entropy-regularized simplex minimizer and log-sum-exp variational identity from convex duality and Gibbs variational arguments
- the continuous-time Foster-Lyapunov route to positive Harris recurrence in the Meyn-Tweedie framework

These references justify the general proof template. The UAS closure inequality itself is derived directly above from the weighted-KL identity and Jensen's inequality.
