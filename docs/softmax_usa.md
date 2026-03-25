Here is the **corrected final version**, written in a more rigorous and paper-ready form.

---

## Unified Archimedean Softmax (UAS)

Consider a system of (N) servers with service rates (\mu_i>0) and current queue lengths (Q_i\ge 0). For a fixed parameter (\alpha>0), define the **look-ahead potential**
[
\Phi_i(Q,\mu)=\frac{Q_i+1}{\mu_i}-\frac{\log \mu_i}{\alpha}.
]

The routing probability is
[
p_i(Q,\mu)
==========

# \frac{e^{-\alpha \Phi_i(Q,\mu)}}{\sum_{j=1}^N e^{-\alpha \Phi_j(Q,\mu)}}

\frac{\mu_i \exp!\left(-\alpha \frac{Q_i+1}{\mu_i}\right)}
{\sum_{j=1}^N \mu_j \exp!\left(-\alpha \frac{Q_j+1}{\mu_j}\right)}.
]

This is a smooth, capacity-weighted softmax policy that combines queue pressure and server speed in a single exponential score.

---

## Step 1: Smoothness

Because (\mu_i>0) and the map ((Q_i,\mu_i)\mapsto \mu_i e^{-\alpha(Q_i+1)/\mu_i}) is smooth for (\mu_i>0), the routing probabilities (p_i) are (C^\infty) in the interior of the domain. So the policy is fully smooth, not piecewise or discontinuous.

---

## Step 2: Empty-queue behavior

If all queues are empty, (Q_i=0), then
[
p_i \propto \mu_i e^{-\alpha/\mu_i}.
]
So the policy **favors faster servers at idle state**, but it is **not exactly proportional to (\mu_i)** unless (\alpha) is very small or the service rates are similar.

That is the correct statement. The earlier claim of exact proportional routing at (Q=0) was too strong.

---

## Step 3: Pairwise crossover law

For two servers (f) and (s), the condition (p_f=p_s) is equivalent to
[
\log \mu_f-\alpha\frac{Q_f+1}{\mu_f}
====================================

\log \mu_s-\alpha\frac{Q_s+1}{\mu_s}.
]

Rearranging gives the exact crossover boundary:
[
\frac{Q_f+1}{\mu_f}-\frac{Q_s+1}{\mu_s}
=======================================

\frac{\log \mu_f-\log \mu_s}{\alpha}.
]

Equivalently,
[
\frac{Q_f}{\mu_f}-\frac{Q_s}{\mu_s}
===================================

\frac{\log \mu_f-\log \mu_s}{\alpha}
+\left(\frac{1}{\mu_s}-\frac{1}{\mu_f}\right).
]

This is the corrected form. The earlier version omitted the (+1)-induced term.

---

## Step 4: Interpretation

The policy works as follows:

* when queues are small, the (\mu_i) factor and the ((Q_i+1)/\mu_i) term strongly prefer faster servers;
* when a server becomes congested, its queue penalty grows and routing shifts away from it;
* the transition between servers is smooth, because the routing law is exponential rather than threshold-based.

So the policy is mathematically coherent as a **soft, capacity-aware, congestion-sensitive routing rule**.

---

## Step 5: Stability statement

A stability claim should be stated carefully.

Let
[
V(Q)=\frac12 \sum_{i=1}^N \frac{Q_i^2}{\mu_i}.
]

To prove stability rigorously, one must show a Foster–Lyapunov drift bound of the form
[
\mathcal{L}V(Q)\le -\varepsilon |Q| + B
]
for some (\varepsilon>0) and finite constant (B), outside a compact set, under the assumed arrival and service model.

If such a drift inequality is established, then standard Foster–Lyapunov theory implies positive recurrence and hence stability.

So the correct statement is:

> **UAS is stable under the usual queueing assumptions provided the corresponding Foster–Lyapunov drift condition is verified.**

That is rigorous. The earlier phrase “Absolutely Stable” was too strong unless the full drift proof is actually supplied.

---

## Final verdict

The corrected final mathematical version is:

[
p_i(Q,\mu)
==========

\frac{\mu_i \exp!\left(-\alpha \frac{Q_i+1}{\mu_i}\right)}
{\sum_{j=1}^N \mu_j \exp!\left(-\alpha \frac{Q_j+1}{\mu_j}\right)}.
]

It is:

* smooth,
* capacity-aware,
* congestion-sensitive,
* correctly favoring faster servers at low load,
* and mathematically consistent.

The only part that still requires a separate proof is the **full stability theorem**, which depends on the exact queueing assumptions and the drift calculation.

If you want, I can now turn this into a **formal theorem–lemma–proof section** in paper style.
