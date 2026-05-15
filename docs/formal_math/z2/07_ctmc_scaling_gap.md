# The Remaining CTMC Scaling Gap

This note explains the original stochastic gap that remained after the
deterministic convergence result in
[06_global_convergence_reflected_ode.md](./06_global_convergence_reflected_ode.md).

Its role in the current package is specific:

- it records why the old smooth-ODE-to-CTMC shortcut was invalid,
- it does **not** override the later direct CTMC route in files `10` and `11`,
- it should be read as a correction note, not as the final word on every
  possible stochastic route.

## 1. What Has Been Fully Proved

The `z2` package now proves:

- the correct reflected deterministic ODE on the orthant
- exact characterization of its unique boundary equilibrium
- convexity and coercivity of the associated potential
- global convergence of every reflected-ODE trajectory to that equilibrium

That is a complete deterministic theorem.

## 2. What A Classical Fluid-Limit Theorem Would Need

The Dai-style queueing result is stronger. It would require that the same
deterministic ODE arise as the limit of the stochastic queue-length process
under the usual fluid scaling

\[
\bar Q^{(r)}(t):=\frac{1}{r}Q^{(r)}(rt).
\]

For many queueing networks, the scaled drift depends on a routing or service map
that remains nontrivial under that scaling, and then one can study the limit ODE
and lift its stability back to the CTMC.

## 3. Why Reflected UAS Is Delicate

For Reflected UAS, the routing probabilities are

\[
p_i(q)
=
\frac{\mu_i^\gamma \exp\!\left(-\alpha (q_i+c)/\mu_i^\beta\right)}
{\sum_{j=1}^N \mu_j^\gamma \exp\!\left(-\alpha (q_j+c)/\mu_j^\beta\right)}.
\]

If one inserts a fluid-scale state \(rq\), then

\[
p_i(rq)
=
\frac{\mu_i^\gamma \exp\!\left(-\alpha (rq_i+c)/\mu_i^\beta\right)}
{\sum_{j=1}^N \mu_j^\gamma \exp\!\left(-\alpha (rq_j+c)/\mu_j^\beta\right)}.
\]

With fixed \(\alpha>0\), the factor \(rq_i\) appears inside the exponent. As
\(r\to\infty\), this heavily amplifies queue differences. The limiting routing
map therefore tends toward a hard argmin rule on the scaled coordinates rather
than staying equal to the smooth softmax map analyzed in `z2`.

So the reflected ODE studied here is not yet identified as the automatic
classical fluid limit of the CTMC under standard queue-length scaling.

## 4. What This Means Mathematically

At the moment, the project supports the following clean statement:

**Proved statement.**
The deterministic reflected ODE built from the smooth reflected-UAS routing map
is globally asymptotically stable.

But the following stronger statement is not established by this
classical-fluid-limit route:

**Separate stochastic statement.**
The original Reflected-UAS CTMC is positive Harris recurrent for every
\(\lambda<\Lambda\).

The gap between those statements is not cosmetic. It is a genuine scaling issue
for this route. Later files `10` and `11` develop a separate direct
Foster-Lyapunov route, whose status is controlled by the package-level
promotion rules.

## 5. Plausible Ways To Close The Gap

There are at least three legitimate research paths from here.

### Path A. Rescaled-parameter fluid theory

One can study a family of policies with \(\alpha=\alpha_r\) shrinking with the
fluid scale, so that the softmax remains nondegenerate in the limit. That would
change the asymptotic regime and would require a new theorem statement.

This path must be kept separate from the original fixed-parameter CTMC. Simply
proving stability of a rescaled family and then "setting \(r=1\)" does not by
itself certify the benchmark model. See
[09_review_of_suggestions.md](./09_review_of_suggestions.md).

### Path B. Direct stochastic Lyapunov theory

One can try to prove CTMC stability directly, without routing through the
classical fluid-limit theorem. That would require a new Foster-Lyapunov
construction that is sharp enough for Reflected UAS.

### Path C. Hybrid comparison theory

One can compare the CTMC to a simpler stable routing rule or to a deterministic
projected flow in a way that survives the fixed-\(\alpha\) scaling. This would
be more specialized than the standard Dai route, but it could still be a valid
paper.

## 6. Project-Level Conclusion

The deterministic `z2` package is now mathematically coherent and complete on
its own terms.

This note closes only one issue: the classical fixed-\(\alpha\) fluid-scaling
shortcut is not enough on its own.

It therefore supports the following safe conclusion:

- global stability of the reflected ODE does **not** by itself prove stochastic
  stability of the original queueing process.

What comes next may be either:

- a scaling-based theorem different from the rejected shortcut, or
- a direct stochastic Lyapunov theorem such as the later route developed in
  files `10` and `11`.
