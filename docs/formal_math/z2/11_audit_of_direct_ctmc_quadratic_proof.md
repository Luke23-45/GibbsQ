# Audit Of The Direct CTMC Quadratic Proof Attempt

This note audits
[10_direct_ctmc_quadratic_proof_attempt.md](./10_direct_ctmc_quadratic_proof_attempt.md)
step by step.

## 1. What Must Be Checked

The proof attempt stands or falls on four points:

1. the CTMC generator formula for the weighted quadratic \(V\),
2. the exact softmax representation with shifted energies \(s_i(Q)\),
3. the minimum bound on \(\sum_i p_i(Q)Q_i/\mu_i^\beta\),
4. the conversion of the resulting drift into a linear \(-\varepsilon |Q|_1\)
   bound.

## 2. Generator Identity

For

\[
V(Q)=\frac12\sum_i \frac{Q_i^2}{\mu_i^\beta},
\]

the one-step increments are exactly

\[
V(Q+e_i)-V(Q)=\frac{Q_i+1/2}{\mu_i^\beta},
\]

\[
V(Q-e_i)-V(Q)=-\frac{Q_i-1/2}{\mu_i^\beta}
\qquad (Q_i>0).
\]

This part is exact and routine.

## 3. Softmax Representation

The routing rule can be rewritten as

\[
p_i(Q)
\propto
\exp\!\left(
-\alpha\left[
\frac{Q_i}{\mu_i^\beta}
+
\frac{c}{\mu_i^\beta}
-\frac{\gamma}{\alpha}\log \mu_i
\right]
\right).
\]

So the shifted energies

\[
s_i(Q)=\frac{Q_i}{\mu_i^\beta}+\kappa_i,
\qquad
\kappa_i=\frac{c}{\mu_i^\beta}-\frac{\gamma}{\alpha}\log \mu_i
\]

are correct.

## 4. Minimum Bound

The entropy variational identity for softmax yields

\[
\sum_i p_i(Q)s_i(Q)\le \min_i s_i(Q)+\frac{\log N}{\alpha}.
\]

Subtracting \(\sum_i p_i(Q)\kappa_i\) and using

\[
\sum_i p_i(Q)\kappa_i \ge \min_i \kappa_i
\]

gives

\[
\sum_i p_i(Q)\frac{Q_i}{\mu_i^\beta}
\le
\min_i s_i(Q)+\frac{\log N}{\alpha}-\min_i\kappa_i.
\]

Also

\[
\min_i s_i(Q)\le \min_i \frac{Q_i}{\mu_i^\beta}+\max_i \kappa_i.
\]

Combining the two inequalities yields the claimed bound

\[
\sum_i p_i(Q)\frac{Q_i}{\mu_i^\beta}
\le
m(Q)+C_1.
\]

This step is correct.

## 5. Drift Closure

The decomposition

\[
Q_i=\mu_i^\beta m(Q)+\Delta_i(Q),
\qquad \Delta_i(Q)\ge 0,
\]

is valid by definition of \(m(Q)=\min_i Q_i/\mu_i^\beta\). It gives

\[
\sum_i \mu_i^{1-\beta}Q_i
=
\Lambda m(Q)+\sum_i \mu_i^{1-\beta}\Delta_i(Q).
\]

The norm comparison

\[
|Q|_1=\left(\sum_i \mu_i^\beta\right)m(Q)+\sum_i \Delta_i(Q)
\]

is also exact. Hence choosing

\[
\varepsilon
=
\min\!\left\{
\frac{\Lambda-\lambda}{\sum_i \mu_i^\beta},
\min_i \mu_i^{1-\beta}
\right\}
\]

does indeed imply

\[
\varepsilon |Q|_1
\le
(\Lambda-\lambda)m(Q)+\sum_i \mu_i^{1-\beta}\Delta_i(Q).
\]

So the final linear drift bound is algebraically correct.

## 6. Relation To The Old SCUAS Failure

The earlier SCUAS theorem used a prior-weighted Jensen route that transformed
the arrival term into a coefficientwise condition. That was sufficient but too
coarse at the benchmark.

The new proof does something different:

- it rewrites Calibrated UAS as an ordinary softmax over the shifted energies
  \(Q_i/\mu_i^\beta+\kappa_i\),
- it uses a direct minimum-type softmax bound,
- it matches that bound with the weighted quadratic decomposition
  \(Q_i=\mu_i^\beta m+\Delta_i\).

So this is not a rephrasing of the failed SCUAS argument. It is a different
proof structure.

## 7. Current Audit Conclusion

I do not see an algebraic gap in
[10_direct_ctmc_quadratic_proof_attempt.md](./10_direct_ctmc_quadratic_proof_attempt.md).

If this audit survives the level of scrutiny required for theorem promotion,
then the fixed-parameter Calibrated-UAS CTMC would be certified under the
natural load condition \(\lambda<\sum_i \mu_i\).

Until that promotion step is explicitly recorded, the correct package-level
reading is narrower:

- file `10` is an audited stochastic certification route,
- the deterministic theorem core remains the only unambiguously promoted theorem
  layer,
- and the project should avoid claiming that internal code validation alone has
  settled the stochastic theorem.
