# Hidden Convex Structure Of The Reflected ODE

This note shows that the reflected Reflected-UAS ODE is a projected gradient
system for an explicit convex potential. That observation is the key step that
turns the equilibrium formula into a genuine deterministic convergence theorem.

## 1. Weighted Geometry

Define the diagonal matrix

\[
D := \operatorname{diag}(\mu_1^\beta,\dots,\mu_N^\beta).
\]

Because every \(\mu_i^\beta>0\), this matrix is positive definite. It induces
the weighted inner product

\[
\langle u,v\rangle_D := u^\top D^{-1} v
\]

and norm

\[
\|u\|_D := \sqrt{u^\top D^{-1}u}.
\]

## 2. The Potential Function

Let

\[
W(q)=\sum_{j=1}^N \mu_j^\gamma
\exp\!\left(-\alpha (q_j+c)/\mu_j^\beta\right),
\]

and define

\[
H(q)
:=
\sum_{i=1}^N \mu_i^{1-\beta} q_i
+
\frac{\lambda}{\alpha}\log W(q),
\qquad q\in\mathbb R_+^N.
\]

Since \(W(q)>0\) for all \(q\), the function \(H\) is \(C^\infty\) on
\(\mathbb R_+^N\).

## 3. Gradient Identity

**Proposition 3.**
For every \(q\in\mathbb R_+^N\),

\[
\partial_i H(q)
=
\frac{\mu_i-\lambda p_i(q)}{\mu_i^\beta},
\qquad i=1,\dots,N.
\]

Equivalently,

\[
\lambda p(q)-\mu = -D\nabla H(q).
\]

**Proof.**
Differentiate \(W\):

\[
\partial_i W(q)
=
-\frac{\alpha}{\mu_i^\beta}\,
\mu_i^\gamma \exp\!\left(-\alpha (q_i+c)/\mu_i^\beta\right)
=
-\frac{\alpha}{\mu_i^\beta} w_i(q).
\]

Hence

\[
\partial_i \log W(q)
=
\frac{\partial_i W(q)}{W(q)}
=
-\frac{\alpha}{\mu_i^\beta}\frac{w_i(q)}{W(q)}
=
-\frac{\alpha}{\mu_i^\beta}p_i(q).
\]

Therefore

\[
\partial_i H(q)
=
\mu_i^{1-\beta}
+
\frac{\lambda}{\alpha}\partial_i \log W(q)
=
\mu_i^{1-\beta}
-\frac{\lambda p_i(q)}{\mu_i^\beta}
=
\frac{\mu_i-\lambda p_i(q)}{\mu_i^\beta}.
\]

Multiplying by \(D\) gives the vector identity. `QED`

So the unconstrained drift is exactly a negative weighted gradient.

## 4. Hessian And Convexity

**Proposition 4.**
The Hessian of \(H\) is

\[
\partial_{ij}^2 H(q)
=
\lambda\alpha
\left[
\frac{\mathbf 1_{\{i=j\}}\,p_i(q)}{\mu_i^{2\beta}}
-\frac{p_i(q)p_j(q)}{\mu_i^\beta\mu_j^\beta}
\right].
\]

In particular, \(H\) is convex on \(\mathbb R_+^N\).

**Proof.**
The softmax derivative is

\[
\partial_j p_i(q)
=
-\alpha p_i(q)
\left[
\frac{\mathbf 1_{\{i=j\}}}{\mu_i^\beta}
-\frac{p_j(q)}{\mu_j^\beta}
\right].
\]

Differentiating the formula from Proposition 3 gives the Hessian expression.

For any \(v\in\mathbb R^N\),

\[
v^\top \nabla^2 H(q)\,v
=
\lambda\alpha
\left[
\sum_{i=1}^N p_i(q)\left(\frac{v_i}{\mu_i^\beta}\right)^2
-\left(\sum_{i=1}^N p_i(q)\frac{v_i}{\mu_i^\beta}\right)^2
\right].
\]

The bracket is the variance of \(v_i/\mu_i^\beta\) under the probability vector
\(p(q)\), so it is nonnegative. Hence \(\nabla^2 H(q)\) is positive
semidefinite for every \(q\), and \(H\) is convex. `QED`

## 5. Coercivity

**Proposition 5.**
If \(\lambda<\Lambda:=\sum_{i=1}^N\mu_i\), then \(H(q)\to+\infty\) as
\(\|q\|_2\to\infty\) with \(q\in\mathbb R_+^N\).

**Proof.**
Let

\[
s_i := \frac{q_i}{\mu_i^\beta},
\qquad
a_i := \mu_i^\gamma e^{-\alpha c/\mu_i^\beta}.
\]

Then

\[
H(q)=\sum_{i=1}^N \mu_i s_i
+\frac{\lambda}{\alpha}\log\!\left(\sum_{i=1}^N a_i e^{-\alpha s_i}\right).
\]

Choose the fixed probability vector

\[
\rho_i:=\frac{\mu_i}{\Lambda},
\qquad
\Lambda:=\sum_{i=1}^N\mu_i.
\]

For any real numbers \(x_i\), \(\max_i x_i\ge \sum_i\rho_i x_i\), and
\(\log\sum_i e^{x_i}\ge \max_i x_i\). Applying this with
\(x_i=\log a_i-\alpha s_i\) gives

\[
\log\!\left(\sum_{i=1}^N a_i e^{-\alpha s_i}\right)
\ge
\sum_{i=1}^N \rho_i\log a_i
-\alpha\sum_{i=1}^N \rho_i s_i.
\]

Therefore

\[
H(q)
\ge
\sum_{i=1}^N \mu_i s_i
-\lambda\sum_{i=1}^N \rho_i s_i
+\frac{\lambda}{\alpha}\sum_{i=1}^N\rho_i\log a_i.
\]

Since \(\rho_i=\mu_i/\Lambda\), this becomes

\[
H(q)
\ge
\left(1-\frac{\lambda}{\Lambda}\right)
\sum_{i=1}^N \mu_i s_i
+ C,
\]

where

\[
C:=\frac{\lambda}{\alpha}\sum_{i=1}^N\rho_i\log a_i
\]

is finite. Because \(\lambda<\Lambda\) and every \(\mu_i>0\), the coefficient
\(1-\lambda/\Lambda\) is strictly positive, and
\(\sum_i\mu_i s_i=\sum_i\mu_i^{1-\beta}q_i\to\infty\) whenever
\(\|q\|_2\to\infty\) with \(q\in\mathbb R_+^N\). Hence
\(H(q)\to+\infty\). `QED`

## 6. Constrained Minimizer And KKT Conditions

Let \(C:=\mathbb R_+^N\) and define

\[
\Phi(q):=H(q)+I_C(q),
\]

where \(I_C\) is the indicator of the orthant:

\[
I_C(q)=
\begin{cases}
0, & q\in C,\\
+\infty, & q\notin C.
\end{cases}
\]

By Propositions 4 and 5, \(\Phi\) is proper, convex, lower semicontinuous, and
coercive on \(C\). Therefore \(\Phi\) has at least one minimizer.

The first-order optimality condition is

\[
0 \in \nabla H(q)+N_C(q),
\]

where \(N_C(q)\) is the normal cone of the orthant. Coordinatewise, this means

\[
q_i\ge 0,
\qquad
\partial_i H(q)\ge 0 \text{ when } q_i=0,
\qquad
\partial_i H(q)=0 \text{ when } q_i>0.
\]

Using Proposition 3, these are exactly the equilibrium complementarity
conditions from
[01_reflected_fluid_model.md](./01_reflected_fluid_model.md):

\[
q_i\ge 0,
\qquad
\lambda p_i(q)\le \mu_i,
\qquad
q_i(\mu_i-\lambda p_i(q))=0.
\]

Hence the constrained minimizers of \(\Phi\) are precisely the equilibria of
the reflected ODE.

## 7. Uniqueness Of The Minimizer

By [02_boundary_equilibrium.md](./02_boundary_equilibrium.md), the equilibrium
is unique and is given explicitly by the scalar \(K^*\) formula. Therefore:

**Corollary 2.**
The constrained convex program

\[
\min_{q\in\mathbb R_+^N} H(q)
\]

has a unique minimizer \(q^*\), and this minimizer is exactly the boundary
equilibrium characterized in
[02_boundary_equilibrium.md](./02_boundary_equilibrium.md).
