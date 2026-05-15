# Review Of `suggestions.md`

This note records which parts of
[suggestions.md](./suggestions.md) are mathematically correct and which parts
should **not** be imported into the formal package.

## 1. Correct Parts Of The Review

The following points in `suggestions.md` are correct.

### A. The note [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md) is correct

The review is right that:

- the exact lattice increment formulas for \(H(Q\pm e_i)-H(Q)\) are correct
- the exact generator identity is correct
- the Taylor decomposition with bounded remainder is legitimate
- the boundary mismatch between the reflected ODE and the CTMC generator is real

Those statements are already consistent with the current `z2` files and should
be retained.

### B. The potential \(H\) does not obviously give the standard linear drift bound

The review is also right that the negative square term in
[08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md) is uniformly
bounded in magnitude. The correct bound is obtained from

\[
0\le \lambda p_i(Q)\le \lambda
\quad\Longrightarrow\quad
\frac{\mu_i-\lambda}{\mu_i^\beta}
\le
\partial_i H(Q)
\le
\mu_i^{1-\beta}.
\]

Hence

\[
0\le \mu_i^\beta \bigl(\partial_i H(Q)\bigr)^2
\le
\mu_i^\beta
\max\!\left\{
\left(\frac{\mu_i-\lambda}{\mu_i^\beta}\right)^2,
\left(\mu_i^{1-\beta}\right)^2
\right\}.
\]

So the `H`-based generator decomposition does **not** directly produce a drift
of the form

\[
\mathcal L H(Q)\le -\varepsilon |Q|_1 + C.
\]

That observation is correct and should be treated as established.

## 2. What The Review Overstates

Two stronger claims in `suggestions.md` should **not** be adopted as formal
results.

### A. "A direct Foster-Lyapunov proof with \(H\) alone cannot work"

That statement is too strong.

What has been shown is narrower:

- the current decomposition of \(\mathcal L H\) does not yield the standard
  **linear** drift bound
- the positive boundary term is a genuine obstruction to a naive lift of the
  deterministic argument

But this does **not** prove that no Foster-Lyapunov argument using \(H\) is
possible. A weaker negative drift condition outside a finite set could still
exist in principle. We simply do not have such a proof.

So the correct conclusion is:

**Supported conclusion.**
The current `H`-based calculation does not close a direct CTMC stability proof.

not

**Unsupported conclusion.**
No direct CTMC Foster-Lyapunov proof using \(H\) can exist.

### B. The proposed temperature-scaling bridge is not a valid proof as written

The second half of `suggestions.md` proposes to define a family of systems with

\[
\lambda^{(r)}=r\lambda_0,
\qquad
\mu_i^{(r)}=r\mu_{i,0},
\qquad
\alpha_r=\alpha_0/r,
\]

and then claims that a Dai-style fluid-limit theorem would certify the original
benchmark by taking \(r=1\).

That step is not valid as written, for two separate reasons.

#### Reason 1. It changes the stochastic model

The CTMC whose stability we want is the fixed benchmark model with fixed
parameters \((\lambda,\mu,\alpha,\beta,\gamma,c)\).

The proposed construction replaces it with a **different family of CTMCs** whose
arrival rates, service rates, and temperature all vary with \(r\). That may be
a mathematically interesting family, but it is not the same object as the
original fixed-parameter CTMC.

#### Reason 2. The fluid-limit theorem does not work by "prove a family, then set \(r=1\)"

Classical fluid-limit theorems study one fixed Markov model and the asymptotic
behavior of trajectories started from states of size \(r\to\infty\). They do not
automatically imply:

- define a different model for each \(r\),
- analyze the limit \(r\to\infty\),
- then conclude stability of the \(r=1\) model.

That inference requires a separate theorem uniform in \(r\), and no such theorem
has been established here.

So the proposed "temperature scaling closes the original stochastic gap" claim
must be rejected in the current formal package.

## 3. Correct Updated Position

After reviewing `suggestions.md`, the correct project position is:

1. The deterministic reflected-ODE theory in `z2` is sound.
2. The CTMC generator obstruction identified in
   [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md) is sound.
3. The suggested temperature-scaling argument does **not** yet certify the
   original benchmark CTMC.
4. The old scaling-based shortcut does **not** yet certify the original
   fixed-parameter CTMC.
5. Any later direct certification route must be judged on its own mathematics
   and not by importing unsupported claims from `suggestions.md`.

## 4. Practical Next Step

The mathematically honest next direction is one of the following:

- strengthen the direct generator argument for the fixed CTMC
- design a different lattice Lyapunov function
- formulate a new theorem for a genuinely rescaled policy family, while keeping
  it clearly separate from the original benchmark theorem

Those are legitimate next steps. The current review does not collapse them into
a finished proof.
