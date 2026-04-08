# Final Review: GibbsQ Paper Alignment

## Repo Truth

The repository already implements three distinct layers and should describe them that way in all public materials.

**1. GibbsQ as the framework**

`GibbsQ` is best understood as the umbrella framework for entropy-regularized queue-routing policies in parallel-server systems. It is not just one routing formula and it is not the name of a literal thermodynamic model.

**2. Raw softmax as the baseline theorem policy**

The raw softmax policy

\[
p_i(Q)=\frac{\exp(-\alpha Q_i)}{\sum_{j=1}^N \exp(-\alpha Q_j)}
\]

is the simplest theorem-backed routing baseline in the repo. It is the clean baseline for the Foster-Lyapunov proof and for the variational softmin argument.

**3. UAS as the heterogeneous theorem-backed extension**

The repository operationalizes UAS in policy definitions, pretraining, empirical comparisons, and drift verification. The intended UAS routing law is

\[
p_i(Q,\mu)=
\frac{\mu_i \exp\!\left(-\alpha \frac{Q_i+1}{\mu_i}\right)}
{\sum_{j=1}^N \mu_j \exp\!\left(-\alpha \frac{Q_j+1}{\mu_j}\right)}.
\]

This policy is not a corollary of raw softmax. It must be presented with its own variational derivation and its own Lyapunov-based stability argument.

**4. N-GibbsQ as the learned policy**

`N-GibbsQ` is the neural policy trained through behavior cloning and REINFORCE against theorem-backed routing baselines. It should remain clearly distinguished from the analytical routing theorems.

## Theorem Truth

The manuscript should present two separate theorem layers.

**1. Raw softmax theorem**

The existing raw-softmax proof is already close to final form:

- CTMC regularity and non-explosion
- irreducibility under the queueing assumptions
- quadratic Lyapunov function
- entropy-regularized variational inequality for softmax
- Foster-Lyapunov conclusion yielding positive Harris recurrence

**2. UAS theorem**

UAS must be stated separately, not as an automatic extension of the raw-softmax theorem.

The paper should explicitly include:

- the UAS potential
  \[
  \Phi_i(Q,\mu)=\frac{Q_i+1}{\mu_i}-\frac{1}{\alpha}\log \mu_i
  \]
- the entropy-regularized variational objective whose minimizer yields the UAS formula
- the UAS routing proposition
- the Archimedean Lyapunov candidate
  \[
  V(Q)=\frac12 \sum_{i=1}^N \frac{Q_i^2}{\mu_i}
  \]
- the generator drift expression and recurrence constants
- the final positive-recurrence theorem under the stated load condition

With that structure, both raw softmax and UAS are theorem-backed, but through separate arguments: raw softmax through the standard entropy bound and UAS through the prior-weighted KL/Jensen closure.

## Language Corrections

The repo should use one precise vocabulary everywhere public.

Replace these phrases:

- `Gibbs free energy` when used as a literal physical claim
- `absolute stability`
- `thermal soft-damping`
- `fundamental capacity condition`

Use these instead:

- `entropy-regularized variational objective`
- `Gibbs variational principle`
- `positive Harris recurrence`
- `Foster-Lyapunov stability`
- `load condition \lambda < \sum_i \mu_i`

The word `Gibbs` is still appropriate in the project name and in variational language, but it should be used in the mathematical softmax/entropy sense, not as an unqualified thermodynamic statement.

Performance language should also be separated from theorem language. Terms such as `superior` or `certified` belong in empirical summaries only if they are tied to reported results, not inside theorem exposition.

## Publication Recommendation

The final paper should be submitted with the following architecture:

**1. Framework level**

Present `GibbsQ` as the general entropy-regularized routing framework.

**2. Analytical level**

Present two theorem-backed routing laws:

- raw softmax as the simplest baseline theorem
- UAS as the heterogeneous theorem-backed extension with its own derivation, weighted drift identity, and prior-weighted variational closure

**3. Learning level**

Present `N-GibbsQ` as the learned neural policy trained and benchmarked against those theorem-backed routing standards.

This resolves the earlier inconsistency in the repo:

- older review language treated UAS as not yet automatically proved
- other docs and code already treated UAS as theorem-facing

The final paper resolves that contradiction by making the UAS theorem explicit rather than implied.

## Final Recommendation

Proceed with a two-theorem manuscript:

- first theorem: raw softmax positive Harris recurrence
- second proposition/theorem chain: UAS weighted variational derivation, weighted drift identity, and positive recurrence via the prior-weighted Jensen bound

Then keep all neural results clearly framed as empirical learning against theorem-backed routing baselines.
