# Thesis Hypotheses For The `z2` Program

This note defines the **defensible thesis hypotheses** for the current
Reflected-UAS project.

It is written to prevent a repeat of the earlier failure mode:

- mixing theorem-level statements with exploratory evidence,
- promoting internal proof routes too early,
- letting weaker empirical side directions dictate the thesis structure.

The rule for this file is simple:

**A hypothesis belongs in the thesis core only if the required evidence is
already in the repo, or if the missing step is named explicitly and narrowly.**

---

## 1. Thesis Orientation

The thesis should be built around the following scientific question:

**Main thesis question.**
What is the correct deterministic stability structure of Reflected UAS, and
how far can that structure be lifted to the original queueing CTMC without
overclaiming?

This is the right question for the current repository because it matches what
the `z2` notes and validation artifacts actually support.

It is **not** correct to frame the thesis around any of the following:

- "we found the best overall routing architecture,"
- "the neural policy is the main contribution,"
- "the full stochastic theory is finished beyond dispute,"
- "every exploratory extension improved the benchmark."

Those are exactly the types of claims that created risk in the earlier
manuscript.

---

## 2. Primary Hypothesis Set

The thesis should use a layered hypothesis structure.

### H1. Deterministic structural hypothesis

**Hypothesis H1.**
For fixed parameters
\(\alpha>0\), \(\beta>0\), \(\gamma\in\mathbb R\), \(c\ge 0\), service rates
\(\mu_i>0\), and load \(\lambda < \Lambda := \sum_i \mu_i\), the correct
deterministic object associated with Reflected UAS is a **reflected ODE** on
\(\mathbb R_+^N\), not the unconstrained drift equation
\(\dot q = \lambda p(q)-\mu\).

**What H1 asserts.**

- the equilibrium must satisfy boundary complementarity conditions,
- the equilibrium is characterized by a scalar consistency equation,
- the reflected ODE admits an explicit convex potential,
- every reflected trajectory converges to the unique boundary equilibrium.

**Current support in the repo.**

- [01_reflected_fluid_model.md](./01_reflected_fluid_model.md)
- [02_boundary_equilibrium.md](./02_boundary_equilibrium.md)
- [05_projected_gradient_structure.md](./05_projected_gradient_structure.md)
- [06_global_convergence_reflected_ode.md](./06_global_convergence_reflected_ode.md)

**Status.**
This is the strongest thesis hypothesis.
It is already theorem-level material.

### H2. Deterministic validation hypothesis

**Hypothesis H2.**
For the benchmark parameter point used in the repo, the exact boundary
equilibrium predicted by the reflected-ODE theory coincides with the stored
deterministic numerical attractor.

**What H2 asserts.**

- the corrected theory is not merely abstract,
- the benchmark attractor is explained by the new equilibrium formula,
- the earlier interior-equilibrium interpretation was wrong.

**Current support in the repo.**

- benchmark check in [02_boundary_equilibrium.md](./02_boundary_equilibrium.md)
- deterministic verification artifact
  [boundary_equilibrium_verification_20260515_091832.csv](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/data/boundary_equilibrium_verification_20260515_091832.csv)

**Status.**
This is a theorem-to-computation consistency claim and is safe for the thesis
core.

### H3. Stochastic obstruction hypothesis

**Hypothesis H3.**
The deterministic Lyapunov structure based on the potential \(H\) does **not**
automatically lift to the queueing CTMC under the earlier stochastic route,
because the CTMC generator contains a genuine boundary mismatch absent from the
reflected ODE.

**What H3 asserts.**

- the old stochastic shortcut was mathematically invalid,
- the gap is structural rather than cosmetic,
- the project's correction is scientifically meaningful.

**Current support in the repo.**

- [07_ctmc_scaling_gap.md](./07_ctmc_scaling_gap.md)
- [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md)
- [09_review_of_suggestions.md](./09_review_of_suggestions.md)

**Status.**
This is also safe for the thesis core.
It is a limitation result, but it is a mathematically valuable one.

### H4. Direct CTMC certification-route hypothesis

**Hypothesis H4.**
A direct Foster-Lyapunov proof for the fixed-parameter Reflected-UAS CTMC can
be built using a weighted quadratic Lyapunov function together with an exact
softmax-minimum bound, and this route certifies the benchmark-default
Reflected-UAS point under the natural load condition \(\lambda < \Lambda\).

**What H4 asserts.**

- there is a viable stochastic theorem route that bypasses the old `H`-based
  obstruction,
- the route is fixed-parameter and benchmark-relevant,
- the benchmark-default policy is not merely empirical if this proof survives
  external review.

**Current support in the repo.**

- proof attempt
  [10_direct_ctmc_quadratic_proof_attempt.md](./10_direct_ctmc_quadratic_proof_attempt.md)
- internal audit
  [11_audit_of_direct_ctmc_quadratic_proof.md](./11_audit_of_direct_ctmc_quadratic_proof.md)
- validation code
  [direct_ctmc_validation.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/gibbsq/experiments/verification/direct_ctmc_validation.py)
- test file
  [test_direct_ctmc_validation.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/tests/test_direct_ctmc_validation.py)
- benchmark audit summary
  [direct_ctmc_validation_summary.md](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/direct_ctmc_validation/direct_ctmc_validation/final_20260515_150335/metadata/direct_ctmc_validation_summary.md)

**Status.**
This is the most important **conditional** hypothesis.

For thesis safety, it must be presented in one of two modes:

- **Mode A: promoted theorem**
  if your advisor signs off on the argument in files `10` and `11`;
- **Mode B: audited stochastic certification route**
  if external theorem-level sign-off is still pending.

The thesis must be written so that it remains defensible under either mode.

### H5. Benchmark empirical hypothesis

**Hypothesis H5.**
Under the anchor benchmark already used in the repo, the benchmark-default
Reflected UAS policy outperforms the baseline UAS and JSSQ policies in mean
steady-state total queue length.

**What H5 asserts.**

- the mathematically motivated closed-form policy is also practically strong,
- the benchmark default remains the main empirical point worth validating,
- the thesis needs only a focused benchmark validation layer, not a sprawling
  architecture search.

**Current support in the repo.**

- quick direct-CTMC rerun summary in
  [direct_ctmc_validation_summary.md](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/direct_ctmc_validation_quick/direct_ctmc_validation/final_20260515_150449/metadata/direct_ctmc_validation_summary.md)

**Status.**
Safe as a validation hypothesis, but secondary to H1-H4.

### H6. Negative-search hypothesis

**Hypothesis H6.**
Within the exploratory directions already tested in this repository, there is
no evidence that SMVR, adaptive-temperature UAS, or other recent probes provide
a stronger thesis direction than Reflected UAS plus the `z2` theorem program.

**What H6 asserts.**

- the project should stop spending its thesis budget on new policy families,
- negative results are part of the evidence base,
- the thesis direction is now mathematically selected, not hype-selected.

**Current support in the repo.**

- historical exploratory-probe notes outside the current `z2` theorem package,
  to be regenerated or attached before H6 is used as a cited thesis claim

**Status.**
This belongs, at most, in a short appendix or decision memo after its supporting
probe artifacts are regenerated or attached. It is not a headline thesis claim.

### H7. Differentiable-policy applicability hypothesis

**Hypothesis H7.**
Reflected UAS is not only competitive with classical routing baselines such as
JSQ and JSSQ, but also defines a **continuous and differentiable** routing map
that is compatible with gradient-based policy learning in a way that those hard
dispatch rules are not.

**What H7 asserts.**

- the softmax-based routing law gives a smooth policy parameterization,
- this smoothness makes behavior cloning and policy-gradient fine-tuning
  technically feasible,
- the project contains limited evidence that this compatibility is real,
- this is a **supporting applicability study**, not the central theorem claim.

**Current support in the repo.**

- the smooth softmax formula stated in this package
- any neural-policy or manuscript support must be restored or attached before
  H7 is cited as an evidence-backed thesis claim

**Status.**
This is safe only as a secondary or tertiary applicability claim, and only when
the supporting learning-study artifacts are restored or attached. It must not
be presented as the "main hero" of the thesis.

---

## 3. Null Hypotheses And Failure Conditions

Each thesis hypothesis must have a clear failure condition.

### For H1-H2

H1-H2 fail if any of the following occur:

- the reflected ODE is not the correct deterministic object,
- the equilibrium characterization is not unique,
- the global convergence proof contains a gap,
- the benchmark attractor does not match the closed-form equilibrium.

At the moment, the repo evidence does not indicate any of these failures.

### For H3

H3 fails if the `H`-based stochastic obstruction in files `07` and `08` is
shown to be illusory or algebraically wrong.

At the moment, the repo evidence supports H3.

### For H4

H4 fails if any of the following occur:

- the proof in file `10` contains an algebraic gap,
- the audit in file `11` misses a hidden assumption,
- the direct-CTMC validation code does not match the theorem statement,
- external review rejects the weighted-quadratic route as incomplete.

This is why H4 must remain conditional until formally promoted.

### For H5

H5 fails if a properly reproduced benchmark shows that Reflected UAS no longer
beats UAS and JSSQ under the declared protocol.

### For H6

H6 fails if one of the exploratory directions is later shown, under a properly
powered and reproducible study, to dominate Reflected UAS in a way that is
both practically meaningful and thesis-relevant.

### For H7

H7 fails if the differentiable-policy study is presented too strongly relative
to the available evidence, or if the thesis tries to use it as a substitute for
the mathematical core.

H7 is not meant to prove that the learned layer is the main contribution.
It is meant only to show that the smooth routing law has downstream learning
compatibility that hard policies like JSQ and JSSQ do not naturally provide.

---

## 4. Safe Thesis Claim Ladder

The thesis should use the following claim ladder and should never move a claim
up the ladder without new evidence.

### Tier 1. Fully proved

These can be stated as theorem-level results.

- H1 deterministic reflected-ODE structure
- H2 benchmark consistency with the deterministic equilibrium
- H3 identification of the old stochastic obstruction

### Tier 2. Audited but promotion-sensitive

These can be stated as theorem-level results only after advisor sign-off.

- H4 direct CTMC certification route

Without that sign-off, H4 must be labeled as:

- "audited direct CTMC proof route,"
- "internally verified stochastic certification argument,"
- or "conditional theorem pending final external theorem review."

### Tier 3. Validated empirically

These can be reported as benchmark facts, not universal laws.

- H5 benchmark performance of Reflected UAS
- H7 differentiable-policy applicability of the softmax-based routing family

### Tier 4. Exploratory negative results

These can justify scope decisions, but should not dominate the thesis.

- H6 exploratory directions did not surpass the main line

---

## 5. Recommended Thesis-Level Master Hypothesis

If the thesis must have one central hypothesis statement, it should be:

**Master hypothesis.**
Reflected UAS admits a corrected reflected-ODE stability theory with a unique
boundary equilibrium and global convergence, and this deterministic structure
provides the right backbone for a stochastic certification program whose
benchmark-default policy is already supported by targeted CTMC validation.

This is deliberately narrower than saying:

- "the full project is finished,"
- "all of Reflected UAS is fully certified,"
- "the neural policy is the main theorem-backed contribution."

The master hypothesis is strong enough for a thesis and narrow enough to defend.

---

## 6. Final Decision Rule

Before any chapter, abstract, or defense slide is written, every sentence
should be checked against these rules:

1. If the sentence depends only on files `01`, `02`, `05`, or `06`, it is
   generally safe to present as established mathematics.
2. If the sentence depends on files `07`, `08`, or `09`, it should be framed as
   a limitation, correction, or obstruction result.
3. If the sentence depends on files `10` or `11`, it must be marked as promoted
   theorem material only after explicit sign-off.
4. If the sentence depends on the neural-policy study, it should be framed as a
   supporting applicability result and must not be allowed to displace the
   theorem core.
5. If the sentence depends on probe files such as SMVR or adaptive-UAS runs, it
   should stay outside the thesis core unless it directly supports a scope
   decision.

That discipline is how this thesis avoids repeating the earlier rejection path.
