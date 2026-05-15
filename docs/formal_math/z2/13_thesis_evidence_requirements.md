# Evidence Requirements For The Thesis Hypotheses

This note converts the hypotheses in
[12_thesis_hypotheses.md](./12_thesis_hypotheses.md) into an operational
evidence checklist.

Its purpose is practical:

- to decide what can go in the thesis core,
- to decide what still needs explicit verification,
- to prevent accidental overclaiming during writing.

---

## 1. Evidence Standard By Claim Type

Use the following standards consistently.

### Standard A. Theorem-ready

A claim is theorem-ready only if:

- the argument is written in a proof-facing note,
- the dependencies are explicit,
- the statement does not rely on unstated simulation evidence,
- and the proof has survived at least one careful internal audit.

### Standard B. Conditional theorem-ready

A claim is conditional theorem-ready if:

- the argument is written and audited,
- the code checks align with the algebra,
- but the project still wants advisor or external sign-off before presenting it
  as finished mathematics.

### Standard C. Empirically validated

A claim is empirically validated only if:

- the protocol is fixed and declared,
- the compared policies are reproduced under that same protocol,
- and the claim is stated only at the level the data actually supports.

### Standard D. Exploratory

A claim is exploratory if:

- it comes from a search or probe,
- replication is limited,
- or the result is used only to guide scope decisions.

Exploratory claims must not anchor the thesis.

---

## 2. Per-Hypothesis Evidence Checklist

### H1. Reflected-ODE structure

**Required evidence.**

- existence of the reflected model statement
- proof that interior equilibrium is impossible under \(\lambda<\Lambda\)
- proof of the boundary complementarity conditions
- proof that the reflected ODE is the correct deterministic object being studied

**Current source files.**

- [01_reflected_fluid_model.md](./01_reflected_fluid_model.md)

**Assessment.**
Meets Standard A.

### H2. Exact boundary equilibrium and convergence

**Required evidence.**

- scalar equilibrium equation derived cleanly
- uniqueness proof for \(K^*\)
- explicit formula for \(q^*\)
- proof that the convex potential identifies the same \(q^*\)
- proof that every reflected trajectory converges to \(q^*\)
- deterministic benchmark check against stored attractor

**Current source files.**

- [02_boundary_equilibrium.md](./02_boundary_equilibrium.md)
- [05_projected_gradient_structure.md](./05_projected_gradient_structure.md)
- [06_global_convergence_reflected_ode.md](./06_global_convergence_reflected_ode.md)

**Assessment.**
Meets Standard A.

### H3. Stochastic obstruction of the old route

**Required evidence.**

- explicit statement of the scaling mismatch
- exact generator calculation for the old potential \(H\)
- explicit identification of the boundary mismatch
- careful separation between "this route fails" and "no route can work"

**Current source files.**

- [07_ctmc_scaling_gap.md](./07_ctmc_scaling_gap.md)
- [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md)
- [09_review_of_suggestions.md](./09_review_of_suggestions.md)

**Assessment.**
Meets Standard A if written as a limitation/correction chapter, not as a failed
theorem chapter.

### H4. Direct CTMC certification route

**Required evidence.**

- exact weighted-quadratic generator identity
- correct softmax-minimum bound
- correct conversion from the decomposition to a linear norm drift
- explicit theorem statement with assumptions
- internal audit with no unresolved algebraic objections
- numerical validation that the theorem constants and drift inequality match the
  implemented route on toy systems and sampled benchmark state banks

**Current source files.**

- [10_direct_ctmc_quadratic_proof_attempt.md](./10_direct_ctmc_quadratic_proof_attempt.md)
- [11_audit_of_direct_ctmc_quadratic_proof.md](./11_audit_of_direct_ctmc_quadratic_proof.md)
- [direct_ctmc_validation.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/gibbsq/experiments/verification/direct_ctmc_validation.py)
- [test_direct_ctmc_validation.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/tests/test_direct_ctmc_validation.py)
- validation outputs under `outputs/direct_ctmc_validation`

**Assessment.**
Currently meets Standard B, not yet automatically Standard A.

**Promotion requirement.**
Before calling H4 a finished theorem in the thesis abstract or conclusion,
obtain one explicit approval event:

- advisor line-by-line approval, or
- final manuscript-level internal proof sign-off.

Without that event, H4 stays conditional.

### H5. Benchmark empirical performance

**Required evidence.**

- fixed benchmark system definition
- fixed protocol definition
- direct comparison of Reflected UAS, UAS, and JSSQ
- no statement stronger than the benchmark actually supports

**Current source files and artifacts.**

- quick rerun summary in
  [direct_ctmc_validation_summary.md](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/GibbsQ/outputs/direct_ctmc_validation_quick/direct_ctmc_validation/final_20260515_150449/metadata/direct_ctmc_validation_summary.md)

**Assessment.**
Meets Standard C for benchmark-level reporting.

### H6. Exploratory negative directions

**Required evidence.**

- clear result that the tested direction did not improve the benchmark in a
  meaningful way
- no attempt to repackage a non-result as a breakthrough

**Current source files and artifacts.**

- historical exploratory-probe artifacts are not present in the current
  checkout and should be regenerated or attached before H6 is cited

**Assessment.**
Does not currently meet a citation-ready evidence standard in this checkout.
Keep out of the thesis core unless the supporting artifacts are restored.

### H7. Differentiable-policy applicability

**Required evidence.**

- a clear statement that softmax-based Reflected UAS is continuous and
  differentiable, unlike hard arg-min dispatch rules such as JSQ and JSSQ
- limited but concrete evidence that this enables policy-learning workflows
  such as behavior cloning and policy-gradient fine-tuning
- explicit scope control so that this study is not mistaken for the main thesis
  result

**Current source files and artifacts.**

- the smooth softmax routing formula is present in the `z2` theorem notes
- learning-study or manuscript artifacts are not present in this checkout and
  should be restored or attached before H7 is cited as evidence-backed

**Assessment.**
Does not currently meet Standard C as an evidence-backed learning study in this
checkout. It remains a plausible supporting applicability claim, not a thesis
centerpiece.

---

## 3. What The Thesis Must Not Claim

The thesis must not claim any of the following unless new evidence is added.

### Forbidden claim A

"The full reflected family is completely certified with no remaining theorem
risk."

Reason:
that is stronger than the current promotion status of H4.

### Forbidden claim B

"The neural policy is the principal contribution of the thesis."

Reason:
the repo's strongest and most defensible contribution is the `z2` mathematical
program, not the neural layer.

### Forbidden claim B1

"Because Reflected UAS is differentiable and trainable, that is by itself the
main reason the policy matters."

Reason:
trainability is a useful supporting property, but not the core scientific claim
that carries the thesis.

### Forbidden claim C

"SMVR or the later exploratory variants are the real breakthrough."

Reason:
the recorded probe results do not support that.

### Forbidden claim D

"The old manuscript was mostly correct except for presentation."

Reason:
the old manuscript had substantive theorem-positioning problems, not just style
problems.

### Forbidden claim E

"The direct CTMC route is proved because the code audit passed."

Reason:
code audit supports a proof route; it does not replace mathematical sign-off.

---

## 4. Minimal Thesis Core If Time Becomes Tight

If you need the smallest still-defensible thesis, retain only this core:

1. corrected reflected-ODE model
2. exact boundary equilibrium characterization
3. convex potential and global convergence theorem
4. explicit stochastic obstruction of the old route
5. direct CTMC proof route plus validation, clearly labeled by promotion status
6. one focused benchmark validation section
7. optional short applicability study showing compatibility with differentiable
   policy learning

This minimal core is enough to tell a complete scientific story.

The following can be reduced first if time or risk becomes critical:

- neural-policy chapters
- large exploratory experiment sections
- negative-search detail
- architecture search narrative

---

## 5. Recommended Final Hypothesis Stack For Writing

When drafting the thesis, use these exact levels.

### Core theorem hypothesis

Reflected UAS admits a corrected deterministic reflected-ODE theory with a
unique boundary equilibrium and global convergence.

### Core correction hypothesis

The old stochastic lift was invalid because the reflected ODE Lyapunov picture
does not coincide with the exact CTMC generator at the boundary.

### Conditional stochastic hypothesis

The fixed-parameter CTMC appears certifiable by a weighted-quadratic
Foster-Lyapunov route that has passed internal audit and targeted numerical
validation.

### Validation hypothesis

The benchmark-default Reflected UAS point is both mathematically central and
empirically competitive relative to UAS and JSSQ.

### Applicability hypothesis

Because Reflected UAS is softmax-based and continuous, it is compatible with
differentiable policy-learning workflows in a way that hard dispatch rules such
as JSQ and JSSQ are not.

This applicability hypothesis is useful, but it is not the main hero of the
thesis.

If the thesis stays inside that stack, it is hard to reject for overclaiming.
