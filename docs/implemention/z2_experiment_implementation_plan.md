# `z2` Experiment Implementation Plan

This document defines the experiment implementation plan for the `z2` formal
math program.

Its purpose is to make the experimental layer match the finalized mathematical
story in `docs/formal_math/z2`:

- deterministic reflected-ODE theory is the established theorem core,
- the old stochastic shortcut is retained only as correction / obstruction
  material,
- the direct CTMC route is the active stochastic certification program,
- differentiable/trainable-policy material is supportive only,
- exploratory policy-search directions are not part of the active `z2`
  experiment backbone.

This plan is intentionally selective. It does **not** try to preserve every
legacy experiment as an active thesis experiment.

## 1. Experiment Objective Hierarchy

The `z2` experiment layer should answer only four questions.

### Objective A. Deterministic correctness

Do the numerical reflected-ODE diagnostics agree with the proved deterministic
results in files `01`, `02`, `05`, and `06`?

### Objective B. Stochastic obstruction correctness

Do the computational checks support the claim that the old `H`-based route does
not automatically certify the fixed-parameter CTMC?

### Objective C. Direct CTMC certification support

Do the theorem constants, sampled drift checks, and benchmark reruns support the
weighted-quadratic direct CTMC route in files `10` and `11`?

### Objective D. Focused empirical benchmark support

Does the benchmark-default Reflected UAS point remain both:

- mathematically central to the `z2` story, and
- empirically competitive against UAS and JSSQ under the declared benchmark?

Any experiment that does not support one of these four objectives is outside the
active `z2` line.

## 2. Current Experiment Surface

The current relevant experiment files are:

### Verification directory

- `gibbsq/experiments/verification/direct_ctmc_validation.py`
- `gibbsq/experiments/legacy/sruas_validation.py`
- `gibbsq/experiments/verification/drift_verification.py`
- `gibbsq/experiments/verification/engine_parity.py`
- `gibbsq/experiments/verification/reflected_uas_proof_search.py`

### Testing directory

- `gibbsq/experiments/legacy/direction12_probe.py`
- `gibbsq/experiments/legacy/state_dependent_uas_probe.py`
- `gibbsq/experiments/legacy/smvr_probe.py`
- `gibbsq/experiments/legacy/reconciliation_proof.py`
- `gibbsq/experiments/testing/check_configs.py`
- `gibbsq/experiments/legacy/compare_softmax_variants.py`
- `gibbsq/experiments/legacy/compare_rl_variants.py`
- `gibbsq/experiments/legacy/reinforce_gradient_check.py`
- `gibbsq/experiments/legacy/stress_test.py`

### Existing z2-relevant tests

- `tests/test_direct_ctmc_validation.py`
- `tests/test_direction12_probe.py`
- `tests/test_smvr_probe.py`
- `tests/test_state_dependent_uas_probe.py`

## 3. Final Classification: Keep, Move, Archive, Add

### A. Keep active in the `z2` line

These remain active and should be treated as part of the final `z2`
experimental backbone.

#### 1. Direct CTMC validation capsule

Current file:
- `gibbsq/experiments/verification/direct_ctmc_validation.py`

Why it stays:
- it is the main computational support for files `10` and `11`
- it already checks theorem constants, sampled drift inequality, and benchmark
  candidate reruns

Required role:
- remain the authoritative validation capsule for the conditional stochastic
  route
- remain tied to `tests/test_direct_ctmc_validation.py`

Required outputs:
- theorem-constant audit summary
- sampled drift-bound audit
- benchmark policy rerun summary

#### 2. Reflected-ODE diagnostic capsule

Current source:
- `gibbsq/experiments/legacy/direction12_probe.py`

Decision:
- keep the **Direction 1** reflected-ODE diagnostic logic
- remove `Direction 2` from the active `z2` line
- move or rename this experiment into the verification layer because it now
  supports theorem files `01`, `02`, `05`, and `06`

New intended role:
- authoritative deterministic diagnostic for:
  - single-attractor support
  - terminal diameter collapse
  - equilibrium residual checks
  - benchmark consistency with the explicit boundary equilibrium

Required outputs:
- deterministic attractor summary
- equilibrium residual summary
- benchmark consistency summary

#### 3. Proof-search helper utilities

Current file:
- `gibbsq/experiments/verification/reflected_uas_proof_search.py`

Why it stays:
- the direct CTMC validation capsule depends on it for state-bank generation and
  theorem-support checks

Required role:
- utility/helper only
- not a standalone thesis experiment

#### 4. Configuration sanity and engine parity

Current files:
- `gibbsq/experiments/testing/check_configs.py`
- `gibbsq/experiments/verification/engine_parity.py`

Why they stay:
- they protect reproducibility
- they prevent experiment invalidation through configuration or engine drift

Required role:
- support experiments only
- not headline thesis figures

### B. Move out of the active `z2` line but keep in the repo

These should not be deleted, but they should no longer be treated as active
`z2` experiments.

#### 1. SRUAS validation

Current file:
- `gibbsq/experiments/legacy/sruas_validation.py`

Decision:
- demote from active `z2` experiment
- keep only as legacy certification-route record

Reason:
- the thesis no longer centers on the SRUAS sufficient-condition route
- the direct CTMC route supersedes it as the active stochastic program

Action:
- move to a legacy/archive area later, or mark clearly as superseded

#### 2. State-dependent UAS probe

Current file:
- `gibbsq/experiments/legacy/state_dependent_uas_probe.py`

Decision:
- archive from the active `z2` line

Reason:
- it is an exploratory adaptive-policy search
- it did not beat the benchmark-default Reflected UAS point
- it does not strengthen the formal `z2` theorem story

#### 3. SMVR probe

Current file:
- `gibbsq/experiments/legacy/smvr_probe.py`

Decision:
- archive from the active `z2` line

Reason:
- it is a failed exploratory policy direction
- it does not support the theorem backbone
- it should not consume thesis implementation budget

#### 4. RL- and policy-search experiments not tied to `z2`

Current files:
- `gibbsq/experiments/legacy/compare_softmax_variants.py`
- `gibbsq/experiments/legacy/compare_rl_variants.py`
- `gibbsq/experiments/legacy/reinforce_gradient_check.py`
- `gibbsq/experiments/legacy/stress_test.py`

Decision:
- keep in the repo
- exclude from the active `z2` implementation plan unless a later supporting
  appendix explicitly needs them

Reason:
- they are not part of the final `z2` mathematical certification program

### C. Add to complete the `z2` experiment line

These experiments should be added because the current surface is close, but not
yet complete, for the finalized `z2` story.

#### 1. Exact boundary-equilibrium verification experiment

Purpose:
- directly support `02_boundary_equilibrium.md`

What it must do:
- compute the scalar equilibrium solution \(K^*\) for declared benchmark and
  small auxiliary systems
- reconstruct the explicit equilibrium vector \(q^*\)
- compare that vector against the deterministic attractor returned by the
  reflected-ODE diagnostic
- report:
  - `K*`
  - equilibrium vector
  - max absolute discrepancy
  - active-set agreement

Why it is needed:
- the theorem note currently contains the benchmark check in prose, but the
  experiment layer should have a dedicated reproducible capsule for it

#### 2. Reflected-ODE multi-start convergence experiment

Purpose:
- directly support `06_global_convergence_reflected_ode.md`

What it must do:
- run the reflected-ODE diagnostic from a diverse bank of initial states
- verify convergence to the same terminal equilibrium neighborhood
- report:
  - terminal pairwise diameter
  - max equilibrium residual norm
  - convergence summary by initial-condition family

Why it is needed:
- the current Direction 1 diagnostic is close to this, but the plan should make
  it an explicit theorem-support experiment rather than a broad probe

#### 3. Small-grid exhaustive CTMC drift audit

Purpose:
- strengthen support for files `10` and `11`

What it must do:
- exhaustively enumerate small state grids for low-dimensional toy systems
- compute the exact weighted-quadratic generator drift
- verify the claimed linear drift inequality on the full grid
- produce machine-readable summaries and a compact markdown report

Why it is needed:
- the current test already does a toy-grid check, but an experiment capsule
  should expose this as a reproducible standalone validation artifact

#### 4. Independent-seed benchmark rerun

Purpose:
- strengthen the benchmark layer for the conditional CTMC route

What it must do:
- rerun UAS, benchmark-default Reflected UAS, and JSSQ on an independent seed
  block not shared with the original anchor benchmark
- report:
  - mean total queue
  - standard error
  - paired deltas versus Reflected UAS where appropriate

Why it is needed:
- it reduces the risk that the benchmark story depends too heavily on one seed
  block

#### 5. Candidate-grid theorem-constant sweep

Purpose:
- organize the direct CTMC parameter picture more cleanly

What it must do:
- sweep a declared compact grid of \((\beta,\gamma,c)\) candidates
- compute theorem constants using the direct CTMC route
- classify each candidate as:
  - positive constant and sampled-bound pass
  - positive constant but sampled-bound issue
  - non-positive constant
- highlight where the benchmark-default point sits in that grid

Why it is needed:
- it replaces the old SRUAS certified-subset storyline with the active direct
  CTMC classification picture

## 4. Required Experiment Reorganization

The active `z2` experiment line should be reorganized into these categories.

### Verification experiments

These should live under `experiments/verification`:

- direct CTMC validation
- reflected-ODE deterministic verification
- exact boundary-equilibrium verification
- small-grid exhaustive CTMC drift audit
- candidate-grid theorem-constant sweep
- engine parity and config sanity helpers

### Archived exploratory experiments

These should no longer be treated as active `z2` experiments:

- SRUAS validation
- state-dependent UAS probe
- SMVR probe
- sparsemax / Direction 2 search
- other policy-search experiments not tied to the final theorem program

If moved later, they should go to a clearly labeled archive path such as
`gibbsq/experiments/archive` or `gibbsq/experiments/exploratory_legacy`.

## 5. Detailed Implementation Sequence

Implement in this order.

### Phase 1. Freeze the active experiment set

1. Declare the active `z2` experiments:
   - direct CTMC validation
   - reflected-ODE deterministic verification
   - exact boundary-equilibrium verification
   - small-grid exhaustive CTMC drift audit
   - independent-seed benchmark rerun
   - candidate-grid theorem-constant sweep
2. Mark SRUAS, SMVR, and state-dependent UAS as non-active for `z2`.

### Phase 2. Reorganize current experiments

1. Promote Direction 1 logic out of `direction12_probe.py` into a deterministic
   verification experiment in `gibbsq/experiments/verification`.
2. Leave `direct_ctmc_validation.py` as the central stochastic validation
   capsule.
3. Keep proof-search helpers internal to verification support.
4. Keep exploratory probes in the repo, but remove them from any future `z2`
   execution pipeline.

### Phase 3. Add the missing `z2` experiments

1. Add the exact boundary-equilibrium verification capsule.
2. Add the multi-start reflected-ODE convergence capsule.
3. Add the standalone exhaustive small-grid CTMC drift audit.
4. Add the independent-seed benchmark rerun.
5. Add the candidate-grid theorem-constant sweep.

### Phase 4. Add tests

Each active `z2` experiment must have at least one targeted test.

Required tests:

- deterministic equilibrium verifier returns active-set and discrepancy fields
- multi-start reflected-ODE verifier reports convergence fields
- small-grid exhaustive drift audit reproduces bound-check logic on toy systems
- candidate-grid sweep classifies candidates correctly
- independent-seed rerun emits comparison rows with required metrics

### Phase 5. Add final artifacts and documentation

Each active experiment must write:

- machine-readable JSON or JSONL metrics
- a markdown summary for human review
- enough metadata to identify:
  - config
  - seed block
  - candidate parameters
  - theorem constants where relevant

## 6. Acceptance Criteria

The `z2` experiment implementation is complete only if all of the following are
true.

### Active-line criteria

- the active `z2` experiments align directly to the theorem/core claim ladder
- no active `z2` experiment is primarily a policy-search probe
- direct CTMC validation remains the central stochastic support capsule

### Reorganization criteria

- Direction 1 deterministic diagnostics are treated as verification, not
  broad exploratory testing
- SRUAS, SMVR, and state-dependent-UAS experiments are no longer part of the
  active `z2` line

### Coverage criteria

- deterministic equilibrium theory has a dedicated validation experiment
- global reflected-ODE convergence has a dedicated validation experiment
- the direct CTMC route has both:
  - small-system exhaustive drift validation
  - benchmark-level sampled and rerun validation

### Output criteria

- every active experiment emits both human-readable and machine-readable output
- every output clearly maps to one of the `z2` theorem or status files

## 7. Final Scope Rule

From this point onward, the `z2` implementation line should answer:

- is the deterministic theory correct and reproducible?
- is the old stochastic shortcut correctly rejected?
- is the direct CTMC route computationally supported?
- is the benchmark-default policy still the right empirical anchor?

It should **not** answer:

- can we find a more exotic policy family?
- can a neural or value-corrected extension beat Reflected UAS?
- can an exploratory adaptive variant become the new main line?

Those are no longer `z2` implementation questions.
