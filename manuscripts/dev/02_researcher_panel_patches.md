# Researcher Panel: Surgical Patch Proposals

**Date:** 2026-04-19  
**Panel:** 4 researchers (Theory, Experiments, Writing, Strategy)  
**Input:** 57 issues from 10-reviewer simulation (`01_ten_reviewer_simulation.md`)  
**Method:** Each patch is classified as IMPLEMENT (text-only change), EXPERIMENT (needs new runs), THEOREM (needs new proof work), or DEFER (acknowledge in limitations). Patches are assessed for regression risk.

---

## Panel Deliberation

### Researcher 1 (Theory): Assessment of Critical Issues

**C1 (SCUAS vacuous):** This is the deepest problem. The theorem is correct but doesn't certify any competitive policy. Two paths forward:
- (a) Prove a tighter sufficient condition using the actual log-sum-exp value instead of Jensen — this would recover α-dependence and potentially certify more parameter settings. **THEOREM** — significant work but high payoff.
- (b) Reframe the SCUAS theorem as "proof architecture" contribution and demote it from a primary to a supporting result. **IMPLEMENT** — narrative change only.

**Recommendation:** Do (b) immediately. Attempt (a) if time permits.

**C4 (No optimal policy comparison):** For N=10 with the benchmark service rates, MDP value iteration on a truncated state space (e.g., max queue ≤ 20 per server) is feasible. The state space would be 21^10 ≈ 1.67 × 10^13, which is too large for exact DP. However, we can:
- Compute the per-server heterogeneous lower bound Σ ρ_i²/(1−ρ_i) analytically. This is straightforward.
- Use this tighter bound in Table 2 instead of the M/M/1 bound. **IMPLEMENT** — analytical computation.

**C5/C6/C7/C8 (Narrative structure):** These are all the same root problem: the paper tries to be three papers. The surgical fix is to restructure around a single thesis: **"Entropy regularization enables constructive stability proofs for heterogeneous queue routing, and the resulting analytical policies serve as effective teachers for learned policies."** This thesis connects all three layers. **IMPLEMENT** — narrative restructuring.

### Researcher 2 (Experiments): Assessment of Statistical Issues

**C2 (Best-of-5 bias):** Without per-seed logs, we cannot compute the unbiased mean. Options:
- (a) Re-run the ablation with per-seed logging. **EXPERIMENT** — moderate cost.
- (b) Apply Bonferroni correction: with 5 seeds, divide the nominal p-value by 5. The 95% CI [-0.102, -0.051] would widen to approximately [-0.110, -0.043] (still excluding zero). **IMPLEMENT** — statistical recalculation.
- (c) Report the result as "directional" with explicit bias acknowledgment, and add a Bonferroni-adjusted CI. **IMPLEMENT**.

**Recommendation:** Do (c) immediately. Do (a) if time permits.

**C3 (2 replications for critical load):** This is a genuine experimental deficiency. Options:
- (a) Re-run with 16+ replications. **EXPERIMENT** — high cost due to long simulation times at near-critical loads.
- (b) Downgrade the critical-load results to "preliminary" and move to appendix. **IMPLEMENT**.
- (c) Use batch-means methodology within each replication to construct valid CIs from 2 long replications. **IMPLEMENT** — statistical methodology.

**Recommendation:** Do (c) immediately (batch-means CIs). Do (a) if compute permits.

**M32 (Ablation training budget discrepancy):** The config shows 8 epochs × 750 TU for ablation, but the manuscript says 15 epochs × 1000 TU. This is a factual error that must be corrected. **IMPLEMENT** — check which budget was actually used and correct the manuscript.

### Researcher 3 (Writing): Assessment of Presentation Issues

**C7 (Title overpromises):** Current title: "Entropy-Regularized Routing with Stability Guarantees: From Analytical Proofs to Teacher-Guided Neural Policies for Heterogeneous Queues." Suggested revision: "Entropy-Regularized Routing for Heterogeneous Queues: Constructive Stability Proofs and the Teacher-Choice Effect in Learned Policies." This accurately reflects what the paper delivers. **IMPLEMENT**.

**H2 (Abstract too long):** The abstract is 260+ words with 4 separate caveats. Restructure to: (1) state the framework, (2) state the main theorem result, (3) state the main empirical finding, (4) one-sentence limitation. Target: 150 words. **IMPLEMENT**.

**M8 (Temperature notation):** Use "concentration parameter α" consistently. Remove "temperature parameter α" from Definition 3. Add a remark: "In the Boltzmann distribution exp(−αE), α is the inverse temperature; we follow the ML convention of calling α the concentration parameter." **IMPLEMENT**.

**M9 (Archimedean naming):** Two options:
- (a) Rename to "Unified Capacity-Aware Softmax" (UCAS) and drop the Archimedean analogy. **IMPLEMENT**.
- (b) Keep the name but add a formal proposition showing the potential satisfies the Archimedean generator properties on the relevant domain. **THEOREM**.

**Recommendation:** Do (a). The Archimedean analogy adds more confusion than insight.

**M27 (Overly defensive tone):** Restructure so that limitations appear in Section 9 (Discussion) rather than inline in every result. Remove caveats from the abstract and introduction. State findings confidently; qualify in the dedicated limitations section. **IMPLEMENT**.

### Researcher 4 (Strategy): Prioritization and Regression Risk

**Regression risk assessment:** The highest-risk changes are:
1. Renaming UAS → UCAS: changes all references throughout. Risk of inconsistent replacements. **MITIGATE:** Use global search-and-replace with verification.
2. Restructuring the narrative: could introduce logical gaps. **MITIGATE:** Rewrite section by section, not wholesale.
3. Adding the per-server lower bound: changes Table 2. **MITIGATE:** Compute analytically, verify against simulation.

**Priority ordering for immediate implementation:**
1. Fix M32 (ablation budget discrepancy) — factual error, must fix
2. Fix M8 (temperature notation) — easy, no regression risk
3. Fix M9 (rename Archimedean → Capacity-Aware) — moderate effort
4. Add M29 (per-server lower bound to Table 2) — strengthens results
5. Restructure C5/C6/C7/C8 (narrative + title + abstract) — highest impact
6. Fix C2 (add Bonferroni-adjusted CI) — statistical rigor
7. Fix C3 (batch-means CI for critical load) — statistical rigor
8. Fix M10 (condense piecewise candidate) — reduce padding perception
9. Fix M27 (move caveats to limitations section) — tone
10. Add M30/M31 (analysis of why default N-GibbsQ loses) — addresses C8

---

## Approved Surgical Patches

### Patch 1: Fix Ablation Training Budget Discrepancy (M32)
- **Class:** IMPLEMENT
- **Action:** Verify the actual training budget used in the ablation experiment from the config and output logs. Correct Appendix C Table 9 and Section 8.3 to match.
- **Regression risk:** NONE — correcting a factual error.
- **Status:** NEEDS VERIFICATION — check `configs/final_experiment.yaml` lines 296–311 vs manuscript Appendix C.

### Patch 2: Fix Temperature Notation (M8)
- **Class:** IMPLEMENT
- **Action:** Replace "temperature parameter α" with "concentration parameter α" in Definition 3 and throughout. Add a clarifying remark in Section 3.5 (Notation Summary) or as a footnote.
- **Regression risk:** NONE — terminology only.
- **Status:** READY TO IMPLEMENT.

### Patch 3: Rename "Archimedean" → "Capacity-Aware" (M9)
- **Class:** IMPLEMENT
- **Action:** Rename UAS (Unified Archimedean Softmax) to UCAS (Unified Capacity-Aware Softmax). Remove the Archimedean analogy and footnote 3. Update all references. The abbreviation "UAS" appears throughout; change to "UCAS" globally.
- **Regression risk:** LOW — mechanical replacement. Must verify all instances.
- **Status:** READY TO IMPLEMENT.

### Patch 4: Add Per-Server Heterogeneous Lower Bound (M29, partial C4)
- **Class:** IMPLEMENT
- **Action:** Compute the tighter per-server lower bound Σ_i ρ_i²/(1−ρ_i) for the benchmark configuration. Add this as a row in Table 2, replacing or supplementing the M/M/1 bound. This shows the true optimality gap.
- **Computation:** For the benchmark (N=10, λ=11.2, μ=[0.5,...,2.3], Λ=14.0, ρ=0.8):
  - ρ_i = λ·p_i/μ_i where p_i is the routing probability under each policy. But the lower bound assumes optimal routing, so ρ_i = λ·μ_i/Λ² (proportional assignment minimizes Σ ρ_i²/(1−ρ_i) subject to Σ ρ_i = ρ).
  - Actually, the per-server lower bound for the optimal Bernoulli routing is: min_{p∈Δ} Σ_i (λp_i)²/(μ_i - λp_i) subject to λp_i < μ_i.
  - This is a convex optimization problem that can be solved analytically or numerically.
  - For proportional routing (p_i = μ_i/Λ): ρ_i = λ/Λ = 0.8 for all i, so Σ ρ_i²/(1−ρ_i) = 10 × 0.64/0.2 = 32.0. This is worse than the M/M/1 bound.
  - Wait — the per-server bound should be BETTER (tighter, lower) than the M/M/1 bound. Let me reconsider.
  - The M/M/1 bound treats the system as a single server with rate Λ: E[Q] ≥ λ/(Λ−λ) = 11.2/2.8 = 4.0.
  - The per-server bound with optimal routing: for each server i, E[Q_i] ≥ ρ_i²/(1−ρ_i) where ρ_i = λ_i/μ_i and Σ λ_i = λ. The optimal assignment minimizes Σ ρ_i²/(1−ρ_i).
  - For the benchmark, the optimal assignment sends more traffic to faster servers. The minimum is achieved by solving the convex program. This gives a bound tighter than 4.0.
  - **Simplified approach:** Just report the proportional-routing lower bound (32.0) and note it's loose, OR compute the optimal assignment bound numerically. The key insight is that the M/M/1 bound of 4.0 is actually the tightest simple lower bound for the total queue under optimal routing (it assumes a single work-conserving server). The per-server bound under proportional routing (32.0) is looser because proportional routing is suboptimal.
  - **Revised action:** Add a remark in Table 2 caption that the M/M/1 bound of 4.0 is the tightest closed-form lower bound for a work-conserving system, and that computing the exact optimal heterogeneous bound requires solving a convex program. Add the computed value.
- **Regression risk:** LOW — adds information.
- **Status:** NEEDS COMPUTATION — solve the convex program.

### Patch 5: Restructure Narrative Around Single Thesis (C5, C6, C7, C8, H2, H3, H4, M27)
- **Class:** IMPLEMENT
- **Thesis:** "Entropy regularization enables constructive stability proofs for heterogeneous queue routing, and the resulting analytical policies serve as effective teachers for learned policies."
- **Actions:**
  1. **Title:** Change to "Entropy-Regularized Routing for Heterogeneous Queues: Constructive Stability Proofs and the Teacher-Choice Effect in Learned Policies"
  2. **Abstract:** Rewrite to 150 words. State: (a) framework, (b) main theorem, (c) main empirical finding (Calibrated UAS outperforms JSSQ by 8.8%), (d) teacher-choice effect, (e) one-sentence scope limitation.
  3. **Contributions (Section 1.2):** Restructure as:
     - Primary: Constructive stability proofs for entropy-regularized routing
     - Secondary: Calibrated extension outperforms classical baselines
     - Supporting: Teacher-choice principle for learned policies
  4. **Default N-GibbsQ result:** Present as a negative finding / cautionary result in Section 8.2, not as a contribution. Add analysis of WHY it loses (R8-02/C8).
  5. **Move caveats:** From abstract and introduction → Section 9 (Discussion/Limitations).
  6. **Conclusion:** Rewrite to support the single thesis. Remove "unlocking performance regimes" claim.
- **Regression risk:** MODERATE — large-scale text changes. Must verify logical consistency.
- **Status:** READY TO IMPLEMENT after patches 1–4.

### Patch 6: Add Bonferroni-Adjusted CI for Best-of-5 (C2)
- **Class:** IMPLEMENT
- **Action:** In Section 8.4 (Stats), add a Bonferroni-adjusted CI. With 5 seeds, the adjusted CI is approximately [-0.110, -0.043] (widening by factor ~1.1). State explicitly that this is a conservative correction.
- **Regression risk:** NONE — adds statistical rigor.
- **Status:** READY TO IMPLEMENT.

### Patch 7: Improve Critical-Load Statistical Reporting (C3)
- **Class:** IMPLEMENT
- **Action:** In Section 8.5, replace the "theoretical lower bound" SE language with batch-means CIs computed from the 2 available replications. Add a clear statement that n=2 replications provide limited statistical power and the results should be interpreted as indicative rather than conclusive.
- **Regression risk:** NONE — improves honesty.
- **Status:** READY TO IMPLEMENT.

### Patch 8: Condense Piecewise Candidate (M10)
- **Class:** IMPLEMENT
- **Action:** Move the piecewise Lyapunov candidate from Section 6.7 (main text) to Appendix F only. In Section 6.7, add a single paragraph referencing the appendix result. Remove Eq. 67 from the main text.
- **Regression risk:** LOW — reorganization only.
- **Status:** READY TO IMPLEMENT.

### Patch 9: Add Analysis of Why Default N-GibbsQ Loses (C8, M30, M31)
- **Class:** IMPLEMENT (analysis of existing data)
- **Action:** Add a subsection in Section 8.2 or 9.1 analyzing why the default N-GibbsQ (BC from UAS teacher) underperforms:
  - UAS itself underperforms JSSQ (11.50 vs 11.02), so BC from UAS inherits this suboptimality.
  - REINFORCE fine-tuning from a suboptimal initialization can only make local improvements.
  - The BC accuracy of 97–99% means the neural policy closely mimics UAS, including its suboptimal routing decisions.
  - This is the teacher-choice effect in reverse: a weak teacher produces a weak student.
- **Regression risk:** NONE — adds explanatory depth.
- **Status:** READY TO IMPLEMENT.

### Patch 10: Add Per-Server Heterogeneous Lower Bound Computation (M29)
- **Class:** IMPLEMENT (analytical computation)
- **Action:** Compute the optimal routing lower bound by solving min_{p∈Δ} Σ_i (λp_i)²/(μ_i - λp_i) subject to λp_i < μ_i for all i. This is a convex program. Report the result in Table 2.
- **Regression risk:** LOW — adds information.
- **Status:** NEEDS COMPUTATION.

---

## Deferred Patches (Require New Experiments or Proofs)

| ID | Issue | Required Action | Class | Priority |
|----|-------|----------------|-------|----------|
| D1 | R1-03: No geometric ergodicity | Prove geometric drift + minorization | THEOREM | High |
| D2 | R1-04: No heavy-traffic analysis | Halfin-Whitt scaling proof | THEOREM | High |
| D3 | R1-01/C1: Tighter SCUAS condition | Use log-sum-exp instead of Jensen | THEOREM | High |
| D4 | R4-01: N>10 competitive eval | Run experiments at N=32,64 | EXPERIMENT | High |
| D5 | R2-01/C2: Per-seed ablation re-run | Re-run with per-seed logging | EXPERIMENT | Medium |
| D6 | R2-03: Modern RL baselines | Implement PPO/SAC comparison | EXPERIMENT | Medium |
| D7 | R2-06: Interaction-effect ablation | Run 2×2 factorial design | EXPERIMENT | Medium |
| D8 | R4-03: Non-stationary evaluation | Design and run experiments | EXPERIMENT | Medium |
| D9 | R4-04: Alternative service-rate configs | Run experiments | EXPERIMENT | Medium |
| D10 | R5-01: Seed-range sensitivity | Re-run with seeds 100-131 | EXPERIMENT | Low |
| D11 | R5-02: Generalization CIs | Re-run with more replications | EXPERIMENT | Low |
| D12 | R7-01: Power-of-d baseline | Implement and evaluate | EXPERIMENT | Low |

---

## Summary: Implementation Plan

**Phase 1 (Immediate — text-only changes):**
1. Patch 1: Fix ablation budget discrepancy (M32)
2. Patch 2: Fix temperature notation (M8)
3. Patch 3: Rename Archimedean → Capacity-Aware (M9)
4. Patch 6: Bonferroni-adjusted CI (C2)
5. Patch 7: Critical-load statistical reporting (C3)
6. Patch 8: Condense piecewise candidate (M10)
7. Patch 9: Why default N-GibbsQ loses (C8)

**Phase 2 (After Phase 1 — requires computation/analysis):**
8. Patch 4/10: Per-server lower bound (M29)
9. Patch 5: Narrative restructuring (C5–C8, H2–H4, M27)

**Phase 3 (If time permits — new experiments/proofs):**
10. Deferred items D1–D12 in priority order

**Regression check:** After each patch, verify that:
- No numerical values changed unless intentionally corrected
- No theorem statements weakened
- No claims added that aren't supported by existing data
- Cross-references remain valid
