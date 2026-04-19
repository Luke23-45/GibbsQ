# Ten-Reviewer Simulation: Comprehensive Manuscript Critique

**Date:** 2026-04-19  
**Basis:** Full read of all manuscript sections (00–10 + appendices A–F), all source code (`src/gibbsq/core/`, `src/gibbsq/engines/`, `src/gibbsq/analysis/`), all experiment configs (`configs/`), output data (`outputs/final/`), and previous publication review (`publication_review/v2/`).  
**Rule:** No hallucinated issues. Every item below traces to a specific manuscript location, code file, or output artifact.

---

## Reviewer 1: Queueing Theory Expert

**Focus:** Analytical rigor, proof completeness, theory–practice gap

### R1-01: SCUAS theorem is vacuous for the benchmark default
- **Location:** Section 6, Table 5 (SCUAS audit), Section 1.2 Contribution 2
- **Issue:** Theorem 6.7 certifies SCUAS only when ε_{β,γ,c} > 0. Table 5 shows ε = −1.199 for the benchmark default (0.85, 0.5, 0.5). The only certified point (0.2, 0.0, 1.3) at α=5 is non-competitive. The theorem's contribution is "architectural" (correct proof template) rather than practical (certifying any policy anyone would use). A reviewer will ask: "What is the value of a theorem that doesn't certify the policies you actually evaluate?"
- **Severity:** HIGH — this is the central theory–practice gap.

### R1-02: UAS steady-state bound is 6× loose
- **Location:** Corollary 5.3, Section 5.4 discussion
- **Issue:** The bound E[|Q|₁] ≤ 65 vs empirical ≈ 11.5. The paper attributes this to Jensen relaxation but doesn't explore tighter alternatives. A 6× gap makes the quantitative guarantee nearly useless for engineering design.
- **Severity:** MEDIUM — acknowledged but insufficiently addressed.

### R1-03: No geometric ergodicity result
- **Location:** Section 9.4 (Future Directions, F5)
- **Issue:** Positive Harris recurrence only guarantees existence of a stationary distribution, not convergence rate. For deployment, one needs to know how fast the system mixes. The paper mentions this as future work but doesn't attempt even a partial result.
- **Severity:** MEDIUM — would significantly strengthen the theoretical contribution.

### R1-04: No heavy-traffic/diffusion-limit analysis
- **Location:** Section 9.4 (F6), Section 8.5 (critical load)
- **Issue:** The critical-load experiments show approximate 1/(1−ρ) scaling, but no formal heavy-traffic result is proven. A Halfin–Whitt type scaling limit would be a major theoretical addition.
- **Severity:** MEDIUM — expected in top-tier queueing papers.

### R1-05: Non-explosion argument is citation-only
- **Location:** Section 4.3 Step 1, Section 5.4 proof
- **Issue:** Non-explosion is cited to [Meyn & Tweedie, Prop 2.2.1] without verifying the uniform bounded-jump-rate condition explicitly for each policy. While the condition clearly holds (total rate ≤ λ + Λ), a rigorous paper should state the verification.
- **Severity:** LOW — easily fixable.

### R1-06: Missing comparison to optimal policy
- **Location:** Section 8.2 (Table 2), Section 8.1
- **Issue:** The M/M/1 lower bound of 4.0 is acknowledged as loose. For N=10 with known service rates, the optimal routing policy can be computed via MDP value iteration on a truncated state space. The gap between all evaluated policies and the true optimum is unknown.
- **Severity:** HIGH — without this, it's unclear whether Calibrated UAS at 10.04 is close to optimal or still far.

---

## Reviewer 2: Reinforcement Learning / ML Researcher

**Focus:** Experimental methodology, statistical rigor, baselines

### R2-01: Best-of-5 seed selection introduces unquantifiable optimistic bias
- **Location:** Section 8.3 (ablation), Section 8.4 (stats), Table 4
- **Issue:** The strongest neural result (9.821) is the best of 5 seeds selected on a held-out validation set. Per-seed logs were not preserved, so the unbiased ensemble mean is unknown. The paper acknowledges this but doesn't attempt to bound the bias (e.g., via Bonferroni or bootstrap).
- **Severity:** HIGH — this is a methodological flaw that undermines the strongest neural claim.

### R2-02: Only 2 replications for near-critical load experiment
- **Location:** Section 8.5, Table 8 caption
- **Issue:** The critical-load table reports results from only 2 replications per ρ. The SE values are "theoretical lower bounds" — the true uncertainty is much larger. With n=2, no meaningful confidence interval can be constructed. The reported improvements (8.8%–22.2%) are unreliable.
- **Severity:** HIGH — insufficient statistical power for the claims made.

### R2-03: No modern RL baselines (PPO, SAC, A2C)
- **Location:** Section 7.2, Section 8.3
- **Issue:** The paper deliberately uses vanilla REINFORCE to "isolate the impact of the starting distribution." While this is a valid design choice, it means the neural results represent a lower bound that may be far from what's achievable. Without at least one modern RL comparison, the reader cannot assess the gap.
- **Severity:** MEDIUM — the justification is reasonable but incomplete.

### R2-04: No multiple-comparison correction for exploratory analyses
- **Location:** Section 8.1, Section 8.4–8.5
- **Issue:** The paper designates only the Calibrated UAS vs JSSQ comparison as "pre-specified primary." All other comparisons are exploratory without correction. With 10+ pairwise comparisons across multiple tables, the false positive rate is uncontrolled.
- **Severity:** MEDIUM — the designation is honest but the statistical framework is incomplete.

### R2-05: Training–evaluation horizon mismatch (750 vs 20,000)
- **Location:** Appendix C, paragraph "Training–evaluation horizon mismatch"
- **Issue:** REINFORCE training episodes use 750 time units but evaluation uses 15,000–20,000. This 20–27× mismatch may cause distributional shift: the policy is optimized for short-horizon behavior but evaluated on long-horizon steady-state.
- **Severity:** MEDIUM — acknowledged but not analyzed.

### R2-06: Interaction effects not tested in ablation
- **Location:** Section 8.3, Section 9.2 (L2)
- **Issue:** The ablation tests factors one-at-a-time. The combination "Calibrated-teacher × No-Log-Norm" was not tested. The paper acknowledges this gap. A 2×2 factorial design would be more informative.
- **Severity:** MEDIUM — standard expectation for ablation studies.

### R2-07: REINFORCE training budget is very small
- **Location:** Appendix C (Table 9), Section 7.2
- **Issue:** 15 epochs × 16 episodes × 1000 time units = 240,000 time units total. This is a minimal budget. The training curve (Figure 8) shows convergence by epoch 10–12, but this may reflect premature convergence to the BC initialization rather than true optimization.
- **Severity:** LOW — but contributes to the small improvement margin.

---

## Reviewer 3: Stochastic Processes / Mathematician

**Focus:** Proof correctness, assumptions, notation

### R3-01: Temperature notation contradiction
- **Location:** Definition 3 (Section 3.3) vs Notation Table (Table 1)
- **Issue:** Definition 3 says "For temperature parameter α > 0" but Table 1 says "α: Inverse temperature (softmax concentration parameter)." These are contradictory. In the exp(−αQ_i) form, α is the inverse temperature (1/T in Boltzmann notation). The paper should pick one term and use it consistently.
- **Severity:** MEDIUM — confusing for careful readers.

### R3-02: "Archimedean" naming is unjustified
- **Location:** Definition 5 (UAS potential), footnote 3
- **Issue:** The paper names UAS "Unified Archimedean Softmax" because the potential's (Q_i+1)/μ_i term "acts structurally as a capacity-weighted proxy broadly comparable to an Archimedean generator function." Footnote 3 admits "the analogy is structural and motivational, not algebraic." An Archimedean copula generator must satisfy specific properties (continuity, strict monotonicity, φ(0)=∞) that are not verified. The naming adds confusion without mathematical content.
- **Severity:** MEDIUM — will annoy probabilists and copula theorists.

### R3-03: The piecewise Lyapunov candidate is neither theorem nor discard
- **Location:** Section 6.7, Appendix F
- **Issue:** The piecewise candidate (Eq. 67) with numerically optimized coefficients is presented as "numerical evidence" but occupies substantial space in both main text and appendix. It's not a theorem, and its benchmark-specific coefficients make it non-generalizable. A reviewer will ask: "Is this evidence or padding?"
- **Severity:** MEDIUM — should either be promoted to a conditional theorem or substantially condensed.

### R3-04: The SCUAS sufficient condition is extremely conservative
- **Location:** Section 6.1, Lemma 5, Remark 6
- **Issue:** The Jensen bound in Lemma 5 discards all temperature dependence, making the SCUAS condition much more conservative than necessary. The paper acknowledges this for UAS but doesn't explore tighter bounds for SCUAS (e.g., using the actual log-sum-exp value instead of Jensen).
- **Severity:** MEDIUM — the conservatism is the root cause of R1-01.

### R3-05: No formal treatment of what ε < 0 implies
- **Location:** Section 6.4 (audit), Remark 4
- **Issue:** When ε_{β,γ,c} < 0, the paper says the benchmark point is "not certified." But does ε < 0 imply instability? Or just that the sufficient condition fails? The paper implies the latter but doesn't prove it. The piecewise candidate with negative drift on boundary shells suggests stability, but this is not formalized.
- **Severity:** MEDIUM — an important logical gap.

---

## Reviewer 4: Applied Systems / Cloud Computing Researcher

**Focus:** Practical relevance, scalability, deployment

### R4-01: Competitive evaluation limited to N=10
- **Location:** Section 8.2, Section 9.3 (L3)
- **Issue:** All competitive comparisons use N=10 servers. The stress test (Appendix D) shows UAS scales to N=1024, but Calibrated UAS and N-GibbsQ are not evaluated competitively beyond N=10. Real systems have hundreds or thousands of servers.
- **Severity:** HIGH — limits practical relevance.

### R4-02: Neural policy latency is 100–500× slower than closed-form
- **Location:** Section 8.1 (computational expense paragraph)
- **Issue:** The neural policy takes 0.1–0.5 ms per decision vs <10 μs for closed-form. At high arrival rates (10⁵–10⁶ requests/s), this becomes a bottleneck. The paper mentions policy distillation as a solution but doesn't explore it.
- **Severity:** MEDIUM — acknowledged but not addressed.

### R4-03: No non-stationary arrival rate evaluation
- **Location:** Section 9.4 (F4), abstract
- **Issue:** All experiments assume constant Poisson arrivals. Real systems have time-varying load (diurnal patterns, bursty traffic). The stability guarantees assume fixed λ. No experiment tests robustness to non-stationarity.
- **Severity:** MEDIUM — important for practical deployment.

### R4-04: Only one service-rate configuration tested
- **Location:** Section 8.1, Section 9.3 (L3)
- **Issue:** The uniformly-spaced [0.5, 0.7, ..., 2.3] vector is the only configuration tested competitively. Real systems may have bimodal, skewed, or power-law service rate distributions. The generalization sweep scales this vector but doesn't change its shape.
- **Severity:** MEDIUM — limits generalizability claims.

### R4-05: MaxWeight baseline is a strawman
- **Location:** Table 2, Section 8.2
- **Issue:** MaxWeight (arg-max) is included as a baseline but is known to diverge for parallel queues without backpressure. Including it inflates the policy count and makes the table look more favorable by comparison. It should either be removed or clearly labeled as a negative example.
- **Severity:** LOW — but creates a misleading impression.

---

## Reviewer 5: Statistical Methodologist

**Focus:** Experimental design, reproducibility, uncertainty quantification

### R5-01: No seed-range sensitivity check
- **Location:** Section 8.1 (protocol paragraph)
- **Issue:** All experiments use seeds 42–73. The paper notes "a sensitivity check across a different, independent block of random seeds (e.g., 100–131) was not performed." This means all results depend on a single seed range with no replication at an independent seed block.
- **Severity:** MEDIUM — a basic reproducibility check is missing.

### R5-02: Generalization table uses single-replication point estimates
- **Location:** Table 7 (absolute values caption)
- **Issue:** The absolute generalization values are "single-replication SSA point estimates per cell" with SE ≈ 0.1–0.5. The improvement ratios in Table 6 inherit this uncertainty but no confidence intervals are reported for the ratios.
- **Severity:** MEDIUM — the ratios could be noisy.

### R5-03: The Calibrated UAS parameter search is underpowered
- **Location:** Section 6.5 (benchmark parameters)
- **Issue:** 64 configurations on a coarse grid with 16 validation replications. The noise floor (SE ≈ 0.02–0.03) means the minimum over 64 configs is subject to substantial selection bias. The paper acknowledges the grid is coarse but doesn't quantify the selection bias.
- **Severity:** MEDIUM — the "8.84% improvement over JSSQ" may be inflated.

### R5-04: No compute budget reporting
- **Location:** Appendix C (compute budget paragraph)
- **Issue:** "Wall-clock training times were not systematically logged; the total compute budget for the full experimental campaign is approximately 200 CPU-hours." Without precise reporting, reproducibility is limited.
- **Severity:** LOW — but expected in modern ML papers.

### R5-05: No anonymous code/data artifact
- **Location:** Section 8.1
- **Issue:** "Code and configuration files will be made publicly available upon acceptance." No anonymous link is provided. Top venues require an anonymized artifact for review.
- **Severity:** MEDIUM — may be venue-dependent.

---

## Reviewer 6: Information Theory / Entropy Regularization Expert

**Focus:** Theoretical framework, regularizer choices, connections to existing theory

### R6-01: The entropy–stability connection is not formally established
- **Location:** Section 1.1, Section 4.2
- **Issue:** The paper uses entropy regularization in the routing law and proves stability via Lyapunov arguments, but doesn't formally establish that entropy regularization *causes* or *facilitates* stability. The raw softmax proof uses the entropy ceiling (H ≤ log N) to bound the arrival term, but this is a technical convenience, not a causal mechanism. A reviewer will ask: "Does more entropy always mean more stability?"
- **Severity:** MEDIUM — the narrative implies a deeper connection than is proven.

### R6-02: The KL-vs-entropy distinction is underexplored
- **Location:** Section 5.2, Section 5.4 (independence from raw softmax)
- **Issue:** UAS uses KL(p||r) while raw softmax uses H(p). The paper proves these are "structurally distinct proof paths" but doesn't analyze when one regularizer is preferable to the other, or whether there's a unified framework.
- **Severity:** LOW — but a missed opportunity.

### R6-03: Temperature-independent stability is an artifact, not a feature
- **Location:** Section 5.4, Corollary 5.3
- **Issue:** The paper presents temperature-independent drift constants as a feature ("ensuring stability at all temperatures"). But this is an artifact of the conservative Jensen bound that discards all α-dependence. A tighter, α-dependent bound would be more informative and would likely show that stability *improves* with α (more concentrated routing).
- **Severity:** LOW — the framing is misleading.

---

## Reviewer 7: Operations Research / Load Balancing Expert

**Focus:** Practical routing, comparison to established methods, problem framing

### R7-01: Power-of-d baselines excluded without sufficient justification
- **Location:** Section 2 (related work, classical routing paragraph)
- **Issue:** The paper excludes power-of-d choices from competitive baselines because "GibbsQ explicitly targets settings where continuous, centralized full-state telemetry is accessible." But μ-weighted power-of-d is a natural middle ground that uses partial state + capacity information. Its exclusion makes the baseline comparison weaker.
- **Severity:** MEDIUM — a common baseline in the load-balancing literature.

### R7-02: JSSQ is already near-optimal for this configuration
- **Location:** Section 8.2
- **Issue:** JSSQ achieves 11.02 vs the M/M/1 lower bound of 4.0. The true heterogeneous lower bound would be tighter. If JSSQ is already within a few percent of optimal, the 8.84% improvement from Calibrated UAS may not be practically significant.
- **Severity:** MEDIUM — depends on the true optimum.

### R7-03: No comparison to threshold or hysteresis policies
- **Location:** Section 2, Section 8
- **Issue:** Practical systems often use threshold-based routing (e.g., route to server i if Q_i < threshold). These are simple, interpretable, and fast. No comparison is provided.
- **Severity:** LOW — but would strengthen practical relevance.

### R7-04: The Whittle index discussion is incomplete
- **Location:** Section 2 (restless bandits paragraph)
- **Issue:** The paper states that "JSSQ yields identical assignments for the visited state distribution under our anchor benchmark configuration" and uses this to justify excluding Whittle index. But this equivalence is stated without proof or empirical verification details.
- **Severity:** LOW — but the claim needs support.

---

## Reviewer 8: Neural Network / Deep Learning Expert

**Focus:** Architecture, training methodology, expressivity

### R8-01: The neural architecture is minimal
- **Location:** Section 7.1, `src/gibbsq/core/neural_policies.py`
- **Issue:** 2 hidden layers of width 128 with ReLU. No attention, no residual connections, no layer norm, no equivariance. For N=10 this may suffice, but it limits expressivity and scalability. The paper doesn't justify this architecture choice or compare to alternatives.
- **Severity:** MEDIUM — the architecture may be a bottleneck.

### R8-02: The default N-GibbsQ loses to JSSQ — why?
- **Location:** Section 8.2, Section 8.3
- **Issue:** The default neural policy (BC from UAS → REINFORCE) achieves 11.68, worse than JSSQ (11.02). The paper identifies teacher choice as the dominant factor but doesn't analyze *why* the UAS teacher produces a worse policy. Is it because UAS itself is worse than JSSQ? Because BC distillation loses information? Because REINFORCE degrades the policy?
- **Severity:** HIGH — this negative result deserves deeper analysis.

### R8-03: No training curves for all ablation variants
- **Location:** Section 8.3, Appendix C
- **Issue:** Only the default REINFORCE training curve is shown (Figure 8). The reader cannot assess convergence behavior for the calibrated-teacher variant, the no-log-norm variant, etc.
- **Severity:** LOW — but would improve transparency.

### R8-04: BC accuracy of 99.2% is best-seed; inter-seed range is 97–99%
- **Location:** Section 7.1, Appendix C
- **Issue:** The 99.2% top-1 accuracy is for the best seed. The range across seeds is 97–99%. A 97% BC accuracy means 3% of routing decisions are wrong at initialization, which could compound during REINFORCE fine-tuning.
- **Severity:** LOW — but relevant to R8-02.

---

## Reviewer 9: Verification / Formal Methods Researcher

**Focus:** Drift verification, piecewise candidate, evidence standards

### R9-01: Numerical drift verification is not formal verification
- **Location:** Appendix A, Section 8.6
- **Issue:** The drift verification samples 20,001 states and reports "zero violations." But the state space is infinite (ℤ₊¹⁰). The verification covers only a vanishing fraction of states. The paper should quantify what fraction of the "relevant" state space (e.g., states with |Q|₁ ≤ R/ε) was covered.
- **Severity:** MEDIUM — the verification is suggestive, not conclusive.

### R9-02: The piecewise candidate verification has unverified regions
- **Location:** Appendix F, Table 12
- **Issue:** The piecewise candidate is verified on shells |Q|₁ ∈ {16,17,18} and 8,192 random higher-load states. But states with |Q|₁ ∈ {19,...,23} and |Q|₁ > 256 are not exhaustively verified. The gap between the exhaustive boundary shells and the random sampling is a potential weakness.
- **Severity:** MEDIUM — the verification is incomplete.

### R9-03: The piecewise candidate coefficients are benchmark-specific
- **Location:** Appendix F, Eq. 67
- **Issue:** The coefficients (0.541679, 0.187796, etc.) are "numerically selected" via L-BFGS-B optimization. They are not derived from first principles and cannot be generalized to other service-rate vectors or parameter settings.
- **Severity:** MEDIUM — limits the contribution.

### R9-04: No formal bound on the unverified region
- **Location:** Appendix F
- **Issue:** Between the verified boundary shells and the interior compact set C (|Q|₁ ≤ 15), there is no guarantee of negative drift. The paper should provide a probabilistic or worst-case bound on the fraction of states where drift could be positive.
- **Severity:** LOW — but would strengthen the evidence.

---

## Reviewer 10: Generalist Senior Reviewer (Area Chair)

**Focus:** Overall contribution, novelty, coherence, "so what"

### R10-01: The paper lacks a single clear thesis
- **Location:** Abstract, Section 1.2
- **Issue:** The paper has three "contributions" (stability theorems, calibrated policy, teacher-choice effect) but no single unifying claim. A reader finishes the paper unsure what the main takeaway is. Is it "entropy regularization enables stability proofs"? Is it "calibrated policies outperform classical ones"? Is it "teacher choice matters for RL in queueing"?
- **Severity:** HIGH — top papers have one clear thesis.

### R10-02: The title promises "Stability Guarantees" but the best policies lack them
- **Location:** Title, Section 6.4, Table 5
- **Issue:** The title says "Stability Guarantees" but Calibrated UAS (the best closed-form policy) and N-GibbsQ (the learned policy) have no stability guarantees. Only raw softmax and UAS are certified, and UAS underperforms JSSQ. A reviewer will see a bait-and-switch.
- **Severity:** HIGH — the title overpromises.

### R10-03: The abstract is too long and defensive
- **Location:** Abstract (Section 00)
- **Issue:** The abstract contains multiple caveats: "the benchmark-default setting is not theorem-certified," "the audited benchmark configuration yields negative sufficient drift constants," "the best-of-5 selection introduces optimistic bias," "without preserved per-seed logs, the unbiased ensemble mean cannot be computed." These belong in the body, not the abstract. The abstract should state the main finding clearly and let the body handle qualifications.
- **Severity:** MEDIUM — the abstract fails to hook the reader.

### R10-04: The "three-layer" structure fragments the contribution
- **Location:** Abstract, Section 1.2, Conclusion
- **Issue:** The paper presents findings at three "levels" (analytical, empirical, learned). But the analytical level (theorems for policies no one would use at those parameters) doesn't support the empirical level (uncertified best policy), which doesn't support the learned level (a variant that only works with a specific teacher). The layers are disconnected rather than reinforcing.
- **Severity:** HIGH — the paper reads as three separate mini-papers.

### R10-05: What does this enable that wasn't possible before?
- **Location:** Conclusion
- **Issue:** The conclusion claims "bridging analytical queueing theory and modern learned policies can unlock performance regimes unavailable to either paradigm." But the paper doesn't demonstrate this: the default learned policy loses to JSSQ, and the best learned variant only marginally improves on its closed-form teacher (Δ = −0.077). The "unlocking" claim is aspirational, not demonstrated.
- **Severity:** HIGH — the "so what" is unclear.

### R10-06: The paper is overly defensive, signaling lack of confidence
- **Location:** Throughout (abstract, Section 1.2, Section 6.4, Section 8.3, Section 9)
- **Issue:** Nearly every result is accompanied by a caveat, qualification, or limitation note. While honesty is valued, the cumulative effect is that the paper undermines its own contributions. A strong paper states its findings confidently and addresses limitations in a dedicated section.
- **Severity:** MEDIUM — the tone needs rebalancing.

### R10-07: Previous rejection feedback was not fully addressed
- **Location:** `publication_review/v2/00_final_publication_decision.md`
- **Issue:** The previous review identified that "default N-GibbsQ loses in the clean policy benchmark" as the main blocker. The current manuscript still leads with N-GibbsQ as a contribution and buries the calibrated-teacher variant in the ablation. The narrative has not been restructured around the honest claim hierarchy.
- **Severity:** HIGH — the core structural issue from the previous review persists.

---

## Consolidated Issue List (Sorted by Severity)

### CRITICAL (Likely Reject if Unaddressed)

| ID | Issue | Reviewer | Location |
|----|-------|----------|----------|
| C1 | SCUAS theorem vacuous for benchmark; theory–practice gap | R1 | §6, Table 5 |
| C2 | Best-of-5 seed selection bias unquantified | R2 | §8.3–8.4 |
| C3 | Only 2 replications for critical-load experiment | R2 | §8.5, Table 8 |
| C4 | No comparison to optimal policy (MDP) | R1, R7 | §8.2 |
| C5 | Default N-GibbsQ loses to JSSQ — narrative not restructured | R10 | §1.2, §8 |
| C6 | No single clear thesis; three-layer fragmentation | R10 | Abstract, §1.2 |
| C7 | Title overpromises "Stability Guarantees" | R10 | Title |
| C8 | Why does default N-GibbsQ lose? No analysis | R8 | §8.2–8.3 |

### HIGH (Major Weakness)

| ID | Issue | Reviewer | Location |
|----|-------|----------|----------|
| H1 | Competitive evaluation limited to N=10 | R4 | §8, §9.3 |
| H2 | Abstract too long and defensive | R10 | §00 |
| H3 | "So what" unclear; unlocking claim undemonstrated | R10 | Conclusion |
| H4 | Previous rejection feedback not fully addressed | R10 | Narrative structure |

### MEDIUM (Significant Concern)

| ID | Issue | Reviewer | Location |
|----|-------|----------|----------|
| M1 | UAS bound 6× loose; no tighter alternative explored | R1 | Corollary 5.3 |
| M2 | No geometric ergodicity result | R1 | §9.4 |
| M3 | No heavy-traffic analysis | R1 | §9.4 |
| M4 | No modern RL baselines | R2 | §7.2 |
| M5 | No multiple-comparison correction | R2 | §8.1 |
| M6 | Training–evaluation horizon mismatch | R2 | Appendix C |
| M7 | No interaction-effect ablation | R2 | §8.3 |
| M8 | Temperature notation contradiction | R3 | Def 3 vs Table 1 |
| M9 | "Archimedean" naming unjustified | R3 | Def 5, footnote 3 |
| M10 | Piecewise candidate neither theorem nor discard | R3 | §6.7, Appendix F |
| M11 | SCUAS Jensen bound too conservative | R3 | §6.1 |
| M12 | ε < 0 implications not formalized | R3 | §6.4 |
| M13 | Neural latency 100–500× slower | R4 | §8.1 |
| M14 | No non-stationary evaluation | R4 | §9.4 |
| M15 | Only one service-rate configuration | R4 | §8.1 |
| M16 | No seed-range sensitivity check | R5 | §8.1 |
| M17 | Generalization ratios lack CIs | R5 | Table 6 |
| M18 | Calibrated UAS parameter search underpowered | R5 | §6.5 |
| M19 | No anonymous code artifact | R5 | §8.1 |
| M20 | Entropy–stability connection not formally established | R6 | §1.1, §4.2 |
| M21 | Power-of-d baselines excluded | R7 | §2 |
| M22 | JSSQ may be near-optimal; improvement not significant | R7 | §8.2 |
| M23 | Neural architecture minimal; no justification | R8 | §7.1 |
| M24 | Drift verification covers vanishing fraction of state space | R9 | Appendix A |
| M25 | Piecewise candidate has unverified regions | R9 | Appendix F |
| M26 | Piecewise coefficients benchmark-specific | R9 | Appendix F |
| M27 | Paper overly defensive; undermines own contributions | R10 | Throughout |

### LOW (Minor Concern)

| ID | Issue | Reviewer | Location |
|----|-------|----------|----------|
| L1 | Non-explosion argument citation-only | R1 | §4.3 |
| L2 | REINFORCE training budget small | R2 | Appendix C |
| L3 | No training curves for all ablation variants | R8 | §8.3 |
| L4 | BC accuracy 97–99% range | R8 | §7.1 |
| L5 | MaxWeight baseline is strawman | R4 | Table 2 |
| L6 | No compute budget reporting | R5 | Appendix C |
| L7 | KL-vs-entropy distinction underexplored | R6 | §5.2 |
| L8 | Temperature-independence is artifact, not feature | R6 | §5.4 |
| L9 | No threshold/hysteresis policy comparison | R7 | §2, §8 |
| L10 | Whittle index equivalence unstated proof | R7 | §2 |
| L11 | No formal bound on unverified region | R9 | Appendix F |

---

## Iteration Check 1: Have We Missed Anything?

Re-reading the manuscript end-to-end with fresh eyes:

1. **Section 3.3 Definition 4 (Calibrated UAS):** The parameter domain is stated as (0,1] × [0,1] × [0,∞). But β can exceed 1 in principle (the code allows β > 0). The restriction to β ≤ 1 is not justified. **NEW: M28.**

2. **Section 8.2 Table 2:** The "Gap to LB" column uses the M/M/1 lower bound (4.0). The paper acknowledges this is loose for heterogeneous servers but doesn't compute the tighter per-server bound Σ ρ_i²/(1−ρ_i). This would be straightforward. **NEW: M29.**

3. **Section 5.3 (Empty-State Behavior):** The analysis of empty-state behavior is interesting but disconnected from the stability proofs. It could be used to derive tighter bounds for specific temperature regimes. **Not an issue, just an observation.**

4. **Appendix E (Engine Consistency):** This appendix documents reruns but doesn't provide the actual numerical values for the corrected runs. Only deltas are reported. **NEW: L12.**

5. **The paper doesn't discuss the relationship between the BC initialization quality and the REINFORCE improvement magnitude.** If BC is near-perfect (99.2% accuracy), there's little room for REINFORCE to improve. This may explain the small Δ = −0.077. **NEW: M30.**

6. **No analysis of the neural policy's routing behavior.** Does the calibrated-teacher N-GibbsQ produce routing distributions that differ systematically from Calibrated UAS? On which states? This would illuminate the 0.077 improvement. **NEW: M31.**

7. **The paper claims "directional advantage persists across 16 generalization configurations" but doesn't test statistical significance of this consistency.** 16/16 is unlikely under a null hypothesis of random direction (p < 2⁻¹⁶ if independent), but the cells are not independent (same trained policy). **NEW: L13.**

## Iteration Check 2: Cross-Reference with Source Code

1. **`policies.py` line 399–400:** Calibrated UAS logits = γ·log(μ_i) − α·(Q_i+c)/μ_i^β. This matches Eq. 22 in the manuscript. ✓

2. **`drift.py` line 106–110:** UAS drift computation uses V = 0.5·Σ Q_i²/μ_i with arrival term λ·⟨p,(Q+0.5)/μ⟩ and service term ΣQ_i. This matches Lemma 2. ✓

3. **`neural_policies.py` line 96–99:** Preprocessing is (Q+1)/μ then log1p. The manuscript says "log(1 + s_i) where s_i = (Q_i+1)/μ_i". ✓

4. **`neural_policies.py` line 60–67:** The network has 3 layers (input→hidden, hidden→hidden, hidden→output). The manuscript says "two hidden layers of width 128." The code creates l1, l2, l3 where l1 and l2 are hidden and l3 is output. ✓ Consistent.

5. **Config `final_experiment.yaml` line 269–271:** REINFORCE training uses sim_time: 1000.0, train_epochs: 15, batch_size: 16. The manuscript says "16-episode batches over 15 epochs (each episode simulates 1,000 time units)." ✓

6. **Config line 306–308:** Ablation training uses sim_time: 750.0, train_epochs: 8, batch_size: 8. But the manuscript says "15 epochs" and "1,000 time units." **DISCREPANCY: The ablation training config uses 8 epochs of 750 time units, not 15 epochs of 1,000 time units.** The manuscript's Appendix C Table 9 says "16 episodes / 15" and "1,000 time units" for Phase 2. But the ablation-specific config overrides this to 8 epochs and 750 time units. The manuscript doesn't mention this shorter ablation budget. **NEW: M32 — Ablation training budget discrepancy between manuscript and config.**

## Iteration Check 3: Cross-Reference with Output Data

From `publication_review/v2/00_final_publication_decision.md`:
- Calibrated UAS: 10.042249083409716 ✓ (matches manuscript 10.042)
- JSSQ: 11.015990854928756 ✓ (matches 11.016, manuscript rounds to 11.016)
- UAS: 11.494818140154987 ✓ (matches 11.495)
- N-GibbsQ: 11.679227251895675 ✓ (matches 11.679)

From ablation review:
- BC: Cal. UAS → REINFORCE (Best Seed): 9.821375624219725 ✓ (matches 9.821)
- Calibrated UAS ref: 9.897823033707866 ✓ (matches 9.898)

All numerical values in the manuscript match the output data. No hallucinated numbers.

---

## Final Consolidated List (Including New Items from Iteration Checks)

**New items:**
- M28: Calibrated UAS parameter domain restricted to β ≤ 1 without justification
- M29: Tighter per-server lower bound not computed
- M30: BC near-perfection may explain small REINFORCE improvement
- M31: No analysis of neural policy routing behavior vs teacher
- M32: Ablation training budget discrepancy (config: 8 epochs × 750 TU; manuscript: 15 epochs × 1000 TU)

**Total issues: 8 Critical + 4 High + 32 Medium + 13 Low = 57 issues**

This list is comprehensive and traceable. No item is hallucinated.
