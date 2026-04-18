GibbsQ: From Provably Stable Entropy Routing to Guided Neural Policies for Heterogeneous Queues

## Abstract::

Routing policies for heterogeneous parallel queues struggle with a fundamental tension. Classical policies are either state-aware but oblivious to capacity, or capacity-aware but **blind** to state, while reinforcement learning **(RL)** approaches **fall short of** formal stability bounds.

We propose **GibbsQ**, a framework that addresses this gap via entropy-regularized routing. We formally demonstrate positive Harris recurrence under the strict load condition  (Λ > λ)  **for two routing laws** via Foster–Lyapunov arguments: raw softmax, established through a quadratic Lyapunov function and an entropy ceiling bound, and the Unified Archimedean Softmax (UAS), established through a capacity-weighted Lyapunov function and a proposed prior-weighted KL variational identity.

With the UAS constants being temperature independent, both of the proof yield explicit,  computable drift constants.

Through these foundations, we establish Calibrated UAS, a formal empirical generlization of UAS with three parameter **despite lacking,** a formal stability bounds which **yielding a reduction of** steady state outperforming JSSQ and UAS by 8.9% and 12.6% respectively in standard evaluation settings.

Through these foundations, we establish Calibrated UAS, a closed-form empirical generalization of UAS with three parameters. **Despite lacking** formal stability bounds,  **it demonstrates high efficacy**, **yielding a reduction of** steady-state queue lengths that outperforms JSSQ and baseline UAS by 8.9% and 12.6%, respectively, in standard evaluation settings.

We establish a teacher-choice principle: an RL agent initialized through behavior cloning from Calibrated UAS and fine-tuned via REINFORCE **attains the minimum queue length** among all evaluated policies, surpassing even its teacher. while an identical RL architecture trained via random initialization **suffers from catastrophic divergence**, suprassing its own teacher.


This learned-policy **performance gain** remains consistent across all 16 operating configurations and persists into near-critical regimes up to ρ = 0.98.

![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)


## Introductions

The dispatching of tasks to heterogeneous parallel servers is a **core challenge** in operations research, **prevalent** in server farms, cloud platforms, and communication networks. The classical policy like Join the Shortest Queue (JSQ) is optimal in under homogeneous servers however it igonores service rates. Extention of JSQ, JSSQ which is service aware for server speed remains a deterministic rule with no natural entropy-regularized generalization **providing** formal stability bounds. While RL experiences catastrophc instability when it is implemented without strict theoretical bounds, despite providing unprecedented adaptability for complex queues.
We closes this gap via designing the routing laws that depend on the current state, are aware of capacity, ensure proven stability, **and are specifically designed to support neural architectures.**


**Entropy-regularized routing.**  We follow the Gibbs-measure perspective, incoming tasks are assigned to servers using a softmax distribution based on server scores, where the temperature parameter α > 0 balances between uniform (exploratory) and greedy (exploitative) routing. The softmax function is regularized using a penalty based on Shannon entropy. The UAS policy is regularized via KL divergence in relation to a capacity-rate prior. The log-sum-exp normalizer for each formulation serves as the convex conjugate of its respective regularizer [3], linking queue routing to the concept of convex duality which allows for precise Lyapunov drift bounds through variational identities, **while naturally delivering the smooth and differentiable outputs needed for neural policy gradients.**


**Contributions.** GibbsQ makes contributions at three different levels:

1. **Stability theorems.** We establish positive Harris recurrence [10], the conventional criterion that ensures a unique stationary distribution for two routing laws under the strict load condition Λ = ∑i µi > λ: (i) raw softmax, applying a Foster–Lyapunov approach with a quadratic Lyapunov function (Theorem 4.1), and (ii) the Unified Archimedean Softmax (UAS), using a capacity-weighted Lyapunov function along with a prior-weighted KL variational identity (Theorem 5.5). Both of the proofs provide specific drift constants. **Critically**, in contrast to the raw softmax bound that diminishes as α approaches 0 the drift constants of the UAS remain unaffected by α, ensuring the stability certainty is consistent across all temperatures.
2. **Calibrated closed-form policy.** While UAS is stable through demonstration, its parameterization favors analytical simplicity rather than empirical optimal performance. Calibrated UAS policy mitigates this issue with a three-parameter extension that achieves E[∑i Qi] = 10.04 using a standard heterogeneous load protocol, surpassing JSSQ (11.02), UAS (11.50), and JSQ (11.81). The 1.45-unit margin over UAS confirms meaningful empirical improvement which is beyond the provably stable parameterization.

3. **Acquired policy through structured ablation.** While a fixed closed-form policy is limited by its parametric form, a neural policy (**N-GibbsQ**) can capture distributional structure that no formula replicates.  Our findings suggest that the decision of teacher policy is the key factor that would influence the quality of the learned policy. The neural policy that is initialized through behavior cloning from Calibrated UAS and then fine-tuned using REINFORCE achieves a score of 9.82, while the same architecture trained from UAS yields a score of only 11.14, and training from scratch leads to catastrophic divergence (mean queue of 5,451).

The advantage of the learned policy is consistent across all 16 operating regime configurations tested and remains effective under near-critical load conditions up to ρ = 0.98.

**Paper organization.** Sections 2–3 survey related work and define the model; Sections 4–5 prove the two stability theorems. Sections 6–7 introduce Calibrated UAS and N-GibbsQ; Section 8 reports experiments; Sections 9–10 discuss and conclude.


## Related works:

**Classical parallel-queue routing.** JSQ is optimal under identical servers [19, 21].  JSSQ extends JSQ by directing traffic to the server with the lowest  Qi/μiQ_i/\mu_i Qi​/μi​ [6]. The power-of-d-choices method [11] significantly decreases the tail of the queue-length distribution by sampling d ≥ 2 servers, although it does not utilize complete queue-state information. GibbsQ uses complete state information with entropy regularization, providing formal stability guarantees.

**MaxWeight and Lyapunov-based scheduling.** MaxWeight policies attain optimal throughput in constrained queueing networks [18]. Neely's drift-plus-penalty framework [12] extends Lyapunov methods to stochastic network optimization with utility maximization. Our proposed works center on routing in parallel queues rather than scheduling in multi-hop networks, and produces continuous-valued routing probabilities rather than deterministic threshold assignments.


**Softmax and entropy regularization.** In reinforcement learning, Boltzmann policies serve as standard methods for exploration [17]. The log-sum-exp function acts as the convex conjugate of the negative entropy [3]. Such duality connects entropy-regularized routing to the Gibbs-measure perspective adopted here. Entropy-regularized optimization underpins mirror descent [13, 2], soft actor-critic [7], and maximum-entropy RL [22]. Despite these relationship to the RL and optimization literatures, explicit Foster–Lyapunov analyses for softmax routing in heterogeneous queues appear to be new to the queueing literature.

**Foster–Lyapunov methods for queueing stability.** The Foster–Lyapunov framework for positive Harris recurrence is the standard stability tool for CTMCs [10]. Quadratic Lyapunov functions are commonly used in homogeneous settings [12], however, weighted versions are used in analyses involving heterogeneous cases [1]. Our UAS proof uses a capacity-weighted Lyapunov function VUAS=12∑iQi2/μiV_{\mathrm{UAS}} = \tfrac{1}{2}\sum_i Q_i^2/\mu_i VUAS​=21​∑i​Qi2​/μi​ together with a prior-weighted KL variational identity a combination that, to our knowledge, has not appeared in the queueing literature.

**Imitation learning and behavior cloning.** Behavior cloning trains a policy through supervised imitation of an expert [14]. DAgger [16] corrects the distribution-shift problem of offline BC by iteratively querying the expert under the learner's induced distribution. Our proposed work uses offline BC for initialization, then followed by REINFORCE fine-tuning to improve beside the cloned policy. The ablation study quantifies the impact of teacher quality that replacing UAS with Calibrated UAS improves the final policy by 1.32 queue-length units.

**RL for queueing and resource allocation.** RL has been applied, e.g., to queueing control, scheduling, and routing [5]. These approaches generally learn policies end-to-end without analytical stability guarantees. GibbsQ addresses this gap by offering proven stable analytical policies as both baselines and BC teachers.

**Mean-field approximations.** For large server pools (N → ∞), mean-field models approximate queueing dynamics [11, 15]. Our theoretical results are finite-N; stress tests validate UAS behavior up to N = 1,024 (Appendix D).

**Positioning.** GibbsQ offers explicit stability proofs with closed-form drift constants, an empirically calibrated closed-form policy, and ablation evidence that teacher-policy choice is the dominant training design factor.