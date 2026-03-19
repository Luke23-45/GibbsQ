# HYPOTHESIS #1

  Statement          : "If we change the REINFORCE loss computation to include the discounting factor (gamma^t) on the log-probabilities based on the action step index, we expect metric RelErr to improve by approximately 50%+ because it will correct the temporal alignment bias between the expected discounted returns and the policy score function."

  Null hypothesis    : "The change will produce no measurable improvement in RelErr, or will cause regression in CosSim."

  Mechanism          : "Policy Gradient Theorem states that the gradient of the expected discounted return E[sum gamma^t R_t] is E[sum gamma^t G_t grad_log_pi_t]. The current implementation computes E[sum G_t grad_log_pi_t] without the gamma^t scaling on the log-probability term, leading to an over-weighting of later actions which makes the REINFORCE magnitude larger than the Finite Difference magnitude."

  Patch type         : LOGIC FIX

  Confounding factors: Batch baseline variance

  Expected impact    : HIGH
  Confidence         : HIGH
  Priority score     : 1

  Affects file(s)    : experiments/testing/reinforce_gradient_check.py
  Related SG #       : 1 (Derived from logsz.md RE mismatch)

## Hypothesis Priority Table

  Rank | H # | Patch Type      | Impact | Confidence | Files Affected | SG Link
  ─────┼─────┼─────────────────┼────────┼────────────┼────────────────┼────────
  1    | #1  | LOGIC FIX       | HIGH   | HIGH       | reinforce_gradient_check.py | SG #1
