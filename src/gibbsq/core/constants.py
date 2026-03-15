"""
Shared numerical constants for GibbsQ.

These constants are synchronized with the scientific proofs in Chapter 4.
Values here are immutable and should not be modified without explicit
re-verification of the Foster-Lyapunov stability conditions.
"""

# --- Numerical Stability & CTMC Dynamics ---
# Two epsilon values are kept separate because they guard different failure modes:

# RATE_GUARD_EPSILON — SSA event-rate floor. Rates below this are treated
# as zero and no event is drawn. Original value: 1e-12.
RATE_GUARD_EPSILON: float = 1e-12

# NUMERICAL_STABILITY_EPSILON — Guards log(0) and 1/0 in continuous
# relaxations (DGA). Original value: 1e-9.
NUMERICAL_STABILITY_EPSILON: float = 1e-9

# --- Differentiable Gillespie Approximation (DGA) ---
# Sigmoid steepness (k) for the soft-indicator function 1(Q > 0).
# A value of 50.0 provides a sharp transition while maintaining gradient flow.
DGA_INDICATOR_STEEPNESS = 50.0

# Small noise added to log-probabilities in Gumbel-Softmax to prevent log(0).
GUMBEL_SMOOTHING = 1e-9

# --- Lyapunov Stability Proof ---
# Fixed additive offset in the Foster-Lyapunov compact set boundary.
LYAPUNOV_OFFSET = 1.0

# --- Default Neural Preprocessing ---
# Global scaling factor for linear queue length normalization (Q / SCALE).
# Normalized by theoretical capacity bound.
NEURAL_LINEAR_CAPACITY_BOUND = 100.0
