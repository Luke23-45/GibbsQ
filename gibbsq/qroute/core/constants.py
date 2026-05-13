"""
Shared numerical constants for GibbsQ.

These constants are synchronized with the scientific proofs in Chapter 4.
Values here are immutable and should not be modified without explicit
re-verification of the Foster-Lyapunov stability conditions.
"""

# Two epsilon values are kept separate because they guard different failure modes:

RATE_GUARD_EPSILON: float = 1e-12

NUMERICAL_STABILITY_EPSILON: float = 1e-9

# Sigmoid steepness (k) for the soft-indicator function 1(Q > 0).
# k=50.0 provides gradient flow while enforcing step transition.
DGA_INDICATOR_STEEPNESS = 50.0

GUMBEL_SMOOTHING = 1e-9

# Fixed additive offset in the Foster-Lyapunov compact set boundary.
LYAPUNOV_OFFSET = 1.0

NEURAL_LINEAR_CAPACITY_BOUND = 100.0
