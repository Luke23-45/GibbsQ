"""
Lyapunov drift verifier.

Computes the **exact** generator action  𝓛V(Q)  and the analytical
upper bounds from the proof at arbitrary states.  Provides both
single-state and fully-vectorised batch evaluation.

Key formulas  (§2 of the proof)
-------------------------------
    V(Q)  = ½ Σ Q_i² / μ_i  (Archimedean)
    V(Q)  = ½ Σ Q_i²        (Standard/Raw)

    𝓛V(Q) = λ Σ p_i(Q) (Q_i+0.5)/μ_i  −  Σ Q_i  +  0.5 Σ 𝟙(Q_i > 0)    (Archimedean)
    𝓛V(Q) = λ Σ p_i(Q) Q_i  −  Σ μ_i Q_i  +  C(Q)                      (Standard)

    C(Q)  = λ/2  +  ½ Σ μ_i 𝟙(Q_i > 0)

Intermediate bound  (Step 4):
    𝓛V(Q) ≤  −(Λ − λ) Q_min  −  Σ μ_i Δ_i  +  R

Simplified bound  (Step 5):
    𝓛V(Q) ≤  −ε |Q|₁  +  R

where  R = (λ log N)/α + (λ + Λ)/2,   ε = min((Λ−λ)/N, min_i μ_i).

Vectorisation strategy
----------------------
Grid evaluation for  N ≤ 3  builds the full state tensor
{0, …, q_max}^N  and computes softmax probabilities, drifts, and
bounds over the *entire* grid in one pass using broadcasting.
No Python ``for`` loop touches individual states.
"""

from __future__ import annotations

import math
import numpy as np
from gibbsq.core.config import SystemConfig, total_capacity
from gibbsq.core.policies import RoutingPolicy
from gibbsq.engines.numpy_engine import SimResult
from dataclasses import dataclass
from itertools import product
from typing import NamedTuple

__all__ = [
    "lyapunov_V",
    "generator_drift",
    "upper_bound",
    "simplified_bound",
    "verify_single",
    "DriftResult",
    "evaluate_grid",
    "evaluate_trajectory",
    "compute_adaptive_q_max",
]


# ──────────────────────────────────────────────────────────────
#  Scalar functions  (single state Q)
# ──────────────────────────────────────────────────────────────

def lyapunov_V(Q: np.ndarray, mu: np.ndarray | None = None) -> float:
    """
    Lyapunov potential V(Q).
    
    If mu is None: V(Q) = ½ ‖Q‖₂² (Standard)
    If mu is given: V(Q) = ½ Σ Q_i² / μ_i (Archimedean)
    """
    Q_f = Q.astype(np.float64)
    if mu is None:
        return 0.5 * float(np.dot(Q_f, Q_f))
    mu_f = np.asarray(mu, dtype=np.float64)
    return 0.5 * float(np.sum(Q_f**2 / mu_f))


def _softmax_probs(Q: np.ndarray, alpha: float, mu: np.ndarray, mode: str = "potential") -> np.ndarray:
    if mode == "potential":
        feat = (Q.astype(np.float64) + 1.0) / mu
        logits = -alpha * feat
    elif mode == "raw":
        feat = Q.astype(np.float64)
        logits = -alpha * feat
    elif mode == "uas":
        # UAS: p_i ∝ μ_i * exp(-α * (Q_i + 1) / μ_i)
        feat = (Q.astype(np.float64) + 1.0) / mu
        logits = -alpha * feat + np.log(mu)  # Add log(μ_i) for capacity weighting
    else:
        raise ValueError(f"Unknown drift mode: {mode}. Valid modes: 'potential', 'raw', 'uas'")
    # Ensure logits is a numpy array for .max() instability
    logits = np.asarray(logits)
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def generator_drift(
    Q: np.ndarray,
    lam: float,
    mu: np.ndarray,
    alpha: float,
    mode: str = "potential",
) -> float:
    """
    Exact generator action 𝓛V(Q) based on mode.
    
    - Logic/Raw (V = 0.5 Σ Q²): 𝓛V = λ⟨p,Q⟩ - ⟨μ,Q⟩ + C
    - Performance/UAS (V = 0.5 Σ Q²/μ): 𝓛V = λ⟨p,(Q+0.5)/μ⟩ - ΣQ + 0.5Σ𝟙(Q>0)
    """
    Q_f  = Q.astype(np.float64)
    mu_f = np.asarray(mu, dtype=np.float64)
    p    = _softmax_probs(Q, alpha, mu_f, mode=mode)
    active = (Q > 0).astype(np.float64)

    if mode == "uas":
        # Archimedean Potential V = 0.5 * sum(Q^2 / mu)
        arrival_term = lam * np.dot(p, (Q_f + 0.5) / mu_f)
        service_term = np.sum(Q_f)
        C_Q          = 0.5 * np.sum(active)
    else:
        # Standard Potential V = 0.5 * sum(Q^2)
        arrival_term  = lam * np.dot(p, Q_f)
        service_term  = np.dot(mu_f, Q_f)
        C_Q           = lam / 2.0 + 0.5 * np.dot(mu_f, active)

    return float(arrival_term - service_term + C_Q)


def upper_bound(
    Q: np.ndarray,
    lam: float,
    mu: np.ndarray,
    alpha: float,
    mode: str = "raw",
) -> float:
    """
    Intermediate bound (Step 4) branched by mode.
    """
    Q_f   = Q.astype(np.float64)
    mu_f  = np.asarray(mu, dtype=np.float64)
    N     = len(mu_f)
    cap   = mu_f.sum()
    Q_min = float(Q_f.min())
    delta = Q_f - Q_min

    if mode == "uas":
        # Archimedean R: λ log N / α + λ / (2 * min_μ) + N / 2
        R = (lam * math.log(N)) / alpha + (lam / (2.0 * mu_f.min())) + (N / 2.0)
        # In UAS weighted space, we use a heuristic bound alignment
        return float(-(cap - lam) / cap * Q_f.sum() + R)
    else:
        # Standard R: λ log N / α + (λ + Λ) / 2
        R = (lam * math.log(N)) / alpha + (lam + cap) / 2.0
        return -(cap - lam) * Q_min - float(np.dot(mu_f, delta)) + R


def simplified_bound(
    Q: np.ndarray,
    lam: float,
    mu: np.ndarray,
    alpha: float,
    mode: str = "raw",
) -> float:
    """
    Simplified bound (Step 5) branched by mode.
    """
    mu_f = np.asarray(mu, dtype=np.float64)
    N    = len(mu_f)
    cap  = mu_f.sum()

    if mode == "uas":
        eps = (cap - lam) / cap
        R   = (lam * math.log(N)) / alpha + (lam / (2.0 * mu_f.min())) + (N / 2.0)
    else:
        eps = min((cap - lam) / N, mu_f.min())
        R   = (lam * math.log(N)) / alpha + (lam + cap) / 2.0

    return -eps * float(Q.sum()) + R


def verify_single(
    Q: np.ndarray,
    lam: float,
    mu: np.ndarray,
    alpha: float,
    mode: str = "raw",
) -> dict:
    """Compute all three quantities and check both bound inequalities."""
    exact = generator_drift(Q, lam, mu, alpha, mode=mode)
    ub    = upper_bound(Q, lam, mu, alpha, mode=mode)
    sb    = simplified_bound(Q, lam, mu, alpha, mode=mode)
    TOL   = 1e-12
    return {
        "exact_drift":       exact,
        "upper_bound":       ub,
        "simplified_bound":  sb,
        "bound_holds":       exact <= ub + TOL,
        "simplified_holds":  exact <= sb + TOL,
    }


# ──────────────────────────────────────────────────────────────
#  Batch result container
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DriftResult:
    """
    Drift evaluation over a set of states.

    Every array has shape  (M,)  where  M  is the number of states.
    """

    states:            np.ndarray          # (M, N)
    exact_drifts:      np.ndarray          # (M,)
    upper_bounds:      np.ndarray          # (M,)
    simplified_bounds: np.ndarray          # (M,)
    violations:        int                 # count where exact > upper + tol
    norms:             np.ndarray          # (M,)  — |Q|₁ per state


# ──────────────────────────────────────────────────────────────
#  Vectorised grid evaluation
# ──────────────────────────────────────────────────────────────

def _vectorised_softmax(Q_all: np.ndarray, alpha: float, mu: np.ndarray, mode: str = "raw") -> np.ndarray:
    """
    Batch softmax over M states at once using selected feature mode.
    """
    if mode == "potential":
        feat = (Q_all.astype(np.float64) + 1.0) / mu
    else:
        feat = Q_all.astype(np.float64)

    logits = -alpha * feat                             # (M, N)
    logits -= logits.max(axis=1, keepdims=True)        # shift per state
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def _vectorised_drift(
    Q_all: np.ndarray,
    lam:   float,
    mu:    np.ndarray,
    alpha: float,
    mode:  str = "raw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute  exact_drifts,  upper_bounds,  simplified_bounds  for
    M  states simultaneously.   **No Python loops.**

    Returns (exact, upper, simplified), each shape (M,).
    """
    Q = Q_all.astype(np.float64)                       # (M, N)
    mu_f = np.asarray(mu, dtype=np.float64)             # (N,)
    M, N = Q.shape
    cap = mu_f.sum()

    # ── Exact drift ───────────────────────────────────
    p      = _vectorised_softmax(Q, alpha, mu_f, mode=mode)
    active = (Q > 0).astype(np.float64)
    
    if mode == "uas":
        # Archimedean Potential V = 0.5 * sum(Q^2 / mu)
        grad_arrival = (Q + 0.5) / mu_f
        arrival_term = (p * grad_arrival).sum(axis=1) * lam
        service_term = Q.sum(axis=1)
        C_Q          = 0.5 * active.sum(axis=1)
    else:
        # Standard Potential V = 0.5 * sum(Q^2)
        arrival_term = lam * (p * Q).sum(axis=1)
        service_term = Q @ mu_f
        C_Q          = lam / 2.0 + 0.5 * (active @ mu_f)

    exact = arrival_term - service_term + C_Q

    # ── Bounds calculation ────────────────────────────
    if mode == "uas":
        # Archimedean Constants
        eps = (cap - lam) / cap
        R   = (lam * math.log(N)) / alpha + (lam / (2.0 * mu_f.min())) + (N / 2.0)
    else:
        # Standard Constants
        eps = min((cap - lam) / N, mu_f.min())
        R   = (lam * math.log(N)) / alpha + (lam + cap) / 2.0

    Q_min = Q.min(axis=1)
    delta = Q - Q_min[:, np.newaxis]
    Q_1   = Q.sum(axis=1)

    if mode == "uas":
        # UAS heuristic alignment
        ub = -eps * Q_1 + R 
    else:
        ub = -(cap - lam) * Q_min - (delta @ mu_f) + R

    sb = -eps * Q_1 + R

    return exact, ub, sb


def evaluate_grid(
    lam:   float,
    mu:    np.ndarray,
    alpha: float,
    q_max: int = 50,
    mode:  str = "raw",
) -> DriftResult:
    """
    Evaluate drift on the **full** grid  {0, …, q_max}^N  for  N ≤ 3.

    The computation is fully vectorised: the grid is built as an
    (M, N)  array and all M drift values are computed in one pass.

    Raises
    ------
    ValueError
        If  N > 3  (grid is too large).
    """
    mu_f = np.asarray(mu, dtype=np.float64)
    N    = len(mu_f)

    if N > 3:
        raise ValueError(f"Grid evaluation infeasible for N = {N} > 3")

    # Build the grid as an (M, N) array without Python loops
    axes = [np.arange(q_max + 1, dtype=np.int64)] * N
    mesh = np.meshgrid(*axes, indexing="ij")
    states = np.column_stack([g.ravel() for g in mesh])   # (M, N)

    exact, ub, sb = _vectorised_drift(states, lam, mu_f, alpha, mode=mode)

    violations = int(np.count_nonzero(exact > ub + 1e-12))

    return DriftResult(
        states=states,
        exact_drifts=exact,
        upper_bounds=ub,
        simplified_bounds=sb,
        violations=violations,
        norms=states.sum(axis=1).astype(np.float64),
    )


def compute_adaptive_q_max(
    lam: float,
    mu: np.ndarray,
    alpha: float,
    safety_factor: float = 2.0,
    mode: str = "raw",
) -> int:
    """
    Compute adaptive q_max based on theoretical compact radius.

    SOTA Enhancement: Ensures grid depth covers the compact set  C 
    defined by  𝓛V(Q) ≤ −1  for  Q ∉ C.

    Parameters
    ----------
    lam : float
        Arrival rate λ.
    mu : ndarray
        Service rates μ_i.
    alpha : float
        Inverse temperature α.
    safety_factor : float
        Multiplier for compact radius (default 2.0).
    mode : str
        Drift mode: 'raw' (Standard potential) or 'uas' (Archimedean potential).

    Returns
    -------
    int
        Recommended q_max value.

    Notes
    -----
    Archimedean Constants (for UAS):
        ε = (Λ − λ) / Λ
        R = (λ log N) / α + λ / (2 * min_μ) + N / 2

    Standard Constants (for Raw Softmax):
        ε = min((Λ − λ) / N, min_i μ_i)
        R = (λ log N) / α + (λ + Λ) / 2

    Compact radius L₁ threshold: radius = (R + 1) / ε.
    """
    mu_f = np.asarray(mu, dtype=np.float64)
    N    = len(mu_f)
    cap  = mu_f.sum()

    # Branch constants by mode (Foster-Lyapunov Proof alignment)
    if mode == "uas":
        eps = (cap - lam) / cap
        R   = (lam * math.log(N)) / alpha + (lam / (2.0 * mu_f.min())) + (N / 2.0)
    else:
        eps = min((cap - lam) / N, mu_f.min())
        R   = (lam * math.log(N)) / alpha + (lam + cap) / 2.0

    if eps <= 0:
        # Capacity condition violated - return large default for diagnostic depth
        return 500

    compact_radius = (R + 1.0) / eps

    # q_max should cover compact set with safety margin
    q_max = max(50, int(np.ceil(compact_radius * safety_factor)))
    return q_max


def evaluate_trajectory(
    states: np.ndarray,
    lam:    float,
    mu:     np.ndarray,
    alpha:  float,
    mode:   str = "raw",
) -> DriftResult:
    """
    Evaluate drift at every state in a sampled trajectory.

    Parameters
    ----------
    states : ndarray, shape (K, N)
        Queue lengths from a :class:`~gibbsq.engines.numpy_engine.SimResult`.

    Returns
    -------
    DriftResult
    """
    mu_f = np.asarray(mu, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)

    exact, ub, sb = _vectorised_drift(states, lam, mu_f, alpha, mode=mode)

    violations = int(np.count_nonzero(exact > ub + 1e-12))

    return DriftResult(
        states=states,
        exact_drifts=exact,
        upper_bounds=ub,
        simplified_bounds=sb,
        violations=violations,
        norms=states.sum(axis=1).astype(np.float64),
    )
