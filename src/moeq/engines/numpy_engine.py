"""
CTMC Simulator — Gillespie Stochastic Simulation Algorithm.

Simulates the continuous-time Markov chain

    Q(t) ∈ Z₊^N

where Poisson arrivals at rate λ are routed to one of N parallel
servers according to a :class:`~moeq.core.policies.RoutingPolicy`, and
each server  i  processes jobs at exponential rate  μ_i.

Architecture
------------
* **Single RNG ownership.**  The simulator owns the sole
  ``numpy.random.Generator`` for each replication.  Policies receive
  it as an argument — they never capture their own.
* **Snapshot sampling.**  The trajectory is recorded at fixed
  intervals  Δt, producing a uniformly-spaced time vector.
* **Event counting.**  Arrivals and departures are tallied for
  post-hoc validation (e.g. Little's law cross-check).

Complexity per event:  O(N)  for rate computation + O(N)  for cum-sum
event selection, so  O(N)  total.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Sequence

from moeq.core.policies import RoutingPolicy

__all__ = [
    "SimResult",
    "simulate",
    "run_replications",
]

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Result container
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SimResult:
    """
    Immutable result of a single simulation replication.

    Attributes
    ----------
    times : ndarray, shape (K,)
        Uniformly-spaced sample times  t_k = k · Δt.
    states : ndarray, shape (K, N)
        Queue lengths at each sample time.
    arrival_count : int
        Total arrival events generated.
    departure_count : int
        Total departure events generated.
    final_time : float
        Clock value when the simulation stopped.
    num_servers : int
        N  (stored for convenience).
    """

    times:           np.ndarray
    states:          np.ndarray
    arrival_count:   int
    departure_count: int
    final_time:      float
    num_servers:     int


# ──────────────────────────────────────────────────────────────
#  Core simulator
# ──────────────────────────────────────────────────────────────

def simulate(
    *,
    num_servers:     int,
    arrival_rate:    float,
    service_rates:   np.ndarray,
    policy:          RoutingPolicy,
    sim_time:        float,
    sample_interval: float       = 0.1,
    rng:             np.random.Generator | None = None,
    log_interval:    float       = 0.0,
) -> SimResult:
    """
    Run one replication of the Gillespie SSA.

    Parameters
    ----------
    num_servers : int
        Number of parallel servers  N.
    arrival_rate : float
        Poisson arrival rate  λ > 0.
    service_rates : ndarray, shape (N,)
        Exponential service rates  μ_i > 0.
    policy : RoutingPolicy
        Routing policy implementing ``__call__(Q, rng) -> probs``.
    sim_time : float
        Maximum simulation clock value  T.
    sample_interval : float
        Time between trajectory snapshots  Δt.
    rng : Generator or None
        NumPy random generator.  Created with default seed if None.
    log_interval : float
        If  > 0, emit a log message every *log_interval* simulation
        time units to monitor progress.  0 disables logging.

    Returns
    -------
    SimResult
        Sampled trajectory and event statistics.
    """
    if rng is None:
        rng = np.random.default_rng()

    N   = num_servers
    lam = float(arrival_rate)
    mu  = np.asarray(service_rates, dtype=np.float64)

    # ── State ──
    Q = np.zeros(N, dtype=np.int64)
    t = 0.0

    # ── Output buffers ──
    max_samples = int(sim_time / sample_interval) + 2
    times_buf   = np.empty(max_samples, dtype=np.float64)
    states_buf  = np.empty((max_samples, N), dtype=np.int64)

    # Initial snapshot
    times_buf[0]  = 0.0
    states_buf[0] = Q.copy()
    sample_idx = 1
    next_sample = sample_interval

    arrival_count   = 0
    departure_count = 0

    # ── Pre-allocated work arrays ──
    rates = np.empty(2 * N, dtype=np.float64)

    # ── Progress tracking ──
    next_log = log_interval if log_interval > 0 else sim_time + 1.0

    # ── Main event loop ──
    while t < sim_time:

        # 1.  Routing probabilities
        probs = policy(Q, rng)

        # 2.  Build event-rate vector
        #     [arrival_to_0 … arrival_to_{N-1}, departure_from_0 … departure_from_{N-1}]
        rates[:N] = lam * probs
        np.multiply(mu, (Q > 0).astype(np.float64), out=rates[N:])

        # 3.  Total propensity
        a0 = rates.sum()
        if a0 <= 0.0:
            break                # degenerate — no events possible

        # 4.  Draw holding time  ~ Exp(a0)
        tau = rng.exponential(1.0 / a0)
        t  += tau
        if t >= sim_time:
            break

        # 5.  Select event via inverse CDF
        u = rng.uniform(0.0, a0)
        cumrates = rates.cumsum()
        event = int(np.searchsorted(cumrates, u, side="right"))
        event = min(event, 2 * N - 1)               # safety clamp

        # 6.  Apply transition
        if event < N:
            Q[event] += 1
            arrival_count += 1
        else:
            srv = event - N
            Q[srv] -= 1                              # safe: rate is 0 if Q=0
            departure_count += 1

        # 7.  Record snapshots at fixed intervals
        while next_sample <= t and sample_idx < max_samples:
            times_buf[sample_idx]  = next_sample
            states_buf[sample_idx] = Q.copy()
            sample_idx  += 1
            next_sample += sample_interval

        # 8.  Optional progress log
        if t >= next_log:
            pct = 100.0 * t / sim_time
            log.info(
                f"  t = {t:,.0f} / {sim_time:,.0f}  ({pct:.0f}%)"
                f"  |  arrivals={arrival_count:,}  departures={departure_count:,}"
                f"  |  Q_total={Q.sum()}"
            )
            next_log += log_interval

    return SimResult(
        times=times_buf[:sample_idx].copy(),
        states=states_buf[:sample_idx].copy(),
        arrival_count=arrival_count,
        departure_count=departure_count,
        final_time=min(t, sim_time),
        num_servers=N,
    )


# ──────────────────────────────────────────────────────────────
#  Multi-replication driver
# ──────────────────────────────────────────────────────────────

def run_replications(
    *,
    num_servers:     int,
    arrival_rate:    float,
    service_rates:   np.ndarray,
    policy:          RoutingPolicy,
    sim_time:        float,
    sample_interval: float = 0.1,
    num_replications: int  = 5,
    base_seed:       int   = 42,
    log_interval:    float = 0.0,
) -> list[SimResult]:
    """
    Run *num_replications* independent replications.

    Replication  r  uses seed  ``base_seed + r``, guaranteeing
    statistical independence while remaining fully reproducible.
    """
    mu = np.asarray(service_rates, dtype=np.float64)
    results: list[SimResult] = []

    for r in range(num_replications):
        rng = np.random.default_rng(base_seed + r)
        log.info(f"Replication {r + 1}/{num_replications}  (seed={base_seed + r})")
        result = simulate(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=mu,
            policy=policy,
            sim_time=sim_time,
            sample_interval=sample_interval,
            rng=rng,
            log_interval=log_interval,
        )
        log.info(
            f"  → {result.arrival_count:,} arrivals, "
            f"{result.departure_count:,} departures, "
            f"final Q_total = {result.states[-1].sum()}"
        )
        results.append(result)

    return results
