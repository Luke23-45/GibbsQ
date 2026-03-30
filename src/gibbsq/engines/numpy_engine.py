"""
CTMC Simulator — Gillespie Stochastic Simulation Algorithm.

Simulates the continuous-time Markov chain

    Q(t) ∈ Z₊^N

where Poisson arrivals at rate λ are routed to one of N parallel
servers according to a :class:`~gibbsq.core.policies.RoutingPolicy`, and
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
from dataclasses import dataclass

from gibbsq.core.policies import RoutingPolicy
from gibbsq.core import constants
from gibbsq.utils.progress import iter_progress

__all__ = [
    "SimResult",
    "simulate",
    "run_replications",
]

log = logging.getLogger(__name__)

def _validate_simulation_inputs(
    *,
    num_servers: int,
    arrival_rate: float,
    service_rates: np.ndarray,
    sim_time: float,
    sample_interval: float,
) -> np.ndarray:
    """Validate simulator inputs and return normalized service-rate array."""
    if num_servers < 1:
        raise ValueError(f"num_servers must be >= 1, got {num_servers}")
    if sample_interval <= 0.0:
        raise ValueError(
            f"sample_interval must be > 0, got {sample_interval}"
        )
    if sim_time < 0.0:
        raise ValueError(f"sim_time must be >= 0, got {sim_time}")
    if arrival_rate < 0.0 or not np.isfinite(arrival_rate):
        raise ValueError(
            f"arrival_rate must be finite and >= 0, got {arrival_rate}"
        )

    mu = np.asarray(service_rates, dtype=np.float64)
    if mu.shape != (num_servers,):
        raise ValueError(
            f"service_rates must have shape ({num_servers},), got {mu.shape}"
        )
    if np.any(mu <= 0.0) or not np.all(np.isfinite(mu)):
        raise ValueError(
            "service_rates must be finite and strictly positive; "
            f"got min={mu.min()}, max={mu.max()}"
        )
    return mu

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
    max_events:      int | None  = None,
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

    N = num_servers
    lam = float(arrival_rate)
    mu = _validate_simulation_inputs(
        num_servers=N,
        arrival_rate=lam,
        service_rates=service_rates,
        sim_time=sim_time,
        sample_interval=sample_interval,
    )

    Q = np.zeros(N, dtype=np.int64)
    t = 0.0

    max_samples = int(sim_time / sample_interval) + 2
    times_buf   = np.empty(max_samples, dtype=np.float64)
    states_buf  = np.empty((max_samples, N), dtype=np.int64)

    times_buf[0]  = 0.0
    states_buf[0] = Q.copy()
    sample_idx = 1
    next_sample = sample_interval

    arrival_count   = 0
    departure_count = 0

    rates = np.empty(2 * N, dtype=np.float64)

    next_log = log_interval if log_interval > 0 else sim_time + 1.0

    total_events = 0
    while t < sim_time:

        probs = np.asarray(policy(Q, rng), dtype=np.float64)
        if probs.shape != (N,):
            raise ValueError(
                f"policy returned shape {probs.shape}; expected ({N},)"
            )
        if np.any(probs < 0.0) or not np.all(np.isfinite(probs)):
            raise ValueError(
                "policy probabilities must be finite and non-negative; "
                f"got min={probs.min()}, max={probs.max()}"
            )
        probs_sum = probs.sum()
        if not np.isclose(probs_sum, 1.0, rtol=1e-10, atol=constants.RATE_GUARD_EPSILON):
            raise ValueError(
                f"policy probabilities must sum to 1.0, got {probs_sum}"
            )

        #     [arrival_to_0 … arrival_to_{N-1}, departure_from_0 … departure_from_{N-1}]
        rates[:N] = lam * probs
        np.multiply(mu, (Q > 0).astype(np.float64), out=rates[N:])

        a0 = rates.sum()
        if a0 <= constants.RATE_GUARD_EPSILON:
            break

        tau = rng.exponential(1.0 / a0)
        t  += tau
        if t >= sim_time:
            break

        u = rng.uniform(0.0, a0)
        cumrates = rates.cumsum()
        event = int(np.searchsorted(cumrates, u, side="right"))
        event = min(event, 2 * N - 1)

        pre_event_Q = Q.copy()
        if event < N:
            Q[event] += 1
            arrival_count += 1
        else:
            srv = event - N
            Q[srv] -= 1
            departure_count += 1

        total_events += 1
        if max_events is not None and total_events >= max_events:
            import warnings
            warnings.warn(
                f"NumPy engine max_events={max_events} exhausted at t={t:.2f} "
                f"before sim_time={sim_time}. Trajectory may be truncated.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        while next_sample <= t and sample_idx < max_samples:
            times_buf[sample_idx]  = next_sample
            states_buf[sample_idx] = pre_event_Q
            sample_idx  += 1
            next_sample += sample_interval

        if t >= next_log:
            pct = 100.0 * t / sim_time
            log.info(
                f"  t = {t:,.0f} / {sim_time:,.0f}  ({pct:.0f}%)"
                f"  |  arrivals={arrival_count:,}  departures={departure_count:,}"
                f"  |  Q_total={Q.sum()}"
            )
            next_log += log_interval

    while next_sample <= sim_time and sample_idx < max_samples:
        times_buf[sample_idx] = next_sample
        states_buf[sample_idx] = Q.copy()
        sample_idx += 1
        next_sample += sample_interval

    return SimResult(
        times=times_buf[:sample_idx].copy(),
        states=states_buf[:sample_idx].copy(),
        arrival_count=arrival_count,
        departure_count=departure_count,
        final_time=min(t, sim_time),
        num_servers=N,
    )

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
    max_events:      int | None = None,
    progress_desc:   str | None = None,
) -> list[SimResult]:
    """
    Run *num_replications* independent replications.

    Replication  r  uses seed  ``base_seed + r``, guaranteeing
    statistical independence while remaining fully reproducible.
    """
    mu = np.asarray(service_rates, dtype=np.float64)
    results: list[SimResult] = []

    rep_iter = iter_progress(
        range(num_replications),
        total=num_replications,
        desc=progress_desc,
        unit="rep",
        leave=False,
    ) if progress_desc else range(num_replications)

    for r in rep_iter:
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
            max_events=max_events,
        )
        log.info(
            f"  -> {result.arrival_count:,} arrivals, "
            f"{result.departure_count:,} departures, "
            f"final Q_total = {result.states[-1].sum()}"
        )
        results.append(result)

    return results
