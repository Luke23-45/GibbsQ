"""
Statistical analysis and metrics for simulation trajectories.

All functions accept a ``SimResult`` and handle burn-in trimming.
Since the SSA trajectory is uniformly sampled in time (via ``sample_interval``),
time-averages over the sampled states are equivalent to exact continuous-time
integrals, up to the discretization resolution Δt.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from gibbsq.engines.numpy_engine import SimResult

__all__ = [
    "time_averaged_queue_lengths",
    "total_queue_trajectory",
    "running_average",
    "queue_length_stats",
    "gini_coefficient",
    "sojourn_time_estimate",
    "stationarity_diagnostic",
    "mser5_truncation",
    "gelman_rubin_diagnostic",
]


def _trim_burn_in(result: SimResult, fraction: float) -> np.ndarray:
    """Return the states array with the first `fraction` discarded."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"burn_in_fraction must be in [0, 1), got {fraction}")
    start_idx = int(len(result.states) * fraction)
    if start_idx >= len(result.states):
        start_idx = max(0, len(result.states) - 1)
    return result.states[start_idx:]


def time_averaged_queue_lengths(
    result: SimResult,
    burn_in_fraction: float = 0.2,
) -> np.ndarray:
    """
    Compute  E[Q_i]  for each server.

    Returns
    -------
    ndarray, shape (N,)
        Time-averaged queue lengths.
    """
    states = _trim_burn_in(result, burn_in_fraction)
    return states.mean(axis=0, dtype=np.float64)


def total_queue_trajectory(result: SimResult) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute  |Q(t)|₁ = Σ Q_i(t)  over the entire simulation.

    Returns
    -------
    times : ndarray, shape (K,)
    total_q : ndarray, shape (K,)
    """
    return result.times, result.states.sum(axis=1)


def running_average(result: SimResult) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the cumulative running average of the total queue length
    from t=0. Validates convergence.

    Returns
    -------
    times : ndarray, shape (K,)
        (Excludes t=0 to avoid division by zero).
    cum_avg : ndarray, shape (K,)
    """
    t, q = total_queue_trajectory(result)
    # Exclude first sample (t=0)
    times = t[1:]
    # Cumulative sum of samples / number of samples
    cum_avg = q[1:].cumsum() / np.arange(1, len(q))
    return times, cum_avg


def queue_length_stats(
    result: SimResult,
    burn_in_fraction: float = 0.2,
) -> dict[str, float]:
    """
    Compute mean, std dev, min, max, and 99th percentile of total Q.
    """
    states = _trim_burn_in(result, burn_in_fraction)
    q_tot = states.sum(axis=1).astype(np.float64)

    return {
        "mean":  float(q_tot.mean()),
        "std":   float(q_tot.std()),
        "min":   float(q_tot.min()),
        "max":   float(q_tot.max()),
        "p99":   float(np.percentile(q_tot, 99)),
    }


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of an array to measure load imbalance.
    (0 = perfect equality, 1 = maximal inequality).

    Algorithm based on mean absolute difference.
    """
    v = np.asarray(values, dtype=np.float64).flatten()
    if np.amin(v) < 0:
        raise ValueError("Gini coefficient undefined for negative values.")
    v = np.sort(v)
    n = len(v)
    if n < 2:
        return 0.0
    s = v.sum()
    if s == 0.0:
        return 0.0
    # Gini = [ 2 * sum(i * v_i) ] / [ n * sum(v_i) ] - (n + 1) / n
    # where i is 1-indexed. We use 1-indexed array space.
    idx = np.arange(1, n + 1)
    return (2.0 * np.dot(idx, v)) / (n * s) - (n + 1.0) / n


def sojourn_time_estimate(
    result: SimResult,
    arrival_rate: float,
    burn_in_fraction: float = 0.2,
) -> float:
    """
    Estimate expected sojourn time E[W] via Little's Law: E[Q] = λ E[W].

    NOTE: The effective arrival rate might differ slightly from the specified
    `arrival_rate` due to finite simulation time, so we use the *empirical*
    throughput (departures / time) or the theoretical arrival rate.
    We use the theoretical arrival rate assuming stationarity.
    """
    states = _trim_burn_in(result, burn_in_fraction)
    mean_q_tot = states.sum(axis=1).mean()
    return float(mean_q_tot / arrival_rate)


def stationarity_diagnostic(
    result: SimResult,
    num_windows: int = 10,
    burn_in_fraction: float = 0.2,
    p_value_threshold: float = 0.05,
) -> dict:
    """
    Test if the total queue length is stationary after burn-in.

    Strategy:
    Split the post-burn-in trajectory into `num_windows` equal chunks.
    Compute the mean total queue in each chunk.
    Run a linear regression on these chunk means over their time index.
    If the slope is significantly positive (p < threshold), the system
    is likely explosive/non-stationary.

    Returns
    -------
    dict
        Diagnostic summary including boolean `is_stationary`.
    """
    states = _trim_burn_in(result, burn_in_fraction)
    q_tot = states.sum(axis=1)

    n = len(q_tot)
    if n < num_windows:
        # Too little data to test reliably
        return {
            "is_stationary": True,  # Fallback
            "slope": 0.0,
            "p_value": 1.0,
            "warning": "Insufficient samples for stationarity test.",
        }

    chunk_size = n // num_windows
    # Truncate slight remainder - ensure integer type for slicing
    q_tot = q_tot[:int(chunk_size * num_windows)]
    windows = q_tot.reshape((int(num_windows), int(chunk_size)))
    means = windows.mean(axis=1)

    x = np.arange(num_windows)
    res = stats.linregress(x, means)

    # We only care about explosive trends (slope > 0). If slope is negative,
    # it's just settling down (still converging), but bounded.
    explosive = (res.slope > 0) and (res.pvalue < p_value_threshold)

    return {
        "is_stationary": not explosive,
        "slope":         float(res.slope),
        "p_value":       float(res.pvalue),
        "means":         means.tolist(),
    }


def mser5_truncation(trajectory: np.ndarray, batch_size: int = 5) -> int:
    """
    Compute the Marginal Standard Error Rule (MSER-5) to optimally detect
    the end of the warm-up period (initialization bias) in a simulation.

    Parameters
    ----------
    trajectory : ndarray, shape (T,)
        A single 1D trajectory of a metric (e.g., total queue length) over time.
    batch_size : int, default=5
        The size of batches to compute MSER. 5 is the standard heuristic.

    Returns
    -------
    int
        The optimal truncation point index in the original trajectory space.
        Discard data before this index to remove initialization bias.
    """
    y = np.asarray(trajectory, dtype=np.float64).flatten()
    n = len(y)
    k = n // batch_size
    
    if k < 2:
        return 0

    # Truncate to a multiple of batch_size and compute batch means
    y_trunc = y[:k * batch_size]
    z = y_trunc.reshape((k, batch_size)).mean(axis=1)

    # Limit search to ensure at least 5 batches remain for statistical significance
    max_d = max(1, k - 5)
    mser_values = np.zeros(max_d)
    
    for d in range(max_d):
        z_remain = z[d:]
        var_remain = np.var(z_remain, ddof=1) if len(z_remain) > 1 else 0.0
        rem_len = len(z_remain)
        mser_values[d] = (var_remain * (rem_len - 1)) / (rem_len ** 2)

    # Optimal d is the minimum MSER
    d_star = int(np.argmin(mser_values))
    
    return d_star * batch_size


def gelman_rubin_diagnostic(trajectories: np.ndarray) -> float:
    """
    Compute the Gelman-Rubin Potential Scale Reduction Factor (R-hat) across 
    multiple simulation replications to verify convergence to steady-state.

    Parameters
    ----------
    trajectories : ndarray, shape (M, N)
        M independent simulated chains (replications), each of length N (post-burn-in).

    Returns
    -------
    float
        The R-hat statistic. Values close to 1.0 (typically < 1.1) indicate 
        that the chains have converged to the same stationary distribution.
    """
    trajectories = np.asarray(trajectories, dtype=np.float64)
    if trajectories.ndim != 2:
        raise ValueError("trajectories must be a 2D array (M chains, N steps).")
    
    M, N = trajectories.shape
    if M < 2 or N < 2:
        return 1.0  # Cannot compute between-chain variance

    # Mean of each chain: shape (M,)
    chain_means = trajectories.mean(axis=1)
    
    # Global mean
    global_mean = chain_means.mean()

    # Between-chain variance (B)
    B = (N / (M - 1.0)) * np.sum((chain_means - global_mean) ** 2)

    # Within-chain variance (W)
    # Variance within each chain, then averaged across chains
    chain_vars = trajectories.var(axis=1, ddof=1)
    W = chain_vars.mean()

    # If W is 0, all chains are identically constant.
    if W == 0.0:
        return 1.0 if B == 0.0 else np.inf

    # Pooled variance estimate
    V_hat = ((N - 1.0) / N) * W + (1.0 / N) * B

    # R-hat
    R_hat = np.sqrt(V_hat / W)
    return float(R_hat)
