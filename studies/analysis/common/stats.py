"""
Statistical testing utilities for the GibbsQ analysis framework.

Provides:
- Bootstrap confidence intervals (BCa method for small N).
- Cohen's d effect size.
- P-value formatting for tables.
- Comprehensive comparison summaries.

Design notes
------------
* With only N=3 seeds, parametric assumptions are unreliable.
  We therefore emphasize **bootstrap CIs** and **effect sizes**
  over p-values, following best practices for small-sample
  experimental research.
* Bootstrap uses the BCa (bias-corrected and accelerated) method
  via ``scipy.stats.bootstrap`` when available, with a fallback
  to percentile-based CIs for older scipy versions.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a bootstrap confidence interval computation.

    Attributes
    ----------
    mean : float
        Sample mean of the input values.
    ci_low : float
        Lower bound of the confidence interval.
    ci_high : float
        Upper bound of the confidence interval.
    alpha : float
        Significance level (e.g. 0.05 for 95% CI).
    n_boot : int
        Number of bootstrap resamples performed.
    n_samples : int
        Number of original data points.
    """

    mean: float
    ci_low: float
    ci_high: float
    alpha: float
    n_boot: int
    n_samples: int

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_high - self.ci_low

    def __repr__(self) -> str:
        pct = int((1 - self.alpha) * 100)
        return (
            f"BootstrapResult(mean={self.mean:.4f}, "
            f"{pct}% CI=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"n={self.n_samples})"
        )


def bootstrap_ci(
    values: Union[Sequence[float], np.ndarray],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    method: str = "BCa",
    random_state: Optional[int] = 42,
) -> BootstrapResult:
    """Compute a bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : array-like
        Sample values (e.g. metric scores across seeds).
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Significance level.  ``0.05`` gives a 95% CI.
    method : str
        Bootstrap method: ``"BCa"`` (bias-corrected and accelerated),
        ``"percentile"``, or ``"basic"``.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
        Contains ``mean``, ``ci_low``, ``ci_high``.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]  # drop NaNs

    if len(arr) == 0:
        return BootstrapResult(
            mean=np.nan, ci_low=np.nan, ci_high=np.nan,
            alpha=alpha, n_boot=0, n_samples=0,
        )

    sample_mean = float(np.mean(arr))

    if len(arr) == 1:
        return BootstrapResult(
            mean=sample_mean, ci_low=sample_mean, ci_high=sample_mean,
            alpha=alpha, n_boot=0, n_samples=1,
        )

    # Try scipy.stats.bootstrap (scipy >= 1.7)
    try:
        rng = np.random.default_rng(random_state)
        result = sp_stats.bootstrap(
            (arr,),
            statistic=np.mean,
            n_resamples=n_boot,
            confidence_level=1 - alpha,
            method=method.lower(),
            random_state=rng,
        )
        ci_low = float(result.confidence_interval.low)
        ci_high = float(result.confidence_interval.high)

    except (TypeError, ValueError) as exc:
        # Fallback to manual percentile bootstrap
        logger.debug(
            "scipy bootstrap failed (%s), using manual percentile method",
            exc,
        )
        rng = np.random.RandomState(random_state)
        boot_means = np.array([
            np.mean(rng.choice(arr, size=len(arr), replace=True))
            for _ in range(n_boot)
        ])
        ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        mean=sample_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        n_boot=n_boot,
        n_samples=len(arr),
    )


def cohens_d(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
) -> float:
    """Compute Cohen's d effect size (pooled standard deviation).

    Parameters
    ----------
    a, b : array-like
        Two groups of observations.

    Returns
    -------
    float
        Cohen's d.  Positive means *a* > *b*.

    Interpretation
    --------------
    |d| < 0.2 : negligible
    0.2 ≤ |d| < 0.5 : small
    0.5 ≤ |d| < 0.8 : medium
    |d| ≥ 0.8 : large
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    n_a, n_b = len(a_arr), len(b_arr)

    if n_a < 2 or n_b < 2:
        logger.warning(
            "Cohen's d requires at least 2 observations per group "
            "(got %d, %d)",
            n_a,
            n_b,
        )
        return np.nan

    mean_diff = float(np.mean(a_arr) - np.mean(b_arr))
    var_a = float(np.var(a_arr, ddof=1))
    var_b = float(np.var(b_arr, ddof=1))

    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    if pooled_std == 0:
        return 0.0 if mean_diff == 0 else np.inf * np.sign(mean_diff)

    return mean_diff / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Human-readable interpretation of Cohen's d magnitude."""
    ad = abs(d)
    if np.isnan(ad):
        return "undefined"
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def format_pvalue(p: float, *, threshold: float = 0.001) -> str:
    """Format a p-value for display in text or tables.

    Parameters
    ----------
    p : float
        Raw p-value.
    threshold : float
        Below this value, display as ``"< threshold"``.

    Returns
    -------
    str
        Formatted string, e.g. ``"< 0.001"``, ``"0.032"``
    """
    if np.isnan(p):
        return "n.a."
    if p < threshold:
        return f"< {threshold}"
    return f"{p:.3f}"
