"""Compatibility wrapper for the active policy-comparison implementation.

The active publication-facing policy comparison lives in
``gibbsq.experiments.evaluation.baselines_comparison``. This wrapper keeps the
older benchmark module path importable without maintaining a second divergent
copy of the experiment logic.
"""

from gibbsq.experiments.evaluation.baselines_comparison import *  # noqa: F401,F403

