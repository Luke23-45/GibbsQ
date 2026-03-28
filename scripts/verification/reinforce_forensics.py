#!/usr/bin/env python3
"""
Forensic reproduction harness for REINFORCE objective mismatches.

This script isolates three layers separately:
1. Internal JAX return logic parity against an independent NumPy reimplementation.
2. Trainer return definition parity against the JAX gradient-check return definition
   on a handcrafted event trace.
3. Trainer return definition parity against the JAX gradient-check return definition
   on a real sampled SSA trajectory.

The goal is to pinpoint whether a failed gradient check comes from:
- a bug in the JAX reverse-scan implementation itself, or
- a deeper mismatch between the return definition used by training and the one used
  by the gradient-check / finite-difference path.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


from experiments.training.train_reinforce import (  # noqa: E402
    collect_trajectory_ssa,
    compute_causal_returns_to_go,
)
from gibbsq.core.config import NeuralConfig  # noqa: E402
from gibbsq.core.neural_policies import NeuralRouter  # noqa: E402
from gibbsq.core.reinforce_objective import (  # noqa: E402
    compute_action_interval_returns_numpy,
    extract_first_action_returns_jax,
)
from gibbsq.engines.jax_ssa import compute_causal_returns_jax  # noqa: E402


@dataclass
class ProbeResult:
    name: str
    passed: bool
    max_abs_diff: float
    l2_diff: float
    details: dict[str, Any]


def _numpy_action_boundary_returns(
    q_integrals: np.ndarray,
    dt: np.ndarray,
    is_arrival: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Independent NumPy reproduction of the canonical JAX return definition."""
    return compute_action_interval_returns_numpy(
        q_integrals=np.asarray(q_integrals, dtype=np.float64),
        dt=np.asarray(dt, dtype=np.float64),
        is_action=np.asarray(is_arrival, dtype=bool),
        valid_mask=np.ones_like(is_arrival, dtype=bool),
        gamma=gamma,
    )


def _trainer_vs_jax_returns(
    q_totals: np.ndarray,
    dt: np.ndarray,
    is_arrival: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare trainer and JAX return definitions on the same event history."""
    q_totals = np.asarray(q_totals, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    is_arrival = np.asarray(is_arrival, dtype=bool)
    q_integrals = q_totals * dt

    initial_jump = 1.0
    jump_times = initial_jump + np.concatenate(([0.0], np.cumsum(dt[:-1])))
    sim_time = float(jump_times[-1] + dt[-1])
    states = [np.array([q], dtype=np.float64) for q in q_totals]
    action_step_indices = np.flatnonzero(is_arrival).tolist()
    valid_mask = np.ones_like(is_arrival, dtype=bool)

    trainer_returns = compute_causal_returns_to_go(
        states=states,
        jump_times=jump_times.tolist(),
        action_step_indices=action_step_indices,
        sim_time=sim_time,
        gamma=gamma,
    )
    jax_returns_full = np.array(
        compute_causal_returns_jax(
            jnp.asarray(q_integrals),
            jnp.asarray(dt),
            jnp.asarray(is_arrival),
            jnp.asarray(valid_mask),
            gamma=gamma,
        )
    )
    jax_returns = jax_returns_full[is_arrival]
    return trainer_returns, jax_returns, q_integrals


def _make_probe(
    name: str,
    lhs: np.ndarray,
    rhs: np.ndarray,
    extra: dict[str, Any],
    atol: float = 1e-9,
) -> ProbeResult:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    diff = lhs - rhs
    max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
    l2_diff = float(np.linalg.norm(diff))
    return ProbeResult(
        name=name,
        passed=bool(max_abs_diff <= atol),
        max_abs_diff=max_abs_diff,
        l2_diff=l2_diff,
        details=extra,
    )


def probe_synthetic(gamma: float) -> list[ProbeResult]:
    """Handcrafted trace with variable action durations."""
    q_totals = np.array([6.0, 4.0, 5.0, 3.0, 7.0], dtype=np.float64)
    dt = np.array([0.5, 1.0, 0.2, 0.7, 0.4], dtype=np.float64)
    is_arrival = np.array([True, False, True, False, True], dtype=bool)
    q_integrals = q_totals * dt

    numpy_jax = _numpy_action_boundary_returns(q_integrals, dt, is_arrival, gamma)
    jax_impl = np.array(
        compute_causal_returns_jax(
            jnp.asarray(q_integrals),
            jnp.asarray(dt),
            jnp.asarray(is_arrival),
            jnp.asarray(np.ones_like(is_arrival, dtype=bool)),
            gamma=gamma,
        )
    )

    trainer_returns, jax_returns, _ = _trainer_vs_jax_returns(q_totals, dt, is_arrival, gamma)
    jax_first_return = np.array(
        extract_first_action_returns_jax(
            step_returns=jnp.asarray(jax_impl)[None, :],
            is_action=jnp.asarray(is_arrival)[None, :],
            valid_mask=jnp.asarray(np.ones_like(is_arrival, dtype=bool))[None, :],
        )
    )[0]

    return [
        _make_probe(
            name="synthetic_jax_internal_parity",
            lhs=jax_impl,
            rhs=numpy_jax,
            extra={
                "q_integrals": q_integrals.tolist(),
                "is_arrival": is_arrival.astype(int).tolist(),
                "gamma": gamma,
            },
            atol=1e-6,
        ),
        _make_probe(
            name="synthetic_trainer_vs_jax_definition",
            lhs=trainer_returns,
            rhs=jax_returns,
            extra={
                "q_totals": q_totals.tolist(),
                "dt": dt.tolist(),
                "action_positions": np.flatnonzero(is_arrival).tolist(),
                "trainer_returns": trainer_returns.tolist(),
                "jax_returns_at_actions": jax_returns.tolist(),
                "gamma": gamma,
            },
            atol=1e-6,
        ),
        _make_probe(
            name="synthetic_fd_proxy_vs_trainer_objective",
            lhs=np.array([trainer_returns[0] if trainer_returns.size else 0.0]),
            rhs=np.array([jax_first_return]),
            extra={
                "trainer_first_action_return": float(trainer_returns[0] if trainer_returns.size else 0.0),
                "jax_first_action_return": float(jax_first_return),
                "gamma": gamma,
            },
            atol=1e-6,
        ),
    ]


def probe_real_trajectory(gamma: float, seed: int, sim_time: float) -> list[ProbeResult]:
    """Real SSA trajectory probe using the current Python collector."""
    service_rates = np.array([1.0, 1.4], dtype=np.float64)
    arrival_rate = 1.6
    rho = float(arrival_rate / np.sum(service_rates))

    model = NeuralRouter(
        num_servers=2,
        config=NeuralConfig(hidden_size=16, preprocessing="log1p", init_type="zero_final"),
        service_rates=service_rates,
        key=jax.random.PRNGKey(seed),
    )
    rng = np.random.default_rng(seed)
    traj = collect_trajectory_ssa(
        policy_net=model,
        num_servers=2,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        sim_time=sim_time,
        rng=rng,
        rho=rho,
    )

    if not traj.all_states or not traj.action_step_indices:
        return [
            ProbeResult(
                name="trajectory_trainer_vs_jax_definition",
                passed=False,
                max_abs_diff=float("inf"),
                l2_diff=float("inf"),
                details={"error": "trajectory had no actions; rerun with a different seed or sim_time"},
            )
        ]

    q_totals = np.array([np.sum(s) for s in traj.all_states], dtype=np.float64)
    jump_times = np.array(traj.jump_times, dtype=np.float64)
    dt = np.diff(jump_times, append=sim_time)
    is_arrival = np.zeros(len(q_totals), dtype=bool)
    is_arrival[np.array(traj.action_step_indices, dtype=int)] = True

    trainer_returns = compute_causal_returns_to_go(
        states=traj.all_states,
        jump_times=traj.jump_times,
        action_step_indices=traj.action_step_indices,
        sim_time=sim_time,
        gamma=gamma,
    )
    jax_returns_full = np.array(
        compute_causal_returns_jax(
            jnp.asarray(q_totals * dt),
            jnp.asarray(dt),
            jnp.asarray(is_arrival),
            jnp.asarray(np.ones_like(is_arrival, dtype=bool)),
            gamma=gamma,
        )
    )
    jax_returns = jax_returns_full[is_arrival]
    jax_first_return = np.array(
        extract_first_action_returns_jax(
            step_returns=jnp.asarray(jax_returns_full)[None, :],
            is_action=jnp.asarray(is_arrival)[None, :],
            valid_mask=jnp.asarray(np.ones_like(is_arrival, dtype=bool))[None, :],
        )
    )[0]

    return [
        _make_probe(
            name="trajectory_trainer_vs_jax_definition",
            lhs=trainer_returns,
            rhs=jax_returns,
            extra={
                "num_steps": int(len(q_totals)),
                "num_actions": int(len(traj.action_step_indices)),
                "action_positions": traj.action_step_indices,
                "trainer_returns_head": trainer_returns[:10].tolist(),
                "jax_returns_head": jax_returns[:10].tolist(),
                "gamma": gamma,
                "seed": seed,
                "sim_time": sim_time,
            },
            atol=1e-5,
        ),
        _make_probe(
            name="trajectory_fd_proxy_vs_trainer_objective",
            lhs=np.array([trainer_returns[0] if trainer_returns.size else 0.0]),
            rhs=np.array([jax_first_return]),
            extra={
                "trainer_first_action_return": float(trainer_returns[0] if trainer_returns.size else 0.0),
                "jax_first_action_return": float(jax_first_return),
                "gamma": gamma,
                "seed": seed,
                "sim_time": sim_time,
            },
            atol=1e-5,
        ),
    ]


def render_report(results: list[ProbeResult]) -> str:
    lines = []
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"[{status}] {result.name}: "
            f"max_abs_diff={result.max_abs_diff:.6f}, l2_diff={result.l2_diff:.6f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Forensic reproduction harness for REINFORCE mismatches.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor to probe.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the real SSA trajectory probe.")
    parser.add_argument("--sim-time", type=float, default=12.0, help="Simulation horizon for the real SSA probe.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "forensics" / "reinforce_forensics.json",
        help="Path to write the JSON report.",
    )
    args = parser.parse_args()

    results = []
    results.extend(probe_synthetic(args.gamma))
    results.extend(probe_real_trajectory(args.gamma, args.seed, args.sim_time))

    print(render_report(results))

    payload = {
        "gamma": args.gamma,
        "seed": args.seed,
        "sim_time": args.sim_time,
        "results": [asdict(r) for r in results],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nReport written to {args.json_out}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
