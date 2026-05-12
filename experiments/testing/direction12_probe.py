"""
Compare professor Direction 1 and Direction 2 on the anchor benchmark.

Direction 1:
    Numerical fluid-limit diagnostics for the existing Calibrated UAS policy.
    This does not search for a better policy; it tests whether the calibrated
    empirical winner looks dynamically well-behaved under a projected fluid ODE.

Direction 2:
    Empirical Fenchel-Young proxy search using sparsemax over UAS-style logits.
    This is a concrete Euclidean-regularized alternative to the current softmax
    family. It is an empirical probe only; no theorem claim is made here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from gibbsq.analysis.metrics import (
    gini_coefficient,
    sojourn_time_estimate,
    time_averaged_queue_lengths,
)
from gibbsq.core.config import ExperimentConfig, load_experiment_config
from gibbsq.core.policies import RoutingPolicy
from gibbsq.engines.numpy_engine import run_replications
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.progress import iter_progress
from gibbsq.utils.run_artifacts import artifacts_dir, metrics_path

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _select_probe_value(raw_cfg: DictConfig, key: str, default):
    value = OmegaConf.select(raw_cfg, f"probe.{key}", default=default)
    return default if value is None else value


def _standard_error(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _load_reference_metrics() -> tuple[dict[str, dict[str, float]], Path | None]:
    candidates = []
    for root in (
        PROJECT_ROOT / "outputs" / "final" / "policy",
        PROJECT_ROOT / "outputs_legacy" / "outputs" / "final" / "policy",
    ):
        if root.exists():
            candidates.extend(root.glob("*/metrics/corrected_comparison_metrics.jsonl"))

    if not candidates:
        return {}, None

    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    refs: dict[str, dict[str, float]] = {}
    with newest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            policy = str(row.get("policy"))
            refs[policy] = {
                "mean_q_total": float(row["mean_q_total"]),
                "se_q_total": float(row.get("se_q_total", 0.0)),
                "mean_gini": float(row.get("mean_gini", float("nan"))),
                "mean_sojourn": float(row.get("mean_sojourn", float("nan"))),
            }
    return refs, newest


def _sparsemax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError(f"sparsemax expects 1D logits, got shape {z.shape}")
    z_sorted = np.sort(z)[::-1]
    z_cumsum = np.cumsum(z_sorted)
    ks = np.arange(1, z.size + 1, dtype=np.float64)
    support = 1.0 + ks * z_sorted > z_cumsum
    if not np.any(support):
        return np.full_like(z, 1.0 / z.size)
    k = int(np.max(np.nonzero(support)[0])) + 1
    tau = (z_cumsum[k - 1] - 1.0) / k
    probs = np.maximum(z - tau, 0.0)
    total = probs.sum()
    if total <= 0.0:
        return np.full_like(z, 1.0 / z.size)
    return probs / total


class SparsemaxEnergyRouting:
    """Fenchel-Young proxy policy using sparsemax over calibrated-UAS logits."""

    __slots__ = ("_mu", "_alpha", "_beta", "_gamma", "_c", "_label")

    def __init__(
        self,
        mu: np.ndarray,
        *,
        alpha: float,
        beta: float,
        gamma: float,
        c: float,
        label: str,
    ) -> None:
        mu = np.asarray(mu, dtype=np.float64)
        if np.any(mu <= 0):
            raise ValueError("All service rates must be > 0")
        self._mu = mu
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._gamma = float(gamma)
        self._c = float(c)
        self._label = str(label)

    @property
    def label(self) -> str:
        return self._label

    def __call__(self, Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        q = Q.astype(np.float64)
        logits = self._gamma * np.log(self._mu) - self._alpha * ((q + self._c) / (self._mu ** self._beta))
        return _sparsemax(logits)


@dataclass(frozen=True)
class Direction2Spec:
    label: str
    alpha: float
    beta: float
    gamma: float
    c: float


def _direction2_candidates(search_mode: str) -> list[Direction2Spec]:
    search_mode = str(search_mode).lower()
    if search_mode not in {"quick", "full"}:
        raise ValueError(f"search_mode must be 'quick' or 'full', got {search_mode!r}")

    alphas = [5.0, 10.0, 20.0, 50.0]
    betas = [0.85, 1.0]
    gammas = [0.5, 1.0]
    offsets = [0.5, 1.0]

    if search_mode == "full":
        alphas.extend([2.0, 100.0])
        betas.extend([0.7])
        gammas.extend([0.25, 0.75])
        offsets.extend([0.25, 0.75])

    candidates: list[Direction2Spec] = []
    seen: set[tuple[float, float, float, float]] = set()
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                for c in offsets:
                    key = (alpha, beta, gamma, c)
                    if key in seen:
                        continue
                    seen.add(key)
                    label = f"sparsemax_a({alpha:g})_b({beta:g})_g({gamma:g})_c({c:g})"
                    candidates.append(Direction2Spec(label, alpha, beta, gamma, c))
    return candidates


def _evaluate_policy(
    policy: RoutingPolicy,
    cfg: ExperimentConfig,
    *,
    num_replications: int,
    sim_time: float,
    sample_interval: float,
    base_seed: int,
    progress_desc: str,
) -> dict[str, float]:
    results = run_replications(
        num_servers=cfg.system.num_servers,
        arrival_rate=cfg.system.arrival_rate,
        service_rates=np.asarray(cfg.system.service_rates, dtype=np.float64),
        policy=policy,
        num_replications=num_replications,
        sim_time=sim_time,
        sample_interval=sample_interval,
        base_seed=base_seed,
        progress_desc=progress_desc,
    )

    burn_in = cfg.simulation.burn_in_fraction
    q_totals = [float(time_averaged_queue_lengths(r, burn_in).sum()) for r in results]
    ginis = [float(gini_coefficient(time_averaged_queue_lengths(r, burn_in))) for r in results]
    sojourns = [
        float(sojourn_time_estimate(r, cfg.system.arrival_rate, burn_in)) for r in results
    ]
    return {
        "mean_q_total": float(np.mean(q_totals)),
        "se_q_total": _standard_error(q_totals),
        "mean_gini": float(np.mean(ginis)),
        "mean_sojourn": float(np.mean(sojourns)),
    }


def _calibrated_uas_prob(
    q: np.ndarray,
    mu: np.ndarray,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> np.ndarray:
    logits = gamma * np.log(mu) - alpha * ((q + c) / (mu ** beta))
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)


def _projected_fluid_step(
    q: np.ndarray,
    mu: np.ndarray,
    lam: float,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
    dt: float,
) -> np.ndarray:
    probs = _calibrated_uas_prob(q, mu, alpha=alpha, beta=beta, gamma=gamma, c=c)
    arrivals = lam * probs
    raw_drift = arrivals - mu
    drift = np.where(q > 0.0, raw_drift, np.maximum(raw_drift, 0.0))
    q_next = q + dt * drift
    return np.maximum(q_next, 0.0)


def _direction1_initial_states(num_servers: int, *, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = [np.zeros(num_servers, dtype=np.float64)]
    for total in (5.0, 10.0, 20.0, 40.0, 80.0):
        balanced = np.full(num_servers, total / num_servers, dtype=np.float64)
        states.append(balanced)
        spike = np.zeros(num_servers, dtype=np.float64)
        spike[0] = total
        states.append(spike)
    for _ in range(12):
        states.append(rng.uniform(0.0, 20.0, size=num_servers))
    return states


def _run_direction1_fluid_diagnostic(
    cfg: ExperimentConfig,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
    dt: float,
    horizon: float,
    seed: int,
) -> dict[str, object]:
    mu = np.asarray(cfg.system.service_rates, dtype=np.float64)
    lam = float(cfg.system.arrival_rate)
    steps = int(np.ceil(horizon / dt))
    initial_states = _direction1_initial_states(cfg.system.num_servers, seed=seed)
    terminal_states: list[np.ndarray] = []
    residual_norms: list[float] = []

    for q0 in initial_states:
        q = q0.copy()
        for _ in range(steps):
            q = _projected_fluid_step(
                q,
                mu,
                lam,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                c=c,
                dt=dt,
            )
        terminal_states.append(q)
        q_after = _projected_fluid_step(
            q,
            mu,
            lam,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            c=c,
            dt=dt,
        )
        residual_norms.append(float(np.linalg.norm((q_after - q) / dt, ord=np.inf)))

    terminals = np.stack(terminal_states, axis=0)
    center = np.mean(terminals, axis=0)
    max_radius = float(np.max(np.linalg.norm(terminals - center, axis=1)))
    pairwise_diameter = float(np.max(np.linalg.norm(terminals[:, None, :] - terminals[None, :, :], axis=2)))
    residual_inf = float(np.max(residual_norms))

    return {
        "policy": "Calibrated UAS",
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "c": c,
        "dt": dt,
        "horizon": horizon,
        "num_initial_states": len(initial_states),
        "terminal_center": center.tolist(),
        "terminal_max_radius": max_radius,
        "terminal_pairwise_diameter": pairwise_diameter,
        "max_terminal_residual_inf": residual_inf,
        "supports_single_attractor": bool(pairwise_diameter < 0.5 and residual_inf < 0.1),
    }


def _write_summary(
    run_dir: Path,
    *,
    references: dict[str, dict[str, float]],
    reference_path: Path | None,
    direction1: dict[str, object],
    direction2_rows: list[dict[str, object]],
    direction2_enabled: bool,
    recommendation: str,
) -> None:
    summary_path = artifacts_dir(run_dir) / "direction12_summary.md"
    cal = references.get("Calibrated UAS", {})
    jssq = references.get("JSSQ (Min Sojourn)", {})
    best_d2 = direction2_rows[0] if direction2_rows else None
    lines = [
        "# Direction 1 vs Direction 2",
        "",
        f"- Reference metrics source: `{reference_path}`" if reference_path else "- Reference metrics source: not found",
        f"- Calibrated UAS mean_q_total: `{cal.get('mean_q_total', float('nan')):.6f}`" if cal else "- Calibrated UAS mean_q_total: unavailable",
        f"- JSSQ mean_q_total: `{jssq.get('mean_q_total', float('nan')):.6f}`" if jssq else "- JSSQ mean_q_total: unavailable",
        "",
        "## Direction 1",
        "",
        f"- Policy performance proxy: existing Calibrated UAS value `{cal.get('mean_q_total', float('nan')):.6f}`",
        f"- Fluid single-attractor support: `{direction1['supports_single_attractor']}`",
        f"- Terminal pairwise diameter: `{direction1['terminal_pairwise_diameter']:.6f}`",
        f"- Max terminal residual inf-norm: `{direction1['max_terminal_residual_inf']:.6f}`",
        "",
        "## Direction 2",
        "",
    ]
    if not direction2_enabled:
        lines.append("- Skipped by configuration.")
    else:
        for row in direction2_rows:
            lines.append(
                f"- `{row['label']}`: mean_q_total=`{row['mean_q_total']:.6f}`, "
                f"se=`{row['se_q_total']:.6f}`, mean_sojourn=`{row['mean_sojourn']:.6f}`"
            )
    if direction2_enabled and best_d2 is not None:
        lines.extend(
            [
                "",
                f"- Best Direction 2 candidate: `{best_d2['label']}` with `{best_d2['mean_q_total']:.6f}`",
            ]
        )
    lines.extend(["", "## Recommendation", "", recommendation, ""])
    summary_path.write_text("\n".join(lines), encoding="utf-8")


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig) -> None:
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "policy")
    run_dir, run_id = get_run_config(cfg, "direction12_probe", resolved_raw_cfg)
    run_logger = setup_wandb(
        cfg,
        resolved_raw_cfg,
        default_group="direction12_probe",
        run_id=run_id,
        run_dir=run_dir,
    )

    references, reference_path = _load_reference_metrics()
    log.info("Loaded references from %s", reference_path)

    direction1 = _run_direction1_fluid_diagnostic(
        cfg,
        alpha=float(_select_probe_value(raw_cfg, "direction1.alpha", 20.0)),
        beta=float(_select_probe_value(raw_cfg, "direction1.beta", 0.85)),
        gamma=float(_select_probe_value(raw_cfg, "direction1.gamma", 0.5)),
        c=float(_select_probe_value(raw_cfg, "direction1.c", 0.5)),
        dt=float(_select_probe_value(raw_cfg, "direction1.dt", 0.02)),
        horizon=float(_select_probe_value(raw_cfg, "direction1.horizon", 200.0)),
        seed=int(_select_probe_value(raw_cfg, "direction1.seed", cfg.simulation.seed)),
    )

    (artifacts_dir(run_dir) / "direction1_fluid.json").write_text(
        json.dumps(direction1, indent=2),
        encoding="utf-8",
    )

    search_mode = str(_select_probe_value(raw_cfg, "direction2.search_mode", "quick"))
    top_k = int(_select_probe_value(raw_cfg, "direction2.top_k", 3))
    pilot_replications = int(_select_probe_value(raw_cfg, "direction2.pilot_replications", 4))
    pilot_sim_time = float(_select_probe_value(raw_cfg, "direction2.pilot_sim_time", 3000.0))
    final_replications = int(_select_probe_value(raw_cfg, "direction2.final_replications", 16))
    final_sim_time = float(_select_probe_value(raw_cfg, "direction2.final_sim_time", 8000.0))
    direction2_enabled = (
        bool(_select_probe_value(raw_cfg, "direction2.enabled", True))
        and top_k > 0
        and pilot_replications > 0
        and final_replications > 0
    )

    mu = np.asarray(cfg.system.service_rates, dtype=np.float64)
    pilot_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    if direction2_enabled:
        candidates = _direction2_candidates(search_mode)
        for spec in iter_progress(
            candidates,
            total=len(candidates),
            desc="direction2 pilot",
            unit="candidate",
            leave=False,
        ):
            policy = SparsemaxEnergyRouting(
                mu,
                alpha=spec.alpha,
                beta=spec.beta,
                gamma=spec.gamma,
                c=spec.c,
                label=spec.label,
            )
            metrics = _evaluate_policy(
                policy,
                cfg,
                num_replications=pilot_replications,
                sim_time=pilot_sim_time,
                sample_interval=cfg.simulation.ssa.sample_interval,
                base_seed=cfg.simulation.seed,
                progress_desc=spec.label,
            )
            row: dict[str, object] = {
                "phase": "pilot",
                "label": spec.label,
                "alpha": spec.alpha,
                "beta": spec.beta,
                "gamma": spec.gamma,
                "c": spec.c,
                **metrics,
            }
            pilot_rows.append(row)
            append_metrics_jsonl(row, metrics_path(run_dir, "direction2_pilot.jsonl"))

        pilot_rows.sort(key=lambda item: float(item["mean_q_total"]))
        finalists = pilot_rows[:top_k]
        for rank, finalist in enumerate(finalists, start=1):
            policy = SparsemaxEnergyRouting(
                mu,
                alpha=float(finalist["alpha"]),
                beta=float(finalist["beta"]),
                gamma=float(finalist["gamma"]),
                c=float(finalist["c"]),
                label=str(finalist["label"]),
            )
            metrics = _evaluate_policy(
                policy,
                cfg,
                num_replications=final_replications,
                sim_time=final_sim_time,
                sample_interval=cfg.simulation.ssa.sample_interval,
                base_seed=cfg.simulation.seed + 10_000 * rank,
                progress_desc=str(finalist["label"]),
            )
            row = {
                "phase": "final",
                "rank": rank,
                "label": finalist["label"],
                "alpha": finalist["alpha"],
                "beta": finalist["beta"],
                "gamma": finalist["gamma"],
                "c": finalist["c"],
                **metrics,
            }
            final_rows.append(row)
            append_metrics_jsonl(row, metrics_path(run_dir, "direction2_final.jsonl"))

        final_rows.sort(key=lambda item: float(item["mean_q_total"]))
    else:
        log.info(
            "Direction 2 skipped: enabled=%s, top_k=%d, pilot_replications=%d, final_replications=%d",
            bool(_select_probe_value(raw_cfg, "direction2.enabled", True)),
            top_k,
            pilot_replications,
            final_replications,
        )

    cal_ref = references.get("Calibrated UAS")
    recommendation = "Direction 2 skipped. Evaluate Direction 1 only."
    if direction2_enabled and final_rows and cal_ref:
        best_d2 = final_rows[0]
        cal_mean = float(cal_ref["mean_q_total"])
        cal_se = float(cal_ref.get("se_q_total", 0.0))
        d2_mean = float(best_d2["mean_q_total"])
        d2_se = float(best_d2["se_q_total"])
        material_margin = 2.0 * np.sqrt(cal_se ** 2 + d2_se ** 2)
        if d2_mean + material_margin < cal_mean:
            recommendation = (
                "Choose Direction 2: the best sparsemax Fenchel-Young proxy beat "
                "the saved Calibrated UAS benchmark by more than the combined noise margin."
            )
        elif bool(direction1["supports_single_attractor"]):
            recommendation = (
                "Choose Direction 1: Direction 2 did not beat Calibrated UAS, while "
                "the fluid diagnostic supports pursuing certification of the existing empirical winner."
            )
        else:
            recommendation = (
                "Do not commit yet: Direction 2 failed to beat Calibrated UAS and "
                "Direction 1 did not show clean fluid convergence."
            )

    summary_payload = {
        "reference_path": str(reference_path) if reference_path else None,
        "references": references,
        "direction1": direction1,
        "direction2_finalists": final_rows,
        "recommendation": recommendation,
    }
    (artifacts_dir(run_dir) / "direction12_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    _write_summary(
        run_dir,
        references=references,
        reference_path=reference_path,
        direction1=direction1,
        direction2_rows=final_rows,
        direction2_enabled=direction2_enabled,
        recommendation=recommendation,
    )

    if run_logger:
        if direction2_enabled and final_rows:
            run_logger.log({"direction2/best_mean_q_total": final_rows[0]["mean_q_total"]})
        run_logger.finish()


if __name__ == "__main__":
    main()
