"""
N-GibbsQ evaluation: neural vs analytical parity.

Compares trained NeuralRouter against scalar-alpha GibbsQ
routing policy in a heterogeneous server environment.

SG-C FIX: Both sides now measured on the true Gillespie SSA (not DGA surrogate).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig
from jaxtyping import Array, Float, PRNGKeyArray

import numpy as np
import functools

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.jax_engine import run_replications_jax
from gibbsq.engines.numpy_engine import simulate, SimResult
from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def _resolve_model_pointer(project_root: Path, output_root: Path) -> Path:
    """Resolve model weights pointer with REINFORCE-first fallback order."""
    candidates = [
        output_root / "latest_domain_randomized_weights.txt",
        output_root / "latest_reinforce_weights.txt",
        output_root / "latest_weights.txt",  # legacy DGA pointer
    ]
    for ptr in candidates:
        if not ptr.exists():
            continue
        raw = Path(ptr.read_text(encoding='utf-8').strip())
        model_path = raw if raw.is_absolute() else (project_root / raw)
        if model_path.exists():
            if ptr.name == "latest_weights.txt":
                log.warning("Using legacy pointer latest_weights.txt; prefer REINFORCE pointers.")
            return model_path
    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "No valid model pointer found. Tried:\n"
        f"{tried}\n"
        "Run Track 1/3 training (reinforce_train or dr_train), or legacy train for latest_weights.txt."
    )


class _NeuralSSAPolicy:
    """
    Bridges NeuralRouter → NumPy SSA engine for true-CTMC evaluation.
    Identical in logic to the adapter in policy_comparison.py (SG#1 fix block).
    """
    def __init__(self, model: "NeuralRouter") -> None:
        import jax as _jax
        self._model = model

        @eqx.filter_jit
        def _forward(m, x):
            return _jax.nn.softmax(m(x))
        self._forward = _forward

        @functools.lru_cache(maxsize=131072)
        def _get_probs(q_tuple):
            Q_jax = jnp.array(q_tuple, dtype=jnp.float32)
            probs = self._forward(self._model, Q_jax)
            probs_np = np.array(probs, dtype=np.float64)
            return probs_np / probs_np.sum()
        self._get_probs = _get_probs

    def __call__(self, Q: "np.ndarray", rng: "np.random.Generator") -> "np.ndarray":
        return self._get_probs(tuple(Q))


def evaluate_model(model: NeuralRouter, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
    """Retained for __init__ type contract; not called in the patched execute()."""
    return model(Q)

class NeuralTuringTest:
    """
    Verification suite comparing NeuralRouter to the analytical baseline.
    """
    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.num_servers = cfg.system.num_servers
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=jnp.float32)
        self.arrival_rate = float(cfg.system.arrival_rate)
        self.temperature = float(cfg.simulation.dga.temperature)
        self.sim_steps = cfg.simulation.dga.sim_steps

        # We run multiple replications to get statistically stable results
        self.num_reps = int(cfg.simulation.num_replications)

    def execute(self, key: PRNGKeyArray):
        """Executes the parity test on the true Gillespie SSA (not DGA surrogate)."""
        _k_unused, k2 = jax.random.split(key, 2)
        
        log.info(f"Environment: N={self.num_servers}, Load={self.arrival_rate}")

        _sc  = self.cfg.simulation
        _ssa = _sc.ssa
        _max_samples = int(_ssa.sim_time / _ssa.sample_interval) + 1
        _mu_np = np.array(self.service_rates, dtype=np.float64)

        # SG#3 FIX: Search for the SSA-optimal alpha (stability_sweep result).
        # This takes precedence over DGA-optimal alpha as it results in a stronger baseline.
        import json
        ssa_alpha = None
        _project_root = Path(__file__).resolve().parents[2]
        _sweep_root = _project_root / "outputs"
        _sweep_candidates = sorted(_sweep_root.glob("**/stability_sweep/*/metrics.jsonl"))
        
        target_rho = self.arrival_rate / float(np.sum(_mu_np))
        
        if _sweep_candidates:
            try:
                best_q = float('inf')
                best_alpha = None
                
                with open(_sweep_candidates[-1], "r") as f:
                    for line in f:
                        data = json.loads(line)
                        # Find alpha that minimizes mean_q_total for rho closest to current
                        if abs(data["rho"] - target_rho) < 0.05:
                            if data["mean_q_total"] < best_q:
                                best_q = data["mean_q_total"]
                                best_alpha = data["alpha"]
                
                if best_alpha is not None:
                    ssa_alpha = best_alpha
                    log.info(f"[SG#3] Found SSA-optimal alpha={ssa_alpha:.4f} for rho={target_rho:.2f} in {_sweep_candidates[-1]}")
            except Exception as e:
                log.warning(f"[SG#3] Failed to parse stability sweep results: {e}")

        # Fallback to SG#8 FIX: gradient-optimal alpha from train_dga.py
        _alpha_candidates = sorted(_sweep_root.glob("**/dga_training/*/optimal_alpha.json"))
        
        if ssa_alpha is not None:
            optimal_alpha = ssa_alpha
            self._alpha_source = "SSA-optimised (from stability_sweep)"
        elif _alpha_candidates:
            _alpha_data = json.loads(_alpha_candidates[-1].read_text(encoding="utf-8"))
            optimal_alpha = float(_alpha_data["alpha"])
            log.info(f"[SG#8] Using persisted DGA-optimal alpha={optimal_alpha:.4f} from {_alpha_candidates[-1]}")
            log.warning(
                f"[SG#3] CAVEAT: alpha={optimal_alpha:.4f} was optimised on the DGA "
                f"surrogate. SSA-optimal alpha is likely ~10.0 (from stability_sweep). "
                f"Parity against this baseline may be misleading."
            )
            self._alpha_source = f"DGA-optimised (from {_alpha_candidates[-1].name})"
        else:
            optimal_alpha = float(self.cfg.system.alpha)
            log.warning(
                f"[SG#8] No persisted optimal_alpha.json found. "
                f"Falling back to cfg.system.alpha={optimal_alpha}. "
                f"Run train_dga first for a valid GibbsQ baseline."
            )
        times_g, states_g, (arrs_g, deps_g) = run_replications_jax(
            num_replications=self.num_reps,
            num_servers=self.num_servers,
            arrival_rate=self.arrival_rate,
            service_rates=jnp.array(_mu_np),
            alpha=optimal_alpha,
            sim_time=_ssa.sim_time,
            sample_interval=_ssa.sample_interval,
            base_seed=_sc.seed,
            max_samples=_max_samples,
            policy_type=3,   # softmax
        )
        g_means = []
        for _r in range(self.num_reps):
            _res = SimResult(
                times=np.array(times_g[_r]),
                states=np.array(states_g[_r]),
                arrival_count=int(arrs_g[_r]),
                departure_count=int(deps_g[_r]),
                final_time=float(times_g[_r][-1]),
                num_servers=self.num_servers,
            )
            g_means.append(float(time_averaged_queue_lengths(_res, _sc.burn_in_fraction).sum()))
        mean_gibbsq_loss = float(np.mean(g_means))
        
        # --- 2. The Challenger: N-GibbsQ (Neural Network) ---
        log.info("\n[Loading Challenger: N-GibbsQ Neural Router]")
        
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
        output_root = self.run_dir.parent.parent
        model_path = _resolve_model_pointer(_PROJECT_ROOT, output_root)

        # Re-initialize skeleton and load weights securely using the validated NeuralConfig
        skeleton = NeuralRouter(num_servers=self.num_servers, config=self.cfg.neural, key=k2)
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
        
        # SG#16 Fix: Validate that the loaded model matches the current config
        if model.layers[0].weight.shape[1] != self.num_servers:
            log.error(f"Model shape mismatch! Loaded model expects N={model.layers[0].weight.shape[1]}, but eval config requires N={self.num_servers}.")
            return

        # --- 3. NeuralRouter on the true Gillespie SSA via NumPy engine ---
        _neural_policy = _NeuralSSAPolicy(model)
        _np_max_events = int(
            (self.arrival_rate + float(_mu_np.sum())) * _ssa.sim_time * 1.5
        ) + 1000
        n_means = []
        for _rep in range(self.num_reps):
            _rng = np.random.default_rng(_sc.seed + _rep)
            _res = simulate(
                num_servers=self.num_servers,
                arrival_rate=self.arrival_rate,
                service_rates=_mu_np,
                policy=_neural_policy,
                sim_time=_ssa.sim_time,
                sample_interval=_ssa.sample_interval,
                rng=_rng,
                max_events=_np_max_events,
            )
            n_means.append(float(time_averaged_queue_lengths(_res, _sc.burn_in_fraction).sum()))
        mean_neural_loss = float(np.mean(n_means))
        
        self._report_results(mean_gibbsq_loss, mean_neural_loss)

    def _report_results(self, mean_gibbsq_loss: float, mean_neural_loss: float):
        """Calculates parity metrics and logs the final showdown string."""
        log.info("\n" + "=" * 60)
        log.info("  PARITY RESULTS")
        log.info("=" * 60)
        log.info(f"GibbsQ (Scalar Math) Expected Queue: {mean_gibbsq_loss:.4f}")
        log.info(f"N-GibbsQ (Neural Net) Expected Queue:  {mean_neural_loss:.4f}")
        
        diff = mean_neural_loss - mean_gibbsq_loss
        perc = (diff / mean_gibbsq_loss) * 100 if mean_gibbsq_loss > 0 else 0
        
        status = ""
        if diff <= 0:
            log.info("\n[+] OUTCOME: Neural router matched or exceeded analytical baseline.")
            log.info("    Performance gap: <= 0%.")
            status = "MATCHED"
        elif perc < self.cfg.verification.parity_threshold_percent:
            log.info(f"\n[+] OUTCOME: Neural router within {perc:.1f}% of analytical baseline.")
            log.info(f"    Neural router converged near analytical optimum (threshold={self.cfg.verification.parity_threshold_percent}%).")
            status = "SUCCESS"
        else:
            log.warning(f"\n[-] OUTCOME: The Neural Network failed to match GibbsQ by {perc:.1f}%.")
            status = "FAILED"
            
        # Log to WandB
        if self.run_logger:
            self.run_logger.log({
                "gibbsq_loss": mean_gibbsq_loss,
                "n_gibbsq_loss": mean_neural_loss,
                "parity_diff_percentage": perc
            })

        summary_path = self.run_dir / "parity_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Status: {status}\n")
            f.write(f"GibbsQ E[Q]: {mean_gibbsq_loss:.4f}\n")
            f.write(f"N-GibbsQ E[Q]: {mean_neural_loss:.4f}\n")
            f.write(f"Performance Gap: {perc:.2f}%\n")
            f.write(f"GibbsQ alpha source: {getattr(self, '_alpha_source', 'unknown')}\n")
            f.write(
                f"NOTE: GibbsQ baseline uses DGA-optimised alpha, not SSA-optimal alpha.\n"
                f"For a valid parity claim, compare against SSA-optimal GibbsQ "
                f"(run stability_sweep, identify alpha that minimises E[Q], rerun parity).\n"
            )

        append_metrics_jsonl({
            "gibbsq_loss": mean_gibbsq_loss,
            "n_gibbsq_loss": mean_neural_loss,
            "parity_diff_percentage": perc,
            "status": status
        }, self.run_dir / "metrics.jsonl")


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "neural_parity", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="n_gibbsq_parity", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase 4: N-GibbsQ Parity Evaluation")
    log.info("=" * 60)
    
    test_suite = NeuralTuringTest(cfg, run_dir, run_logger)
    
    seed_key = jax.random.PRNGKey(cfg.simulation.seed)
    test_suite.execute(seed_key)

if __name__ == "__main__":
    main()
