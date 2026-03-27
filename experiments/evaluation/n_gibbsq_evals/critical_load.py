"""
N-GibbsQ critical load test.

Tests N-GibbsQ as ρ → 1, where queueing systems become unstable.

Compares the neural router's expected queue length against GibbsQ
at load factors approaching the critical boundary.

SG-D FIX: Both sides now measured on the true Gillespie SSA (not DGA surrogate).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig
from jaxtyping import Array, Float, PRNGKeyArray
import matplotlib.pyplot as plt
import numpy as np
import functools

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.jax_engine import run_replications_jax
from gibbsq.engines.numpy_engine import simulate, SimResult
from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.model_io import NeuralSSAPolicy, resolve_model_pointer
from gibbsq.engines.jax_ssa import compute_poisson_max_steps


# _NeuralSSAPolicy moved to gibbsq.utils.model_io.NeuralSSAPolicy


logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


# _resolve_model_pointer moved to gibbsq.utils.model_io.resolve_model_pointer

def evaluate_model(model: NeuralRouter, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
    """Pure functional bridge."""
    return model(Q)

class CriticalLoadTest:
    """Evaluates N-GibbsQ at high load factors."""
    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.num_servers = cfg.system.num_servers
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=jnp.float32)
        self.sim_steps = cfg.simulation.dga.sim_steps
        self.temperature = float(cfg.simulation.dga.temperature)
        self.ssa_sim_time = cfg.simulation.ssa.sim_time
        self.ssa_sample_interval = cfg.simulation.ssa.sample_interval
        
        # Symmetrical and Extreme ρ range
        self.rho_vals = list(cfg.generalization.rho_boundary_vals)

    def execute(self, key: PRNGKeyArray):
        """Sweeps ρ and measures stability."""
        k_load, k_sweep = jax.random.split(key)
        
        # 1. Load trained model
        _PROJECT_ROOT = Path(__file__).resolve().parents[3]
        output_root = self.run_dir.parent.parent
        model_path = resolve_model_pointer(_PROJECT_ROOT, output_root)
        skeleton = NeuralRouter(num_servers=self.num_servers, config=self.cfg.neural, service_rates=self.service_rates, key=k_load)
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
        
        # SG#16 Fix: Validate that the loaded model matches the current config
        from gibbsq.utils.model_io import validate_neural_model_shape
        try:
            validate_neural_model_shape(model, self.cfg.neural, self.num_servers)
        except ValueError as e:
            log.error(f"Model shape mismatch! {e}")
            return
        
        total_capacity = jnp.sum(self.service_rates)
        
        neural_results = []
        gibbs_results = []
        
        log.info(f"System Capacity: {total_capacity:.2f}")
        log.info(f"Targeting Load Boundary: {self.rho_vals}")
        
        num_reps = int(self.cfg.simulation.num_replications)
        
        for idx, rho in enumerate(self.rho_vals):
            arrival_rate = rho * total_capacity
            
            log.info(f"Evaluating Boundary rho={rho:.3f} (Arrival={arrival_rate:.3f})...")
            
            # SG#9 FIX: Linear mixing-time scaling is correct for fixed N
            # (spectral gap ∝ (1-ρ) for fixed-N birth-death chains; see
            # Goldberg & Li 2022, Oper. Res. bounds that scale as 1/(1-ρ)).
            # The Halfin-Whitt quadratic scaling O(1/(1-ρ)²) applies only to
            # the many-server asymptotic regime where both N and λ grow.
            _base_rho = self.cfg.stress.critical_load_base_rho
            # SG#5: Heavy-traffic mixing time scales as O(1/(1-rho)) per Goldberg & Li 2022
            _rho_factor = max(1.0, ((1.0 - _base_rho) / max(1.0 - float(rho), 1e-6)))
            _uncapped_sim_time = self.ssa_sim_time * _rho_factor
            _cap = float(self.cfg.stress.critical_load_max_sim_time)
            _rho_sim_time = min(_uncapped_sim_time, _cap)
            if _uncapped_sim_time > _cap:
                log.warning(
                    f"  [!] rho={rho:.4f}: mixing-time formula predicts "
                    f"{_uncapped_sim_time:,.0f}s sim_time but cap is {_cap:,.0f}s. "
                    f"E[Q] near criticality may be underestimated (stationarity "
                    f"not guaranteed). Interpret rho>{rho:.3f} results cautiously."
                )
            _max_s = int(_rho_sim_time / self.ssa_sample_interval) + 1
            _mu_np = np.array(self.service_rates, dtype=np.float64)
            _rho_seed = self.cfg.simulation.seed + idx * 1000

            # GibbsQ on true SSA
            _pmap = {"uniform": 0, "proportional": 1, "jsq": 2, "softmax": 3, "power_of_d": 4, "uas": 6}
            times_g, states_g, (arrs_g, deps_g) = run_replications_jax(
                num_replications=num_reps, num_servers=self.num_servers,
                arrival_rate=float(arrival_rate), service_rates=jnp.array(_mu_np),
                alpha=float(self.cfg.system.alpha), sim_time=_rho_sim_time,
                sample_interval=self.ssa_sample_interval, base_seed=_rho_seed,
                max_samples=_max_s, policy_type=_pmap.get(self.cfg.policy.name, 3),
            )
            g_vals = []
            for _r in range(num_reps):
                _np_times = np.array(times_g[_r])
                _np_states = np.array(states_g[_r])
                _valid_mask = _np_times > 0
                _valid_mask[0] = True
                _vl = int(np.sum(_valid_mask))
                _np_times = _np_times[:_vl]
                _np_states = _np_states[:_vl]

                _res = SimResult(
                    times=_np_times, states=_np_states,
                    arrival_count=int(arrs_g[_r]), departure_count=int(deps_g[_r]),
                    final_time=float(times_g[_r][-1]), num_servers=self.num_servers,
                )
                g_vals.append(float(time_averaged_queue_lengths(
                    _res, self.cfg.simulation.burn_in_fraction).sum()))
            g_loss = float(np.mean(g_vals))

            # Neural on true SSA
            _neural_ssa = NeuralSSAPolicy(model, mu=_mu_np, rho=rho)
            _max_ev = compute_poisson_max_steps(float(arrival_rate), _mu_np, _rho_sim_time)
            n_vals = []
            for _rep in range(num_reps):
                _rng = np.random.default_rng(_rho_seed + _rep)
                _res_n = simulate(
                    num_servers=self.num_servers, arrival_rate=float(arrival_rate),
                    service_rates=_mu_np, policy=_neural_ssa,
                    sim_time=_rho_sim_time, sample_interval=self.ssa_sample_interval,
                    rng=_rng, max_events=_max_ev,
                )
                n_vals.append(float(time_averaged_queue_lengths(
                    _res_n, self.cfg.simulation.burn_in_fraction).sum()))
            n_loss = float(np.mean(n_vals))

            neural_results.append(n_loss)
            gibbs_results.append(g_loss)
            
            log.info(f"   => N-GibbsQ E[Q]: {n_loss:.2f} | GibbsQ E[Q]: {g_loss:.2f}")

            append_metrics_jsonl({
                "rho": float(rho),
                "neural_eq": n_loss,
                "gibbs_eq": g_loss
            }, self.run_dir / "metrics.jsonl")

        self._plot(self.rho_vals, neural_results, gibbs_results)

    def _plot(self, rho_vals, neural_r, gibbs_r):
        """Generates the stability breakdown plot using chart-type-aware styling."""
        from gibbsq.analysis.plotting import plot_critical_load

        plot_path = self.run_dir / "critical_load_curve"
        fig = plot_critical_load(
            rho_values=np.array(rho_vals),
            neural_eq=np.array(neural_r),
            gibbs_eq=np.array(gibbs_r),
            save_path=plot_path,
            theme="publication",
            formats=["png", "pdf"],
        )
        plt.close(fig)

        log.info(f"Critical load test complete. Curve saved to {plot_path}.png, {plot_path}.pdf")

        if self.run_logger:
            self.run_logger.log({
                "critical_load/rho": rho_vals,
                "critical_load/neural_eq": neural_r,
                "critical_load/gibbs_eq": gibbs_r
            })
            try:
                import wandb
                self.run_logger.log({"critical_load_curve": wandb.Image(str(self.run_dir / "critical_load_curve.png"))})
            except Exception:
                pass

@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "critical_load", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="n_gibbsq_verification", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase VIII: The Critical Stability Boundary")
    log.info("=" * 60)
    
    test = CriticalLoadTest(cfg, run_dir, run_logger)
    test.execute(jax.random.PRNGKey(cfg.simulation.seed))

if __name__ == "__main__":
    main()
