"""
Unbiased DGA vs SSA Bias Verification
-------------------------------------
Empirically compares the DGA surrogate's Expected Queue Length (E[Q])
against the true Gillespie SSA E[Q] for identical routing policies.
"""

import jax
import jax.numpy as jnp
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig
import numpy as np

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.differentiable_engine import simulate_dga_jax, default_policy
from gibbsq.engines.jax_engine import run_replications_jax
from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.engines.numpy_engine import SimResult

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

class BiasVerification:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_servers = cfg.system.num_servers
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=jnp.float32)
        self.dga_sim_steps = cfg.simulation.dga.sim_steps
        self.dga_temperature = float(cfg.simulation.dga.temperature)
        self.num_reps = int(cfg.simulation.num_replications)
        
        self.vmap_dga = jax.jit(
            jax.vmap(simulate_dga_jax, in_axes=(None, None, None, None, None, 0, None, None)),
            static_argnums=(0, 4, 7)
        )

    def execute(self, key):
        k1, _ = jax.random.split(key, 2)
        
        total_cap = float(jnp.sum(self.service_rates))
        rhos = [0.5, 0.7, 0.9, 0.95, 0.99]
        
        log.info("=" * 60)
        log.info(f"  DGA vs SSA BIAS VERIFICATION (N={self.num_servers})")
        log.info(f"  Policy: GibbsQ (alpha={self.cfg.system.alpha})")
        log.info("=" * 60)
        
        for rho in rhos:
            lam = float(rho * total_cap)
            log.info(f"\n--- Load Factor rho = {rho:.2f} (Arrival = {lam:.2f}) ---")
            
            # --- DGA Evaluation ---
            dga_keys = jax.random.split(k1, self.num_reps)
            
            # SG-6 FIX: sim_steps is a static JAX arg. Scaling it by rho 
            # triggers a recompile for every load factor, potentially 
            # causing CPU timeouts or GPU OOM at high rho.
            rho_adjusted_steps = self.dga_sim_steps
            
            # SSA sim_time expansion must remain rho-proportional (independent of DGA steps)
            _base_rho = 0.8
            _rho_factor = max(1.0, (1.0 - _base_rho) / max(1.0 - float(rho), 1e-6))



            dga_loss_arr = self.vmap_dga(
                self.num_servers, lam, self.service_rates, 
                jnp.float32(self.cfg.system.alpha), rho_adjusted_steps, 
                dga_keys, self.dga_temperature, default_policy
            )
            dga_eq = float(jnp.mean(dga_loss_arr))
            dga_se = float(jnp.std(dga_loss_arr)) / np.sqrt(self.num_reps)
            
            # --- SSA Evaluation ---
            _sc = self.cfg.simulation
            _rho_sim_time = min(_sc.ssa.sim_time * _rho_factor, _sc.ssa.sim_time * 400.0)
            _max_samples = int(_rho_sim_time / _sc.ssa.sample_interval) + 1
            _mu_np = np.array(self.service_rates, dtype=np.float64)
            
            times_g, states_g, (arrs_g, deps_g) = run_replications_jax(
                num_replications=self.num_reps,
                num_servers=self.num_servers,
                arrival_rate=lam,
                service_rates=jnp.array(_mu_np),
                alpha=float(self.cfg.system.alpha),
                sim_time=_rho_sim_time,
                sample_interval=_sc.ssa.sample_interval,
                base_seed=_sc.seed,
                max_samples=_max_samples,
                policy_type=3, # Softmax
            )
            
            ssa_list = []
            for r in range(self.num_reps):
                res = SimResult(
                    times=np.array(times_g[r]),
                    states=np.array(states_g[r]),
                    arrival_count=int(arrs_g[r]),
                    departure_count=int(deps_g[r]),
                    final_time=float(times_g[r][-1]),
                    num_servers=self.num_servers,
                )
                ssa_list.append(float(time_averaged_queue_lengths(res, _sc.burn_in_fraction).sum()))
            ssa_data = np.array(ssa_list)
            ssa_eq = float(np.mean(ssa_data))
            ssa_se = float(np.std(ssa_data)) / np.sqrt(self.num_reps)
            
            diff = ssa_eq - dga_eq
            diff_pct = (diff / ssa_eq) * 100 if ssa_eq > 0 else 0.0
            
            log.info(f"DGA E[Q]: {dga_eq:7.3f} +/- {dga_se:5.3f}")
            log.info(f"SSA E[Q]: {ssa_eq:7.3f} +/- {ssa_se:5.3f}")
            log.info(f"Gap:      {diff:7.3f} ({diff_pct:+.2f}%)")
            
            if abs(diff_pct) > 5.0:
                log.warning(f"  [!] SIGNIFICANT BIAS DETECTED (>5% deviation)")

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)
    
    test = BiasVerification(cfg)
    test.execute(jax.random.PRNGKey(cfg.simulation.seed))

if __name__ == "__main__":
    main()
