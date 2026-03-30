"""
Self-contained FAST BC + Evaluation comparison for softmax variant experts.

Generates synthetic expert data (random queue states + expert labels) to avoid
running 9 slow SSA simulations in ``collect_robust_expert_data``.

Usage:
    python -m experiments.testing.compare_rl_variants
"""
import logging, time, math
import jax, jax.numpy as jnp, numpy as np
import equinox as eqx, optax
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.config import NeuralConfig
from gibbsq.core.policies import RoutingPolicy
from gibbsq.engines.numpy_engine import simulate
from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.core.features import sojourn_time_features

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

class RawQueueExpert(RoutingPolicy):
    """p_i ∝ exp(-α Q_i)  — Paper formulation."""
    __slots__ = ("_mu", "_alpha")
    def __init__(self, mu, alpha=1.0):
        self._mu = np.asarray(mu, dtype=np.float64)
        self._alpha = float(alpha)
    @property
    def mu(self): return self._mu
    @property
    def alpha(self): return self._alpha
    def __call__(self, Q, rng):
        logits = -self._alpha * Q.astype(np.float64)
        logits -= logits.max()
        w = np.exp(logits); return w / w.sum()

class SojournTimeExpert(RoutingPolicy):
    """p_i ∝ exp(-α (Q_i+1)/μ_i)  — Implementation formulation."""
    __slots__ = ("_mu", "_alpha")
    def __init__(self, mu, alpha=1.0):
        self._mu = np.asarray(mu, dtype=np.float64)
        self._alpha = float(alpha)
    @property
    def mu(self): return self._mu
    @property
    def alpha(self): return self._alpha
    def __call__(self, Q, rng):
        sojourn = (Q.astype(np.float64) + 1.0) / self._mu
        logits = -self._alpha * sojourn
        logits -= logits.max()
        w = np.exp(logits); return w / w.sum()

def fast_bc_train(expert_cls, mu, num_servers, key, num_steps=200):
    """Train a neural policy via BC using SYNTHETIC expert data (no SSA)."""
    rng = np.random.default_rng(42)
    expert = expert_cls(mu, alpha=1.0)

    n_samples = 2000
    states, rhos, probs, mus_arr = [], [], [], []
    for _ in range(n_samples):
        Q = rng.integers(0, 40, size=num_servers).astype(np.float64)
        rho = rng.uniform(0.4, 0.98)
        mu_scale = rng.choice([0.5, 1.0, 2.0])
        scaled_mu = mu * mu_scale
        scaled_expert = expert_cls(scaled_mu, alpha=1.0)
        p = scaled_expert(Q, rng)
        states.append(Q)
        rhos.append(rho)
        probs.append(p)
        mus_arr.append(scaled_mu)

    X = jnp.array(states, dtype=jnp.float32)
    R = jnp.array(rhos, dtype=jnp.float32)
    Y = jnp.array(probs, dtype=jnp.float32)
    MU = jnp.array(mus_arr, dtype=jnp.float32)

    neu_cfg = NeuralConfig(hidden_size=64, preprocessing="log1p", use_rho=False)
    key, actor_key = jax.random.split(key)
    policy_net = NeuralRouter(
        num_servers=num_servers, config=neu_cfg,
        service_rates=jnp.array(mu), key=actor_key,
    )

    optimizer = optax.adamw(3e-3, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(policy_net, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(model, x, r, target_probs, mu_batch):
        s_feat = jax.vmap(sojourn_time_features)(x, mu_batch)
        logits = jax.vmap(model)(s_feat, r)
        soft_labels = target_probs * 0.9 + 0.1 / num_servers
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce_loss = -jnp.mean(jnp.sum(soft_labels * log_probs, axis=-1))
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(target_probs, axis=-1))
        return ce_loss, acc

    @eqx.filter_jit
    def step(model, opt_state, x, r, target_probs, mu_batch):
        (loss, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, x, r, target_probs, mu_batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, acc

    t0 = time.time()
    for i in range(num_steps + 1):
        policy_net, opt_state, loss, acc = step(policy_net, opt_state, X, R, Y, MU)
        if i % 50 == 0:
            log.info(f"    Step {i:4d} | Loss: {loss:.4f} | Acc: {acc:.2%}")
    log.info(f"    BC completed in {time.time()-t0:.1f}s")
    return policy_net

class NeuralPolicyWrapper:
    """Pure NumPy forward pass using extracted weights. No JAX overhead per event."""
    def __init__(self, net, mu):
        self.params = net.get_numpy_params()
        self.mu = np.asarray(mu, dtype=np.float64)
        self.preprocessing = net.config.preprocessing
        self.capacity_bound = net.config.capacity_bound
        self.service_rates = np.array(net.service_rates, dtype=np.float64)

    def __call__(self, Q, rng):
        Q = np.asarray(Q, dtype=np.float64)
        # Apply sojourn-time features to match training (line 95 uses sojourn_time_features)
        sojourn = (Q + 1.0) / self.mu
        if self.preprocessing == "log1p":
            x = np.log1p(sojourn)
        elif self.preprocessing == "linear_min_max":
            x = sojourn / self.capacity_bound
        else:
            x = sojourn.copy()
        # Append normalized service rates (matches _single_forward)
        mu_sum = np.sum(self.service_rates)
        mu_norm = self.service_rates / max(mu_sum, 1.0)
        x = np.concatenate([x, mu_norm])
        for i, (w, b) in enumerate(self.params):
            x = x @ w.T  # Linear: weight shape is (out, in)
            if b is not None:
                x = x + b
            if i < len(self.params) - 1:  # ReLU on all but last
                x = np.maximum(x, 0.0)
        x = x - x.max()
        probs = np.exp(x)
        probs /= probs.sum()
        return probs

def eval_policy(name, policy, mu, num_servers, sim_time=500.0, n_reps=3):
    """Evaluate a policy via short SSA runs."""
    rng = np.random.default_rng(123)
    rho_eval = 0.90
    lam_eval = rho_eval * np.sum(mu)
    q_totals = []
    for _ in range(n_reps):
        res = simulate(
            num_servers=num_servers, arrival_rate=lam_eval,
            service_rates=mu, policy=policy,
            sim_time=sim_time, sample_interval=1.0, rng=rng,
        )
        q_avg = time_averaged_queue_lengths(res, 0.2)
        q_totals.append(q_avg.sum())
    mean_q = float(np.mean(q_totals))
    std_q = float(np.std(q_totals))
    log.info(f"  {name:35s} → E[Q] = {mean_q:.2f} ± {std_q:.2f}")
    return mean_q

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  SOFTMAX VARIANT COMPARISON (FAST)")
    log.info("=" * 60)

    N = 5
    mu = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    key = jax.random.PRNGKey(42)

    log.info("\n--- Analytical Expert Baselines ---")
    q_raw_an = eval_policy("Raw Queue Expert", RawQueueExpert(mu), mu, N)
    q_soj_an = eval_policy("Sojourn Time Expert", SojournTimeExpert(mu), mu, N)

    log.info("\n--- Neural BC Pipelines ---")
    log.info("  Training BC from Raw Queue Expert...")
    k1, k2, key = jax.random.split(key, 3)
    raw_net = fast_bc_train(RawQueueExpert, mu, N, k1, num_steps=200)
    log.info("  Training BC from Sojourn Time Expert...")
    soj_net = fast_bc_train(SojournTimeExpert, mu, N, k2, num_steps=200)

    log.info("\n--- Neural Policy Evaluation ---")
    q_raw_nn = eval_policy("Neural (BC from Raw Queue)", NeuralPolicyWrapper(raw_net, mu), mu, N)
    q_soj_nn = eval_policy("Neural (BC from Sojourn Time)", NeuralPolicyWrapper(soj_net, mu), mu, N)

    log.info(f"\n{'='*60}")
    log.info(f"  FINAL SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Analytical Raw Queue:      E[Q] = {q_raw_an:.2f}")
    log.info(f"  Analytical Sojourn Time:   E[Q] = {q_soj_an:.2f}")
    log.info(f"  Neural (BC→Raw Queue):     E[Q] = {q_raw_nn:.2f}")
    log.info(f"  Neural (BC→Sojourn Time):  E[Q] = {q_soj_nn:.2f}")
    winner_an = "Raw Queue" if q_raw_an < q_soj_an else "Sojourn Time"
    winner_nn = "Raw Queue" if q_raw_nn < q_soj_nn else "Sojourn Time"
    log.info(f"\n  >>> Analytical Winner: {winner_an}")
    log.info(f"  >>> Neural Winner:     {winner_nn}")
    log.info(f"{'='*60}")
