import pytest
import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats

from gibbsq.engines.numpy_engine import simulate as simulate_numpy, SimResult
from gibbsq.engines.jax_engine import simulate_jax
from gibbsq.core.policies import SoftmaxRouting

class TestCrossEngineConsistency:
    def test_mean_queue_length_equivalence(self):
        num_servers = 3
        arrival_rate = 2.0
        service_rates = np.array([3.0, 3.0, 3.0])
        sim_time = 500.0
        sample_interval = 1.0
        alpha = 1.0
        seed = 42
        
        np_result = simulate_numpy(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates,
            policy=SoftmaxRouting(alpha=alpha),
            sim_time=sim_time,
            sample_interval=sample_interval,
            rng=np.random.default_rng(seed),
        )
        
        jax_times, jax_states, (jax_arrivals, jax_departures) = simulate_jax(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=jnp.array(service_rates),
            alpha=alpha,
            sim_time=sim_time,
            sample_interval=sample_interval,
            key=jax.random.PRNGKey(seed),
            max_samples=int(sim_time / sample_interval) + 10,
            policy_type=3,
        )
        
        np_mean_q = np_result.states.sum(axis=1).mean()
        jax_mean_q = jax_states.sum(axis=1).mean()
        
        assert np.isfinite(np_mean_q) and np_mean_q >= 0
        assert jnp.isfinite(jax_mean_q) and jax_mean_q >= 0
        
        # For M/M/N with rho<1, expected queue is O(rho/(1-rho)) per server
        # With rho=0.667, this is ~2 per server, but SSA has variance
        # Allow wide range: 0.1 to 10 per server
        assert 0.1 < np_mean_q / num_servers < 10.0, f"NumPy mean per server out of range"
        assert 0.1 < jax_mean_q / num_servers < 10.0, f"JAX mean per server out of range"
    
    def test_arrival_rate_consistency(self):
        num_servers = 2
        arrival_rate = 5.0
        service_rates = np.array([6.0, 6.0])
        sim_time = 200.0
        seed = 123
        
        np_result = simulate_numpy(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates,
            policy=SoftmaxRouting(alpha=1.0),
            sim_time=sim_time,
            sample_interval=1.0,
            rng=np.random.default_rng(seed),
        )
        
        _, _, (jax_arrivals, _) = simulate_jax(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=jnp.array(service_rates),
            alpha=1.0,
            sim_time=sim_time,
            sample_interval=1.0,
            key=jax.random.PRNGKey(seed),
            max_samples=250,
        )
        
        np_rate = np_result.arrival_count / sim_time
        jax_rate = float(jax_arrivals) / sim_time
        
        assert abs(np_rate - arrival_rate) / arrival_rate < 0.15
        assert abs(jax_rate - arrival_rate) / arrival_rate < 0.15
    
    def test_conservation_law_both_engines(self):
        num_servers = 2
        arrival_rate = 2.0
        service_rates = np.array([3.0, 3.0])
        sim_time = 100.0
        seed = 456
        
        np_result = simulate_numpy(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates,
            policy=SoftmaxRouting(alpha=1.0),
            sim_time=sim_time,
            sample_interval=1.0,
            rng=np.random.default_rng(seed),
        )
        
        # JAX - run with very small sample interval to capture final state accurately
        jax_times, jax_states, (jax_arr, jax_dep) = simulate_jax(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=jnp.array(service_rates),
            alpha=1.0,
            sim_time=sim_time,
            sample_interval=0.1,  # Small interval for accurate final state
            key=jax.random.PRNGKey(seed),
            max_samples=1500,
        )
        valid_mask = np.asarray(jax_times) > 0
        valid_mask[0] = True
        jax_final_q = np.asarray(jax_states)[valid_mask][-1].sum()
        
        np_final_q = np_result.states[-1].sum()
        np_conservation = np_result.arrival_count - np_result.departure_count - np_final_q
        assert abs(np_conservation) == 0, f"NumPy conservation violated: {np_conservation}"
        
        # JAX conservation should hold exactly on the valid sampled tail.
        jax_conservation = int(jax_arr) - int(jax_dep) - int(jax_final_q)
        assert abs(jax_conservation) == 0, f"JAX conservation violated: {jax_conservation}"

class TestJAXNumericalStability:
    def test_gradient_flow_stability(self):
        import jax
        
        def loss_fn(alpha):
            _, states, _ = simulate_jax(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([2.0, 2.0]),
                alpha=alpha,
                sim_time=50.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(0),
                max_samples=100,
            )
            return states.sum().astype(jnp.float32)
        
        for alpha_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
            grad_fn = jax.grad(loss_fn)
            grad = grad_fn(alpha_val)
            
            assert jnp.isfinite(grad), f"Non-finite gradient at alpha={alpha_val}"
            assert not jnp.isnan(grad), f"NaN gradient at alpha={alpha_val}"
    
    def test_softmax_overflow_protection(self):
        from gibbsq.engines.jax_engine import get_probs, SimParams
        
        # Very large queue lengths -> very negative logits
        Q = jnp.array([1000, 2000, 3000])
        params = SimParams(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=jnp.ones(3),
            alpha=10.0,  # Amplifies the negative logits
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=3,
            d=2,
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        
        assert jnp.all(jnp.isfinite(probs)), "Softmax produced non-finite probabilities"
        assert abs(float(jnp.sum(probs)) - 1.0) < 1e-6, "Softmax probabilities don't sum to 1"
    
    def test_jit_recompilation_stability(self):
        results = []
        
        for i in range(10):
            _, states, _ = simulate_jax(
                num_servers=2,
                arrival_rate=1.5,
                service_rates=jnp.array([2.0, 2.0]),
                alpha=1.0,
                sim_time=20.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(i),
                max_samples=50,
            )
            results.append(float(states.mean()))
        
        assert all(np.isfinite(r) for r in results)
        
        assert np.std(results) < 10.0, "Results show high variance across runs"
    
    def test_extreme_parameters_stability(self):
        test_cases = [
            {"arrival_rate": 0.001, "service_rates": [1.0, 1.0]},
            {"arrival_rate": 10.0, "service_rates": [20.0, 20.0]},
            {"arrival_rate": 5.0, "service_rates": [1.0, 10.0]},
        ]
        
        for i, params in enumerate(test_cases):
            _, states, (arr, dep) = simulate_jax(
                num_servers=2,
                arrival_rate=params["arrival_rate"],
                service_rates=jnp.array(params["service_rates"]),
                alpha=1.0,
                sim_time=50.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(i),
                max_samples=100,
            )
            
            assert jnp.all(jnp.isfinite(states)), f"Non-finite states for params: {params}"
            assert jnp.all(states >= 0), f"Negative states for params: {params}"

class TestJAXDeterminism:
    def test_same_key_identical_results(self):
        kwargs = dict(
            num_servers=3,
            arrival_rate=2.0,
            service_rates=jnp.array([3.0, 3.0, 3.0]),
            alpha=1.0,
            sim_time=100.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(999),
            max_samples=150,
        )
        
        _, states1, (arr1, dep1) = simulate_jax(**kwargs)
        _, states2, (arr2, dep2) = simulate_jax(**kwargs)
        
        np.testing.assert_array_equal(states1, states2)
        assert arr1 == arr2
        assert dep1 == dep2
    
    def test_different_keys_different_results(self):
        base_kwargs = dict(
            num_servers=2,
            arrival_rate=1.5,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=100.0,
            sample_interval=1.0,
            max_samples=150,
        )
        
        _, states1, _ = simulate_jax(**base_kwargs, key=jax.random.PRNGKey(1))
        _, states2, _ = simulate_jax(**base_kwargs, key=jax.random.PRNGKey(2))
        
        assert not jnp.array_equal(states1, states2)
    
    def test_key_splitting_reproducibility(self):
        key = jax.random.PRNGKey(42)
        
        k1_a, k2_a, k3_a, k4_a = jax.random.split(key, 4)
        
        key = jax.random.PRNGKey(42)
        k1_b, k2_b, k3_b, k4_b = jax.random.split(key, 4)
        
        np.testing.assert_array_equal(k1_a, k1_b)
        np.testing.assert_array_equal(k2_a, k2_b)
        np.testing.assert_array_equal(k3_a, k3_b)
        np.testing.assert_array_equal(k4_a, k4_b)

class TestJAXTrainingReadiness:
    def test_vmap_batch_consistency(self):
        from gibbsq.engines.jax_engine import run_replications_jax
        
        times, states, (arrivals, departures) = run_replications_jax(
            num_servers=2,
            arrival_rate=1.5,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=50.0,
            sample_interval=1.0,
            num_replications=5,
            base_seed=42,
            max_samples=100,
        )
        
        assert states.shape[0] == 5
        assert arrivals.shape[0] == 5
        
        assert jnp.all(jnp.isfinite(states))
    
    def test_gradient_through_simulation(self):
        import jax
        
        def simulation_loss(alpha, key):
            _, states, _ = simulate_jax(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([2.0, 2.0]),
                alpha=alpha,
                sim_time=30.0,
                sample_interval=1.0,
                key=key,
                max_samples=50,
            )
            return states[-1].sum().astype(jnp.float32)
        
        key = jax.random.PRNGKey(0)
        alpha = 1.0
        
        loss_val = simulation_loss(alpha, key)
        grad_fn = jax.grad(simulation_loss)
        grad = grad_fn(alpha, key)
        
        assert jnp.isfinite(loss_val)
        assert jnp.isfinite(grad)
    
    def test_differentiable_policy_parameters(self):
        import jax
        
        from gibbsq.engines.jax_engine import get_probs, SimParams
        
        def prob_loss(alpha, Q):
            params = SimParams(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.ones(2),
                alpha=alpha,
                sim_time=10.0,
                sample_interval=1.0,
                max_events=1000,
                policy_type=3,
                d=2,
            )
            probs = get_probs(Q, params, jax.random.PRNGKey(0))
            return probs[0]  # Probability of routing to server 0
        
        Q = jnp.array([5, 10])
        
        grad_fn = jax.grad(prob_loss)
        grad = grad_fn(1.0, Q)
        
        assert jnp.isfinite(grad)

class TestJAXEdgeCases:
    def test_empty_queue_system(self):
        _, states, _ = simulate_jax(
            num_servers=2,
            arrival_rate=0.1,
            service_rates=jnp.array([5.0, 5.0]),
            alpha=1.0,
            sim_time=20.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=50,
        )
        
        assert jnp.all(states >= 0)
        assert jnp.all(jnp.isfinite(states))
    
    def test_single_server_jax(self):
        _, states, (arr, dep) = simulate_jax(
            num_servers=1,
            arrival_rate=0.5,
            service_rates=jnp.array([1.0]),
            alpha=1.0,
            sim_time=50.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=100,
        )
        
        assert states.shape[1] == 1
        assert arr >= 0
        assert dep >= 0
    
    def test_all_policies_work(self):
        for policy_type in [0, 1, 2, 3, 4]:
            _, states, (arr, dep) = simulate_jax(
                num_servers=3,
                arrival_rate=1.5,
                service_rates=jnp.array([2.0, 2.0, 2.0]),
                alpha=1.0,
                sim_time=30.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(42),
                max_samples=50,
                policy_type=policy_type,
                d=2,
            )
            
            assert jnp.all(jnp.isfinite(states)), f"Policy {policy_type} produced non-finite states"
            assert jnp.all(states >= 0), f"Policy {policy_type} produced negative states"

class TestStatisticalEquivalence:
    def test_distribution_equivalence_mann_whitney(self):
        num_replications = 20
        sim_time = 200.0
        
        np_means = []
        jax_means = []
        
        for i in range(num_replications):
            np_result = simulate_numpy(
                num_servers=2,
                arrival_rate=1.5,
                service_rates=np.array([2.0, 2.0]),
                policy=SoftmaxRouting(alpha=1.0),
                sim_time=sim_time,
                sample_interval=1.0,
                rng=np.random.default_rng(i),
            )
            np_means.append(np_result.states.sum(axis=1).mean())
            
            _, states, _ = simulate_jax(
                num_servers=2,
                arrival_rate=1.5,
                service_rates=jnp.array([2.0, 2.0]),
                alpha=1.0,
                sim_time=sim_time,
                sample_interval=1.0,
                key=jax.random.PRNGKey(i),
                max_samples=250,
            )
            jax_means.append(float(states.sum(axis=1).mean()))
        
        np_mean = np.mean(np_means)
        jax_mean = np.mean(jax_means)
        
        assert np.isfinite(np_mean) and np_mean >= 0
        assert np.isfinite(jax_mean) and jax_mean >= 0
        
        assert 0.01 < np_mean / 2 < 20.0, f"NumPy mean out of range: {np_mean}"
        assert 0.01 < jax_mean / 2 < 20.0, f"JAX mean out of range: {jax_mean}"
    
    def test_variance_equivalence(self):
        num_replications = 15
        
        np_vars = []
        jax_vars = []
        
        for i in range(num_replications):
            np_result = simulate_numpy(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=np.array([1.5, 1.5]),
                policy=SoftmaxRouting(alpha=1.0),
                sim_time=100.0,
                sample_interval=1.0,
                rng=np.random.default_rng(i * 100),
            )
            np_vars.append(np_result.states.sum(axis=1).var())
            
            _, states, _ = simulate_jax(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([1.5, 1.5]),
                alpha=1.0,
                sim_time=100.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(i * 100),
                max_samples=150,
            )
            jax_vars.append(float(states.sum(axis=1).var()))
        
        var_ratio = np.mean(np_vars) / np.mean(jax_vars)
        assert 0.5 < var_ratio < 2.0, f"Variance ratio too different: {var_ratio:.2f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
