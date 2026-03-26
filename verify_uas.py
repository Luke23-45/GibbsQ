#!/usr/bin/env python
"""Verification script for UAS implementation and test fixes."""

import sys
sys.path.insert(0, 'src')

import numpy as np

print('=' * 60)
print('UAS Implementation Verification')
print('=' * 60)

# Test 1: Test 3 fix - generator_drift import
print('\n[1] Testing Test 3 fix (generator_drift import)...')
try:
    from gibbsq.core.drift import generator_drift
    print('    PASS: generator_drift imported successfully')
except ImportError as e:
    print(f'    FAIL: {e}')
    sys.exit(1)

# Test 2: Test 4 fix - stationarity_diagnostic
print('\n[2] Testing Test 4 fix (stationarity_diagnostic)...')
try:
    from gibbsq.analysis.metrics import stationarity_diagnostic
    
    # Mock SimResult with 'states' attribute (what _trim_burn_in expects)
    class MockResult:
        def __init__(self):
            self.times = np.linspace(0, 100, 200)
            self.states = np.random.randint(0, 10, (200, 3))
    
    result = MockResult()
    diag = stationarity_diagnostic(result, burn_in_fraction=0.2)
    is_stat = diag['is_stationary']
    print(f'    PASS: stationarity_diagnostic works (is_stationary={is_stat})')
except Exception as e:
    print(f'    FAIL: {e}')
    sys.exit(1)

# Test 3: UASRouting class
print('\n[3] Testing UASRouting class...')
try:
    from gibbsq.core.policies import UASRouting, SojournTimeSoftmaxRouting
    from gibbsq.core.registry import ComponentRegistry
    
    mu = np.array([1.0, 2.0, 3.0])
    Q = np.array([5, 3, 0])
    rng = np.random.default_rng(42)
    
    uas = UASRouting(mu, alpha=1.0)
    probs = uas(Q, rng)
    
    assert abs(probs.sum() - 1.0) < 1e-10, 'Probabilities must sum to 1'
    assert 'uas' in ComponentRegistry.list_policies(), 'uas must be registered'
    
    # Compare with SojournTime
    sojourn = SojournTimeSoftmaxRouting(mu, alpha=1.0)
    probs_sojourn = sojourn(Q, rng)
    assert probs[2] > probs_sojourn[2], 'UAS should weight faster servers higher'
    
    print(f'    PASS: UASRouting works correctly')
    print(f'    UAS probs: {probs}')
    print(f'    Sojourn probs: {probs_sojourn}')
except Exception as e:
    print(f'    FAIL: {e}')
    sys.exit(1)

# Test 4: features.py UAS functions
print('\n[4] Testing features.py UAS functions...')
try:
    from gibbsq.core.features import softmax_on_sojourn_uas_numpy, softmax_on_sojourn_uas
    
    probs_feat = softmax_on_sojourn_uas_numpy(Q, mu, alpha=1.0)
    assert np.allclose(probs, probs_feat), 'UASRouting must match feature function'
    print(f'    PASS: softmax_on_sojourn_uas_numpy works')
except Exception as e:
    print(f'    FAIL: {e}')
    sys.exit(1)

# Test 5: drift.py UAS mode
print('\n[5] Testing drift.py UAS mode...')
try:
    from gibbsq.core.drift import _softmax_probs
    
    probs_drift = _softmax_probs(Q, 1.0, mu, mode='uas')
    assert np.allclose(probs, probs_drift), 'UASRouting must match drift function'
    print(f'    PASS: _softmax_probs with mode="uas" works')
except Exception as e:
    print(f'    FAIL: {e}')
    sys.exit(1)

# Test 6: JAX engine policy_type 6
print('\n[6] Testing JAX engine policy_type 6...')
try:
    from gibbsq.engines.jax_engine import _validate_inputs
    import jax.numpy as jnp
    
    _validate_inputs(
        num_servers=3, arrival_rate=1.0, service_rates=jnp.array([1.0, 2.0, 3.0]),
        sim_time=10.0, sample_interval=1.0, max_samples=10, policy_type=6, d=2
    )
    print(f'    PASS: policy_type=6 accepted by JAX engine')
except Exception as e:
    print(f'    FAIL: {e}')
    sys.exit(1)

print('\n' + '=' * 60)
print('ALL TESTS PASSED')
print('=' * 60)
print('\nSummary:')
print('  - Test 3 fix (ImportError): VERIFIED')
print('  - Test 4 fix (TypeError): VERIFIED')
print('  - UASRouting class: IMPLEMENTED')
print('  - JAX engine policy_type 6: IMPLEMENTED')
print('  - features.py UAS functions: IMPLEMENTED')
print('  - drift.py UAS mode: IMPLEMENTED')
