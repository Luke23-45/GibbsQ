#!/usr/bin/env python
"""Test that deprecation warning is raised for SojournTimeSoftmaxRouting."""

import sys
sys.path.insert(0, 'src')

import warnings
import numpy as np

# Test 1: Verify deprecation warning is raised
print("Test 1: Deprecation warning check")
print("-" * 40)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from gibbsq.core.policies import SojournTimeSoftmaxRouting
    
    mu = np.array([1.0, 2.0, 3.0])
    policy = SojournTimeSoftmaxRouting(mu, alpha=1.0)
    
    if len(w) == 1 and issubclass(w[0].category, DeprecationWarning):
        print("PASS: DeprecationWarning raised")
        print(f"  Message: {w[0].message}")
    else:
        print(f"FAIL: Expected DeprecationWarning, got {len(w)} warnings")
        for warning in w:
            print(f"  {warning.category.__name__}: {warning.message}")

# Test 2: Verify UASRouting does NOT raise warning
print("\nTest 2: UASRouting (no warning expected)")
print("-" * 40)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from gibbsq.core.policies import UASRouting
    
    policy_uas = UASRouting(mu, alpha=1.0)
    
    if len(w) == 0:
        print("PASS: No warning raised for UASRouting")
    else:
        print(f"FAIL: Unexpected warning for UASRouting")
        for warning in w:
            print(f"  {warning.category.__name__}: {warning.message}")

# Test 3: Verify both policies work correctly
print("\nTest 3: Functional correctness")
print("-" * 40)

Q = np.array([5, 3, 0])
rng = np.random.default_rng(42)

# Suppress warning for functional test
warnings.filterwarnings("ignore", category=DeprecationWarning)
policy_sojourn = SojournTimeSoftmaxRouting(mu, alpha=1.0)
probs_sojourn = policy_sojourn(Q, rng)

policy_uas = UASRouting(mu, alpha=1.0)
probs_uas = policy_uas(Q, rng)

print(f"SojournTime probs: {probs_sojourn} (sum={probs_sojourn.sum():.6f})")
print(f"UAS probs: {probs_uas} (sum={probs_uas.sum():.6f})")

if abs(probs_sojourn.sum() - 1.0) < 1e-10 and abs(probs_uas.sum() - 1.0) < 1e-10:
    print("PASS: Both policies produce valid probability distributions")
else:
    print("FAIL: Invalid probability distributions")

print("\n" + "=" * 40)
print("All deprecation tests completed")
