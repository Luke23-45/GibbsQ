"""
Tests for ComponentRegistry and Builders functionality.

These tests verify the correct operation of the decorator-based policy
registration system and the builder functions that depend on it.

IMPORTANT: Tests that call ComponentRegistry.clear() must restore the
original registrations afterward to avoid breaking other tests.
"""

import pytest
import numpy as np
import sys


def _get_registered_policies():
    """Snapshot of current registry state."""
    from gibbsq.core.registry import ComponentRegistry
    return dict(ComponentRegistry._policies)


def _restore_registered_policies(snapshot):
    """Restore registry to previous state."""
    from gibbsq.core.registry import ComponentRegistry
    ComponentRegistry._policies.clear()
    ComponentRegistry._policies.update(snapshot)


class TestRegistryImportOrder:
    """Critical tests for the import order dependency bug.
    
    The registry uses decorator-based registration. If builders.py is imported
    without first importing policies.py, the registry will be empty and
    build_policy_by_name() will fail.
    """

    def test_registry_populated_after_policies_import(self):
        """Verify that importing policies.py populates the registry."""
        # Import policies to trigger registration
        from gibbsq.core import policies as policies_module
        from gibbsq.core.registry import ComponentRegistry
        
        # Check that all expected policies are registered
        registered = ComponentRegistry.list_policies()
        
        expected_policies = [
            "softmax", "uniform", "proportional", 
            "jsq", "power_of_d", "jssq", "uas"
        ]
        
        for policy_name in expected_policies:
            assert policy_name in registered, (
                f"Policy '{policy_name}' not registered. "
                f"Available: {registered}"
            )

    def test_build_policy_by_name_works_after_policies_import(self):
        """Verify build_policy_by_name works after policies are imported."""
        from gibbsq.core.builders import build_policy_by_name
        from gibbsq.core.registry import ComponentRegistry
        
        # This should NOT raise KeyError if import chain is correct
        policy = build_policy_by_name("softmax", alpha=1.0)
        assert policy is not None
        
        policy2 = build_policy_by_name("jsq")
        assert policy2 is not None


class TestRegistryStandaloneImport:
    """Test what happens when importing builders without policies.
    
    This simulates the bug scenario where stability_sweep.py imports
    builders.py directly without first importing policies.
    """
    
    def test_builders_import_without_policies_fails(self, tmp_path):
        """Verify that importing builders alone does NOT auto-register policies.
        
        This test creates an isolated Python subprocess to verify the bug.
        """
        import subprocess
        
        # Create a test script that imports builders without policies
        test_script = '''
import sys
# Simulate the problematic import order from stability_sweep.py
from gibbsq.core.builders import build_policy_by_name

try:
    policy = build_policy_by_name("softmax", alpha=1.0)
    print("SUCCESS")
except KeyError as e:
    print(f"KEYERROR: {e}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
'''
        
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )
        
        # If the bug exists, this will print KEYERROR
        # If fixed, it should print SUCCESS
        output = result.stdout.strip()
        
        # Document current behavior - this SHOULD fail with the current bug
        # After fix, this should pass
        if "KEYERROR" in output:
            pytest.fail(
                f"Import order bug confirmed: build_policy_by_name fails when "
                f"builders.py is imported without policies.py. Output: {output}"
            )


class TestComponentRegistry:
    """Tests for ComponentRegistry class functionality."""

    def test_register_policy_decorator(self):
        """Test the register_policy decorator."""
        from gibbsq.core.registry import ComponentRegistry
        
        # Snapshot and restore pattern
        snapshot = _get_registered_policies()
        
        try:
            ComponentRegistry.clear()
            
            @ComponentRegistry.register_policy("test_policy")
            class TestPolicy:
                def __init__(self, value=1):
                    self.value = value
                
                def __call__(self, Q, rng):
                    return np.ones(len(Q)) / len(Q)
            
            # Verify registration
            assert "test_policy" in ComponentRegistry.list_policies()
            
            # Verify we can build it
            policy = ComponentRegistry.build_policy("test_policy")
            assert isinstance(policy, TestPolicy)
            assert policy.value == 1
        finally:
            _restore_registered_policies(snapshot)

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises ValueError."""
        from gibbsq.core.registry import ComponentRegistry
        
        snapshot = _get_registered_policies()
        
        try:
            ComponentRegistry.clear()
            
            @ComponentRegistry.register_policy("dup_test")
            class Policy1:
                pass
            
            with pytest.raises(ValueError, match="already registered"):
                @ComponentRegistry.register_policy("dup_test")
                class Policy2:
                    pass
        finally:
            _restore_registered_policies(snapshot)

    def test_unknown_policy_raises_keyerror(self):
        """Test that unknown policy name raises KeyError with helpful message."""
        from gibbsq.core.registry import ComponentRegistry
        
        # Import policies first to ensure registry is populated
        import gibbsq.core.policies  # noqa: F401
        
        with pytest.raises(KeyError, match="Unknown policy"):
            ComponentRegistry.build_policy("nonexistent_policy_xyz")

    def test_list_policies_returns_sorted(self):
        """Test that list_policies returns sorted list."""
        from gibbsq.core.registry import ComponentRegistry
        from gibbsq.core.policies import SoftmaxRouting  # trigger registration
        
        policies = ComponentRegistry.list_policies()
        assert policies == sorted(policies)


class TestBuildPolicyByName:
    """Tests for build_policy_by_name convenience function."""

    def test_build_softmax(self):
        """Test building softmax policy."""
        from gibbsq.core.builders import build_policy_by_name
        from gibbsq.core.policies import SoftmaxRouting
        
        policy = build_policy_by_name("softmax", alpha=2.5)
        
        assert isinstance(policy, SoftmaxRouting)
        assert policy.alpha == 2.5

    def test_build_uniform(self):
        """Test building uniform policy."""
        from gibbsq.core.builders import build_policy_by_name
        from gibbsq.core.policies import UniformRouting
        
        policy = build_policy_by_name("uniform")
        
        assert isinstance(policy, UniformRouting)
        
        # Verify it works
        rng = np.random.default_rng(42)
        probs = policy(np.array([1, 2, 3]), rng)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])

    def test_build_proportional_requires_mu(self):
        """Test that proportional policy requires mu parameter."""
        from gibbsq.core.builders import build_policy_by_name
        
        with pytest.raises(ValueError, match="requires.*mu"):
            build_policy_by_name("proportional")

    def test_build_proportional_with_mu(self):
        """Test building proportional policy with mu."""
        from gibbsq.core.builders import build_policy_by_name
        from gibbsq.core.policies import ProportionalRouting
        
        mu = np.array([1.0, 2.0, 3.0])
        policy = build_policy_by_name("proportional", mu=mu)
        
        assert isinstance(policy, ProportionalRouting)
        
        rng = np.random.default_rng(42)
        probs = policy(np.array([5, 5, 5]), rng)
        np.testing.assert_allclose(probs, [1/6, 2/6, 3/6])

    def test_build_jsq(self):
        """Test building JSQ policy."""
        from gibbsq.core.builders import build_policy_by_name
        from gibbsq.core.policies import JSQRouting
        
        policy = build_policy_by_name("jsq")
        
        assert isinstance(policy, JSQRouting)

    def test_build_power_of_d(self):
        """Test building power-of-d policy."""
        from gibbsq.core.builders import build_policy_by_name
        from gibbsq.core.policies import PowerOfDRouting
        
        policy = build_policy_by_name("power_of_d", d=3)
        
        assert isinstance(policy, PowerOfDRouting)

    def test_build_jssq_requires_mu(self):
        """Test that JSSQ policy requires mu parameter."""
        from gibbsq.core.builders import build_policy_by_name
        
        with pytest.raises(ValueError, match="requires.*mu"):
            build_policy_by_name("jssq")

    def test_build_jssq_with_mu(self):
        """Test building JSSQ policy with mu."""
        from gibbsq.core.builders import build_policy_by_name
        from gibbsq.core.policies import JSSQRouting
        
        mu = np.array([1.0, 2.0])
        policy = build_policy_by_name("jssq", mu=mu)
        
        assert isinstance(policy, JSSQRouting)

    def test_build_unweighted_potential_softmax_raises_error(self):
        """Test that legacy unweighted softmax calculation is no longer available by the old name."""
        from gibbsq.core.builders import build_policy_by_name
        
        # sojourn_softmax has been removed - should raise KeyError
        mu = np.array([1.0, 2.0])
        with pytest.raises(KeyError, match="Unknown policy.*sojourn_softmax"):
            build_policy_by_name("sojourn_softmax", mu=mu, alpha=5.0)

    def test_build_uas_with_mu(self):
        """Test building uas policy with mu (replacement for legacy unweighted nomenclature)."""
        from gibbsq.core.builders import build_policy_by_name
        from gibbsq.core.policies import UASRouting
        
        mu = np.array([1.0, 2.0])
        policy = build_policy_by_name("uas", mu=mu, alpha=5.0)
        
        assert isinstance(policy, UASRouting)
        assert policy.alpha == 5.0


class TestBuildPolicy:
    """Tests for build_policy function with config slices."""

    def test_build_policy_from_config(self):
        """Test building policy from config objects."""
        from gibbsq.core.builders import build_policy
        from gibbsq.core.config import PolicyConfig, SystemConfig
        from gibbsq.core.policies import SoftmaxRouting
        
        policy_cfg = PolicyConfig(name="softmax", d=2)
        system_cfg = SystemConfig(
            num_servers=3,
            arrival_rate=1.5,
            service_rates=[1.0, 1.0, 1.0],
            alpha=2.0,
        )
        
        policy = build_policy(policy_cfg, system_cfg)
        
        assert isinstance(policy, SoftmaxRouting)
        assert policy.alpha == 2.0


class TestMakePolicyBackwardCompat:
    """Tests for backward compatibility of make_policy function."""

    def test_make_policy_delegates_to_registry(self):
        """Test that make_policy delegates to ComponentRegistry."""
        from gibbsq.core.policies import make_policy, SoftmaxRouting
        
        policy = make_policy("softmax", alpha=1.5)
        
        assert isinstance(policy, SoftmaxRouting)
        assert policy.alpha == 1.5

    def test_make_policy_all_types(self):
        """Test make_policy for all policy types."""
        from gibbsq.core.policies import make_policy
        
        # Test each policy type
        p1 = make_policy("softmax", alpha=1.0)
        assert p1 is not None
        
        p2 = make_policy("uniform")
        assert p2 is not None
        
        p3 = make_policy("proportional", mu=np.array([1.0, 2.0]))
        assert p3 is not None
        
        p4 = make_policy("jsq")
        assert p4 is not None
        
        p5 = make_policy("power_of_d", d=2)
        assert p5 is not None
        
        p6 = make_policy("jssq", mu=np.array([1.0, 2.0]))
        assert p6 is not None
        
        p7 = make_policy("uas", mu=np.array([1.0, 2.0]), alpha=1.0)
        assert p7 is not None


class TestRegistryThreadSafety:
    """Tests for thread safety of ComponentRegistry.
    
    Note: The current implementation uses class-level mutable dicts without
    synchronization, which could cause issues in multi-threaded environments.
    These tests document the current behavior.
    """

    def test_concurrent_registration(self):
        """Test concurrent registration from multiple threads.
        
        This test documents potential race conditions.
        """
        import threading
        from gibbsq.core.registry import ComponentRegistry
        
        snapshot = _get_registered_policies()
        
        try:
            ComponentRegistry.clear()
            errors = []
            
            def register_policy(i):
                try:
                    # Each thread tries to register a unique policy
                    name = f"thread_policy_{i}"
                    
                    @ComponentRegistry.register_policy(name)
                    class ThreadPolicy:
                        pass
                except Exception as e:
                    errors.append((i, e))
            
            threads = [
                threading.Thread(target=register_policy, args=(i,))
                for i in range(10)
            ]
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # All should succeed since each registers a unique name
            assert len(errors) == 0, f"Errors during concurrent registration: {errors}"
            
            # Verify all policies registered
            registered = ComponentRegistry.list_policies()
            for i in range(10):
                assert f"thread_policy_{i}" in registered
        finally:
            _restore_registered_policies(snapshot)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
