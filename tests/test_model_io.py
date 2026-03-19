"""
Tests for model_io utilities.

Tests for:
- resolve_model_pointer: Model weight pointer resolution
- save_model_pointer: Writing pointer files
- NeuralSSAPolicy: JAX-to-NumPy bridge
- DeterministicNeuralPolicy: Greedy policy wrapper
- StochasticNeuralPolicy: Stochastic policy wrapper
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


class TestResolveModelPointer:
    """Tests for resolve_model_pointer function."""

    def test_resolve_dr_pointer_first(self, tmp_path):
        """Test that DR pointer is resolved first."""
        from gibbsq.utils.model_io import resolve_model_pointer
        
        # Create mock project structure
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        # Create model file
        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        # Create all three pointer files
        (output_root / "latest_domain_randomized_weights.txt").write_text(
            str(model_file.relative_to(project_root))
        )
        (output_root / "latest_reinforce_weights.txt").write_text(
            "nonexistent_path"
        )
        (output_root / "latest_weights.txt").write_text(
            "nonexistent_path2"
        )
        
        # Should resolve to DR pointer first
        result = resolve_model_pointer(project_root, output_root)
        assert result == model_file

    def test_resolve_reinforce_pointer_second(self, tmp_path):
        """Test that REINFORCE pointer is resolved second."""
        from gibbsq.utils.model_io import resolve_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        # Only create reinforce pointer (no DR)
        (output_root / "latest_reinforce_weights.txt").write_text(
            str(model_file.relative_to(project_root))
        )
        
        result = resolve_model_pointer(project_root, output_root)
        assert result == model_file

    def test_resolve_legacy_pointer_third(self, tmp_path):
        """Test that legacy pointer is resolved third with warning."""
        from gibbsq.utils.model_io import resolve_model_pointer
        import logging
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        # Only create legacy pointer
        (output_root / "latest_weights.txt").write_text(
            str(model_file.relative_to(project_root))
        )
        
        # Should resolve but log warning
        with pytest.MonkeyPatch().context() as m:
            # Capture log warnings
            result = resolve_model_pointer(project_root, output_root)
            assert result == model_file

    def test_resolve_absolute_path_pointer(self, tmp_path):
        """Test that absolute paths in pointers work."""
        from gibbsq.utils.model_io import resolve_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        # Write absolute path in pointer
        (output_root / "latest_reinforce_weights.txt").write_text(
            str(model_file.resolve())
        )
        
        result = resolve_model_pointer(project_root, output_root)
        assert result == model_file

    def test_resolve_nonexistent_model_raises(self, tmp_path):
        """Test that nonexistent model path raises FileNotFoundError."""
        from gibbsq.utils.model_io import resolve_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        # Pointer points to nonexistent file
        (output_root / "latest_reinforce_weights.txt").write_text(
            "nonexistent/model.eqx"
        )
        
        # Should skip this pointer and try others, then fail
        with pytest.raises(FileNotFoundError, match="No valid model pointer"):
            resolve_model_pointer(project_root, output_root)

    def test_resolve_no_pointer_files_raises(self, tmp_path):
        """Test that missing pointer files raises FileNotFoundError."""
        from gibbsq.utils.model_io import resolve_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        with pytest.raises(FileNotFoundError, match="No valid model pointer"):
            resolve_model_pointer(project_root, output_root)


class TestSaveModelPointer:
    """Tests for save_model_pointer function."""

    def test_save_creates_pointer_file(self, tmp_path):
        """Test that save_model_pointer creates the pointer file."""
        from gibbsq.utils.model_io import save_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        
        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        result = save_model_pointer(
            model_path=model_file,
            project_root=project_root,
            output_root=output_root,
            pointer_name="latest_test_weights.txt"
        )
        
        assert result.exists()
        assert result.name == "latest_test_weights.txt"
        
        # Verify content is relative path (normalize separators for cross-platform)
        content = result.read_text().strip().replace("\\", "/")
        assert content == "models/weights.eqx"

    def test_save_creates_output_directory(self, tmp_path):
        """Test that save_model_pointer creates output directory if needed."""
        from gibbsq.utils.model_io import save_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs" / "nested" / "dir"
        project_root.mkdir()
        
        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        result = save_model_pointer(
            model_path=model_file,
            project_root=project_root,
            output_root=output_root,
        )
        
        assert output_root.exists()
        assert result.exists()

    def test_save_model_outside_project_raises(self, tmp_path):
        """Test that model_path outside project_root raises ValueError."""
        from gibbsq.utils.model_io import save_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        # Model is outside project root
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        model_file = other_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        # This should raise ValueError because model is not under project_root
        with pytest.raises(ValueError, match="not in the subpath"):
            save_model_pointer(
                model_path=model_file,
                project_root=project_root,
                output_root=output_root,
            )


class TestDeterministicNeuralPolicy:
    """Tests for DeterministicNeuralPolicy wrapper."""

    def test_deterministic_returns_one_hot(self):
        """Test that policy returns one-hot vector for greedy selection."""
        from gibbsq.utils.model_io import DeterministicNeuralPolicy
        
        # Create a mock neural network
        class MockNet:
            def get_numpy_params(self):
                return [
                    (np.eye(3), np.zeros(3)),  # layer 1: identity
                    (np.eye(3), np.zeros(3)),  # layer 2: identity
                    (np.eye(3), np.zeros(3)),  # layer 3: identity
                ]
            
            def numpy_forward(self, x, params, config):
                # Simple forward: just return x
                return x
            
            config = type('Config', (), {'preprocessing': 'none', 'use_rho': False})()
        
        mu = np.array([1.0, 1.0, 1.0])
        policy = DeterministicNeuralPolicy(MockNet(), mu)
        
        rng = np.random.default_rng(42)
        
        # With equal features, argmax should pick first (index 0)
        Q = np.array([0.0, 0.0, 0.0])
        probs = policy(Q, rng)
        
        # Should be one-hot at argmax position
        assert np.sum(probs) == 1.0
        assert np.sum(probs == 1.0) == 1

    def test_deterministic_ignores_rng(self):
        """Test that deterministic policy ignores RNG (always same output)."""
        from gibbsq.utils.model_io import DeterministicNeuralPolicy
        
        class MockNet:
            def get_numpy_params(self):
                return [(np.eye(2), np.zeros(2))]
            
            def numpy_forward(self, x, params, config):
                return x
            
            config = type('Config', (), {'preprocessing': 'none', 'use_rho': False})()
        
        mu = np.array([1.0, 1.0])
        policy = DeterministicNeuralPolicy(MockNet(), mu)
        
        Q = np.array([1.0, 2.0])
        
        # Different RNGs should give same result
        probs1 = policy(Q, np.random.default_rng(1))
        probs2 = policy(Q, np.random.default_rng(999))
        
        np.testing.assert_array_equal(probs1, probs2)


class TestStochasticNeuralPolicy:
    """Tests for StochasticNeuralPolicy wrapper."""

    def test_stochastic_returns_valid_distribution(self):
        """Test that policy returns valid probability distribution."""
        from gibbsq.utils.model_io import StochasticNeuralPolicy
        
        class MockNet:
            def get_numpy_params(self):
                return [(np.eye(3), np.zeros(3))]
            
            def numpy_forward(self, x, params, config):
                return x
            
            config = type('Config', (), {'preprocessing': 'none', 'use_rho': False})()
        
        mu = np.array([1.0, 1.0, 1.0])
        policy = StochasticNeuralPolicy(MockNet(), mu)
        
        rng = np.random.default_rng(42)
        Q = np.array([1.0, 2.0, 3.0])
        probs = policy(Q, rng)
        
        # Should sum to 1
        assert np.isclose(np.sum(probs), 1.0)
        # All probabilities should be non-negative
        assert np.all(probs >= 0)

    def test_stochastic_numerical_stability(self):
        """Test numerical stability with large logit differences."""
        from gibbsq.utils.model_io import StochasticNeuralPolicy
        
        class MockNet:
            def get_numpy_params(self):
                return [(np.eye(2), np.zeros(2))]
            
            def numpy_forward(self, x, params, config):
                # Return large values to test stability
                return x
            
            config = type('Config', (), {'preprocessing': 'none', 'use_rho': False})()
        
        mu = np.array([1.0, 1.0])
        policy = StochasticNeuralPolicy(MockNet(), mu)
        
        rng = np.random.default_rng(42)
        
        # Large difference in input
        Q = np.array([1000.0, 0.0])
        probs = policy(Q, rng)
        
        # Should not have NaN or Inf
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        assert np.isclose(np.sum(probs), 1.0)


class TestNeuralSSAPolicyCache:
    """Tests for NeuralSSAPolicy LRU cache behavior."""

    def test_cache_returns_same_result_for_same_input(self):
        """Test that cache returns consistent results for same queue state."""
        pytest.importorskip("jax")
        pytest.importorskip("equinox")
        
        from gibbsq.utils.model_io import NeuralSSAPolicy
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        
        # Create a simple model
        class SimpleModel(eqx.Module):
            weight: jnp.ndarray
            
            def __init__(self, key):
                self.weight = jax.random.normal(key, (3,))
            
            def __call__(self, x):
                return self.weight * x
        
        key = jax.random.PRNGKey(42)
        model = SimpleModel(key)
        
        policy = NeuralSSAPolicy(model)
        
        rng = np.random.default_rng(42)
        Q = np.array([1.0, 2.0, 3.0])
        
        # Call twice with same Q
        probs1 = policy(Q, rng)
        probs2 = policy(Q, rng)
        
        # Should return identical results (from cache)
        np.testing.assert_array_equal(probs1, probs2)

    def test_cache_different_inputs(self):
        """Test that different inputs produce different outputs."""
        pytest.importorskip("jax")
        pytest.importorskip("equinox")
        
        from gibbsq.utils.model_io import NeuralSSAPolicy
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        
        class SimpleModel(eqx.Module):
            weight: jnp.ndarray
            
            def __init__(self, key):
                self.weight = jax.random.normal(key, (3,))
            
            def __call__(self, x):
                return self.weight * x
        
        key = jax.random.PRNGKey(42)
        model = SimpleModel(key)
        
        policy = NeuralSSAPolicy(model)
        
        rng = np.random.default_rng(42)
        
        Q1 = np.array([1.0, 2.0, 3.0])
        Q2 = np.array([3.0, 2.0, 1.0])
        
        probs1 = policy(Q1, rng)
        probs2 = policy(Q2, rng)
        
        # Different inputs should give different outputs
        assert not np.allclose(probs1, probs2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
