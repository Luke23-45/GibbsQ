import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace

class TestResolveModelPointer:
    def test_resolve_reinforce_pointer_first(self, tmp_path):
        from gibbsq.utils.model_io import resolve_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        (output_root / "latest_domain_randomized_weights.txt").write_text(
            "nonexistent_path"
        )
        (output_root / "latest_reinforce_weights.txt").write_text(
            str(model_file.relative_to(project_root))
        )
        
        result = resolve_model_pointer(project_root, output_root)
        assert result == model_file

    def test_resolve_dr_pointer_second(self, tmp_path):
        from gibbsq.utils.model_io import resolve_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        (output_root / "latest_domain_randomized_weights.txt").write_text(
            str(model_file.relative_to(project_root))
        )
        
        result = resolve_model_pointer(project_root, output_root)
        assert result == model_file

    def test_resolve_legacy_pointer_third(self, tmp_path):
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
        
        (output_root / "latest_weights.txt").write_text(
            str(model_file.relative_to(project_root))
        )
        
        with pytest.MonkeyPatch().context() as m:
            result = resolve_model_pointer(project_root, output_root)
            assert result == model_file

    def test_resolve_absolute_path_pointer(self, tmp_path):
        from gibbsq.utils.model_io import resolve_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        (output_root / "latest_reinforce_weights.txt").write_text(
            str(model_file.resolve())
        )
        
        result = resolve_model_pointer(project_root, output_root)
        assert result == model_file

    def test_resolve_nonexistent_model_raises(self, tmp_path):
        from gibbsq.utils.model_io import resolve_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        (output_root / "latest_reinforce_weights.txt").write_text(
            "nonexistent/model.eqx"
        )
        
        with pytest.raises(FileNotFoundError, match="No valid model pointer"):
            resolve_model_pointer(project_root, output_root)

    def test_resolve_no_pointer_files_raises(self, tmp_path):
        from gibbsq.utils.model_io import resolve_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        with pytest.raises(FileNotFoundError, match="No valid model pointer"):
            resolve_model_pointer(project_root, output_root)

    def test_missing_pointer_message_mentions_public_training_commands(self, tmp_path):
        from gibbsq.utils.model_io import resolve_model_pointer

        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_model_pointer(project_root, output_root)

        message = str(exc_info.value)
        assert "reinforce_train" in message
        assert "bc_train" in message
        assert "Track 1/3" not in message
        assert "MoEQ" not in message
        assert "dr_train" not in message

    def test_strict_public_eval_rejects_bc_pointer(self, tmp_path):
        from gibbsq.utils.model_io import resolve_model_pointer

        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()

        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        (output_root / "latest_bc_weights.txt").write_text(str(model_file.relative_to(project_root)))

        with pytest.raises(FileNotFoundError, match="BC warm-start only"):
            resolve_model_pointer(
                project_root,
                output_root,
                allow_bc=False,
                allow_legacy=False,
            )

    def test_strict_public_eval_optional_lookup_returns_none_for_bc_only(self, tmp_path):
        from gibbsq.utils.model_io import resolve_model_pointer_or_none

        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()

        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        (output_root / "latest_bc_weights.txt").write_text(str(model_file.relative_to(project_root)))

        result = resolve_model_pointer_or_none(
            project_root,
            output_root,
            allow_bc=False,
            allow_legacy=False,
        )
        assert result is None

    def test_legacy_pointer_warning_mentions_public_pointer_names(self, tmp_path, caplog):
        from gibbsq.utils.model_io import resolve_model_pointer

        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()

        model_dir = project_root / "models"
        model_dir.mkdir()
        model_file = model_dir / "weights.eqx"
        model_file.write_text("mock weights")
        (output_root / "latest_weights.txt").write_text(str(model_file.relative_to(project_root)))

        with caplog.at_level("WARNING"):
            result = resolve_model_pointer(project_root, output_root)

        assert result == model_file
        warning_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "latest_reinforce_weights.txt" in warning_text
        assert "latest_bc_weights.txt" in warning_text
        assert "prefer REINFORCE pointers" not in warning_text

class TestSaveModelPointer:
    def test_save_creates_pointer_file(self, tmp_path):
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
        
        content = result.read_text().strip().replace("\\", "/")
        assert content == "models/weights.eqx"

    def test_save_creates_output_directory(self, tmp_path):
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
        from gibbsq.utils.model_io import save_model_pointer
        
        project_root = tmp_path / "project"
        output_root = tmp_path / "outputs"
        project_root.mkdir()
        output_root.mkdir()
        
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        model_file = other_dir / "weights.eqx"
        model_file.write_text("mock weights")
        
        with pytest.raises(ValueError, match="not in the subpath"):
            save_model_pointer(
                model_path=model_file,
                project_root=project_root,
                output_root=output_root,
            )

class TestValidateNeuralModelShape:
    def test_accepts_model_without_service_rate_features(self):
        from gibbsq.core.config import NeuralConfig
        from gibbsq.utils.model_io import validate_neural_model_shape

        class MockLayer:
            def __init__(self, out_dim, in_dim):
                self.weight = np.zeros((out_dim, in_dim), dtype=np.float32)

        class MockModel:
            layers = [MockLayer(8, 3)]

        cfg = NeuralConfig(hidden_size=8, use_service_rates=False, use_rho=False)
        validate_neural_model_shape(MockModel(), cfg, num_servers=3)

    def test_rejects_missing_service_rate_features_when_enabled(self):
        from gibbsq.core.config import NeuralConfig
        from gibbsq.utils.model_io import validate_neural_model_shape

        class MockLayer:
            def __init__(self, out_dim, in_dim):
                self.weight = np.zeros((out_dim, in_dim), dtype=np.float32)

        class MockModel:
            layers = [MockLayer(8, 3)]

        cfg = NeuralConfig(hidden_size=8, use_service_rates=True, use_rho=False)
        with pytest.raises(ValueError, match="Model input dimension mismatch"):
            validate_neural_model_shape(MockModel(), cfg, num_servers=3)

class TestBCReuseMetadata:
    def _make_cfg(self):
        return SimpleNamespace(
            system=SimpleNamespace(
                num_servers=3,
                service_rates=[1.0, 1.5, 2.0],
                alpha=1.25,
            ),
            neural=SimpleNamespace(
                hidden_size=16,
                preprocessing="log1p",
                capacity_bound=10.0,
                init_type="zero_final",
                use_rho=True,
                use_service_rates=True,
                rho_input_scale=10.0,
            ),
            neural_training=SimpleNamespace(
                bc_num_steps=100,
                bc_lr=0.002,
                bc_label_smoothing=0.1,
                weight_decay=1e-4,
            ),
            simulation=SimpleNamespace(seed=42),
        )

    def _make_bc_data(self):
        return {
            "rhos": [0.45, 0.65],
            "mu_scales": [0.5, 1.0],
            "samples_per_rho": 100,
            "expert_sim_time": 50.0,
            "sample_interval": 1.0,
            "augmentation_noise_min": -1,
            "augmentation_noise_max": 1,
        }

    def test_bc_reuse_metadata_round_trip(self, tmp_path):
        from gibbsq.utils.model_io import (
            get_bc_metadata_path,
            load_bc_reuse_metadata,
            validate_bc_reuse_metadata,
            write_bc_reuse_metadata,
        )

        cfg = self._make_cfg()
        bc_data = self._make_bc_data()
        model_path = tmp_path / "policy.eqx"
        model_path.write_text("weights")

        metadata_path = write_bc_reuse_metadata(model_path, cfg=cfg, bc_data_config=bc_data)

        assert metadata_path == get_bc_metadata_path(model_path)
        assert metadata_path.exists()
        loaded = load_bc_reuse_metadata(model_path)
        assert loaded["artifact_kind"] == "bc_actor_warm_start"
        assert loaded["compatibility"]["neural"]["hidden_size"] == 16
        assert loaded["provenance"]["simulation_seed"] == 42
        assert loaded["fingerprint"]
        validated = validate_bc_reuse_metadata(model_path, cfg=cfg, bc_data_config=bc_data)
        assert validated["fingerprint"] == loaded["fingerprint"]

    def test_bc_reuse_metadata_rejects_mismatch(self, tmp_path):
        from gibbsq.utils.model_io import validate_bc_reuse_metadata, write_bc_reuse_metadata

        cfg = self._make_cfg()
        bc_data = self._make_bc_data()
        model_path = tmp_path / "policy.eqx"
        model_path.write_text("weights")
        write_bc_reuse_metadata(model_path, cfg=cfg, bc_data_config=bc_data)

        mismatched_cfg = self._make_cfg()
        mismatched_cfg.neural.hidden_size = 32

        with pytest.raises(ValueError, match="BC metadata fingerprint mismatch"):
            validate_bc_reuse_metadata(model_path, cfg=mismatched_cfg, bc_data_config=bc_data)

    def test_bc_reuse_metadata_accepts_provenance_only_change(self, tmp_path):
        from gibbsq.utils.model_io import validate_bc_reuse_metadata, write_bc_reuse_metadata

        cfg = self._make_cfg()
        bc_data = self._make_bc_data()
        model_path = tmp_path / "policy.eqx"
        model_path.write_text("weights")
        write_bc_reuse_metadata(model_path, cfg=cfg, bc_data_config=bc_data)

        provenance_changed_cfg = self._make_cfg()
        provenance_changed_cfg.simulation.seed = 99
        provenance_changed_cfg.neural_training.bc_num_steps = 500
        provenance_changed_cfg.neural_training.bc_lr = 0.01

        validated = validate_bc_reuse_metadata(
            model_path,
            cfg=provenance_changed_cfg,
            bc_data_config=bc_data,
        )
        assert validated["fingerprint"]

    def test_bc_reuse_metadata_rejects_tampered_compatibility_payload(self, tmp_path):
        import json

        from gibbsq.utils.model_io import get_bc_metadata_path, validate_bc_reuse_metadata, write_bc_reuse_metadata

        cfg = self._make_cfg()
        bc_data = self._make_bc_data()
        model_path = tmp_path / "policy.eqx"
        model_path.write_text("weights")
        write_bc_reuse_metadata(model_path, cfg=cfg, bc_data_config=bc_data)

        metadata_path = get_bc_metadata_path(model_path)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["compatibility"]["neural"]["hidden_size"] = 32
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="stored compatibility payload"):
            validate_bc_reuse_metadata(model_path, cfg=cfg, bc_data_config=bc_data)

    def test_bc_reuse_metadata_rejects_missing_sidecar(self, tmp_path):
        from gibbsq.utils.model_io import validate_bc_reuse_metadata

        cfg = self._make_cfg()
        bc_data = self._make_bc_data()
        model_path = tmp_path / "policy.eqx"
        model_path.write_text("weights")

        with pytest.raises(FileNotFoundError):
            validate_bc_reuse_metadata(model_path, cfg=cfg, bc_data_config=bc_data)

class TestDeterministicNeuralPolicy:
    def test_deterministic_returns_one_hot(self):
        from gibbsq.utils.model_io import DeterministicNeuralPolicy
        
        class MockNet:
            def get_numpy_params(self):
                return [
                    (np.eye(3), np.zeros(3)),  # layer 1: identity
                    (np.eye(3), np.zeros(3)),  # layer 2: identity
                    (np.eye(3), np.zeros(3)),  # layer 3: identity
                ]
            
            def numpy_forward(self, x, params, config, **kwargs):
                return x
            
            config = type('Config', (), {'preprocessing': 'none', 'use_rho': False})()
        
        mu = np.array([1.0, 1.0, 1.0])
        policy = DeterministicNeuralPolicy(MockNet(), mu)
        
        rng = np.random.default_rng(42)
        
        Q = np.array([0.0, 0.0, 0.0])
        probs = policy(Q, rng)
        
        assert np.sum(probs) == 1.0
        assert np.sum(probs == 1.0) == 1

    def test_deterministic_ignores_rng(self):
        from gibbsq.utils.model_io import DeterministicNeuralPolicy
        
        class MockNet:
            def get_numpy_params(self):
                return [(np.eye(2), np.zeros(2))]
            
            def numpy_forward(self, x, params, config, **kwargs):
                return x
            
            config = type('Config', (), {'preprocessing': 'none', 'use_rho': False})()
        
        mu = np.array([1.0, 1.0])
        policy = DeterministicNeuralPolicy(MockNet(), mu)
        
        Q = np.array([1.0, 2.0])
        
        probs1 = policy(Q, np.random.default_rng(1))
        probs2 = policy(Q, np.random.default_rng(999))
        
        np.testing.assert_array_equal(probs1, probs2)

class TestStochasticNeuralPolicy:
    def test_stochastic_returns_valid_distribution(self):
        from gibbsq.utils.model_io import StochasticNeuralPolicy
        
        class MockNet:
            def get_numpy_params(self):
                return [(np.eye(3), np.zeros(3))]
            
            def numpy_forward(self, x, params, config, **kwargs):
                return x
            
            config = type('Config', (), {'preprocessing': 'none', 'use_rho': False})()
        
        mu = np.array([1.0, 1.0, 1.0])
        policy = StochasticNeuralPolicy(MockNet(), mu)
        
        rng = np.random.default_rng(42)
        Q = np.array([1.0, 2.0, 3.0])
        probs = policy(Q, rng)
        
        assert np.isclose(np.sum(probs), 1.0)
        assert np.all(probs >= 0)

    def test_stochastic_numerical_stability(self):
        from gibbsq.utils.model_io import StochasticNeuralPolicy
        
        class MockNet:
            def get_numpy_params(self):
                return [(np.eye(2), np.zeros(2))]
            
            def numpy_forward(self, x, params, config, **kwargs):
                return x
            
            config = type('Config', (), {'preprocessing': 'none', 'use_rho': False})()
        
        mu = np.array([1.0, 1.0])
        policy = StochasticNeuralPolicy(MockNet(), mu)
        
        rng = np.random.default_rng(42)
        
        Q = np.array([1000.0, 0.0])
        probs = policy(Q, rng)
        
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        assert np.isclose(np.sum(probs), 1.0)

class TestBuildNeuralEvalPolicy:
    def test_build_deterministic_policy(self):
        from gibbsq.utils.model_io import DeterministicNeuralPolicy, build_neural_eval_policy

        class MockNet:
            def get_numpy_params(self):
                return [(np.eye(2), np.zeros(2))]

            def numpy_forward(self, x, params, config, **kwargs):
                return x

            config = type('Config', (), {'preprocessing': 'none', 'use_rho': False})()

        policy = build_neural_eval_policy(MockNet(), np.array([1.0, 1.0]), mode="deterministic")
        assert isinstance(policy, DeterministicNeuralPolicy)

    def test_build_stochastic_policy(self):
        pytest.importorskip("jax")
        pytest.importorskip("equinox")
        from gibbsq.utils.model_io import NeuralSSAPolicy, build_neural_eval_policy
        import jax
        import jax.numpy as jnp
        import equinox as eqx

        class SimpleModel(eqx.Module):
            weight: jnp.ndarray

            def __init__(self, key):
                self.weight = jax.random.normal(key, (2,))

            def __call__(self, x, rho=None, mu=None):
                return self.weight * x

        policy = build_neural_eval_policy(
            SimpleModel(jax.random.PRNGKey(0)),
            np.array([1.0, 1.0]),
            mode="stochastic",
        )
        assert isinstance(policy, NeuralSSAPolicy)

    def test_build_invalid_mode_raises(self):
        from gibbsq.utils.model_io import build_neural_eval_policy

        class MockNet:
            config = type('Config', (), {'preprocessing': 'none', 'use_rho': False})()

        with pytest.raises(ValueError, match="Unknown neural evaluation mode"):
            build_neural_eval_policy(MockNet(), np.array([1.0]), mode="bad-mode")

class TestNeuralSSAPolicyCache:
    def test_cache_returns_same_result_for_same_input(self):
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
            
            def __call__(self, x, rho=None, mu=None):
                return self.weight * x
        
        key = jax.random.PRNGKey(42)
        model = SimpleModel(key)
        
        policy = NeuralSSAPolicy(model)
        
        rng = np.random.default_rng(42)
        Q = np.array([1.0, 2.0, 3.0])
        
        probs1 = policy(Q, rng)
        probs2 = policy(Q, rng)
        
        np.testing.assert_array_equal(probs1, probs2)

    def test_cache_different_inputs(self):
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
            
            def __call__(self, x, rho=None, mu=None):
                return self.weight * x
        
        key = jax.random.PRNGKey(42)
        model = SimpleModel(key)
        
        policy = NeuralSSAPolicy(model)
        
        rng = np.random.default_rng(42)
        
        Q1 = np.array([1.0, 2.0, 3.0])
        Q2 = np.array([3.0, 2.0, 1.0])
        
        probs1 = policy(Q1, rng)
        probs2 = policy(Q2, rng)
        
        assert not np.allclose(probs1, probs2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
