"""Tests for DataParams transform parameter derivation and validation.

Tests cover:
- Automatic derivation of transform_backend, transform_context, transform_preset
- Automatic derivation of target_data_name, target_data_group
- Integration with get_dataloader_kwargs
"""

import pytest
import torch
from dynvision.params.data_params import DataParams


class TestTransformDerivation:
    """Test automatic derivation of transform parameters."""

    def get_minimal_params(self, **overrides):
        """Get minimal DataParams with required fields."""
        defaults = {
            "seed": 42,
            "log_level": "INFO",
            "data_name": "imagenette",
            "data_group": "all",
            "train": True,
            "use_ffcv": False,
            "batch_size": 32,
            "num_workers": 4,
            "persistent_workers": False,
            "drop_last": True,
            "train_ratio": 0.9,
            "pixel_range": "0-1",
            "data_timesteps": 1,
            "non_input_value": 0,
            "non_label_index": -1,
            "use_distributed": False,
            "encoding": "image",
            "writer_mode": "smart",
            "max_resolution": 256,
            "compress_probability": 0.5,
            "jpeg_quality": 90,
            "chunksize": 100,
            "page_size": 4096,
            "precision": "32",
            "batches_ahead": 2,
            "order": "RANDOM",
            "pin_memory": False,
            "shuffle": True,
        }
        defaults.update(overrides)
        return DataParams(**defaults)

    def test_derive_torch_backend(self):
        """Test deriving torch backend when use_ffcv=False."""
        params = self.get_minimal_params(use_ffcv=False)
        assert params.transform_backend == "torch"

    def test_derive_ffcv_backend(self):
        """Test deriving FFCV backend when use_ffcv=True."""
        params = self.get_minimal_params(use_ffcv=True)
        assert params.transform_backend == "ffcv"

    def test_derive_train_context(self):
        """Test deriving train context when train=True."""
        params = self.get_minimal_params(train=True)
        assert params.transform_context == "train"

    def test_derive_test_context(self):
        """Test deriving test context when train=False."""
        params = self.get_minimal_params(train=False)
        assert params.transform_context == "test"

    def test_derive_preset_from_data_name(self):
        """Test deriving preset from data_name."""
        params = self.get_minimal_params(data_name="mnist")
        assert params.transform_preset == "mnist"

        params = self.get_minimal_params(data_name="imagenette")
        assert params.transform_preset == "imagenette"

    def test_explicit_preset_override(self):
        """Test explicit preset overrides derived value."""
        params = self.get_minimal_params(
            data_name="mnist", transform_preset="custom_preset"
        )
        assert params.transform_preset == "custom_preset"

    def test_derive_target_data_name(self):
        """Test deriving target_data_name from data_name."""
        params = self.get_minimal_params(data_name="mnist")
        assert params.target_data_name == "mnist"

    def test_derive_target_data_group_train(self):
        """Test deriving target_data_group='all' for training."""
        params = self.get_minimal_params(train=True, data_group="specific_group")
        assert params.target_data_group == "all"

    def test_derive_target_data_group_test(self):
        """Test deriving target_data_group from data_group for testing."""
        params = self.get_minimal_params(train=False, data_group="01")
        assert params.target_data_group == "01"


class TestTransformKwargs:
    """Test transform parameters in dataloader kwargs."""

    def get_minimal_params(self, **overrides):
        """Get minimal DataParams with required fields."""
        defaults = {
            "seed": 42,
            "log_level": "INFO",
            "data_name": "imagenette",
            "data_group": "all",
            "train": True,
            "use_ffcv": False,
            "batch_size": 32,
            "num_workers": 4,
            "persistent_workers": False,
            "drop_last": True,
            "train_ratio": 0.9,
            "pixel_range": "0-1",
            "data_timesteps": 1,
            "non_input_value": 0,
            "non_label_index": -1,
            "use_distributed": False,
            "encoding": "image",
            "writer_mode": "smart",
            "max_resolution": 256,
            "compress_probability": 0.5,
            "jpeg_quality": 90,
            "chunksize": 100,
            "page_size": 4096,
            "precision": "32",
            "batches_ahead": 2,
            "order": "RANDOM",
            "pin_memory": False,
            "shuffle": True,
        }
        defaults.update(overrides)
        return DataParams(**defaults)

    def test_transform_params_in_kwargs(self):
        """Test transform parameters are included in dataloader kwargs."""
        params = self.get_minimal_params()
        kwargs = params.get_dataloader_kwargs()

        assert "transform_backend" in kwargs
        assert "transform_context" in kwargs
        assert "transform_preset" in kwargs
        assert "target_data_name" in kwargs
        assert "target_data_group" in kwargs

    def test_kwargs_values_match_params(self):
        """Test kwargs values match parameter values."""
        params = self.get_minimal_params(
            use_ffcv=True, train=False, data_name="mnist", data_group="01"
        )
        kwargs = params.get_dataloader_kwargs()

        assert kwargs["transform_backend"] == "ffcv"
        assert kwargs["transform_context"] == "test"
        assert kwargs["transform_preset"] == "mnist"
        assert kwargs["target_data_name"] == "mnist"
        assert kwargs["target_data_group"] == "01"


class TestTransformScenarios:
    """Test realistic transform configuration scenarios."""

    def get_minimal_params(self, **overrides):
        """Get minimal DataParams with required fields."""
        defaults = {
            "seed": 42,
            "log_level": "INFO",
            "data_name": "imagenette",
            "data_group": "all",
            "train": True,
            "use_ffcv": False,
            "batch_size": 32,
            "num_workers": 4,
            "persistent_workers": False,
            "drop_last": True,
            "train_ratio": 0.9,
            "pixel_range": "0-1",
            "data_timesteps": 1,
            "non_input_value": 0,
            "non_label_index": -1,
            "use_distributed": False,
            "encoding": "image",
            "writer_mode": "smart",
            "max_resolution": 256,
            "compress_probability": 0.5,
            "jpeg_quality": 90,
            "chunksize": 100,
            "page_size": 4096,
            "precision": "32",
            "batches_ahead": 2,
            "order": "RANDOM",
            "pin_memory": False,
            "shuffle": True,
        }
        defaults.update(overrides)
        return DataParams(**defaults)

    def test_torch_train_imagenette_scenario(self):
        """Test PyTorch training on Imagenette."""
        params = self.get_minimal_params(
            use_ffcv=False, train=True, data_name="imagenette", data_group="all"
        )

        assert params.transform_backend == "torch"
        assert params.transform_context == "train"
        assert params.transform_preset == "imagenette"
        assert params.target_data_name == "imagenette"
        assert params.target_data_group == "all"

    def test_ffcv_train_mnist_scenario(self):
        """Test FFCV training on MNIST."""
        params = self.get_minimal_params(
            use_ffcv=True, train=True, data_name="mnist", data_group="all"
        )

        assert params.transform_backend == "ffcv"
        assert params.transform_context == "train"
        assert params.transform_preset == "mnist"
        assert params.target_data_name == "mnist"
        assert params.target_data_group == "all"

    def test_torch_test_imagenette_one_scenario(self):
        """Test PyTorch testing on Imagenette 'one' subset."""
        params = self.get_minimal_params(
            use_ffcv=False, train=False, data_name="imagenette", data_group="one"
        )

        assert params.transform_backend == "torch"
        assert params.transform_context == "test"
        assert params.transform_preset == "imagenette"
        assert params.target_data_name == "imagenette"
        assert params.target_data_group == "one"

    def test_ffcv_test_scenario(self):
        """Test FFCV testing scenario."""
        params = self.get_minimal_params(
            use_ffcv=True, train=False, data_name="mnist", data_group="01"
        )

        assert params.transform_backend == "ffcv"
        assert params.transform_context == "test"
        assert params.transform_preset == "mnist"
        assert params.target_data_name == "mnist"
        assert params.target_data_group == "01"

    def test_custom_preset_scenario(self):
        """Test custom preset override scenario."""
        params = self.get_minimal_params(
            use_ffcv=False,
            train=True,
            data_name="mnist",
            transform_preset="custom_augmentation",
        )

        assert params.transform_backend == "torch"
        assert params.transform_context == "train"
        assert params.transform_preset == "custom_augmentation"


class TestEffectiveDtype:
    """Regression tests for DataParams.effective_dtype.

    effective_dtype previously did a string-keyed dict lookup on self.dtype,
    but validate_dtype already normalizes self.dtype to a torch.dtype, so the
    lookup always missed and silently fell back to torch.float32 regardless
    of the configured dtype.
    """

    def get_minimal_params(self, **overrides):
        return TestTransformDerivation().get_minimal_params(**overrides)

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            ("float16", torch.float16),
            ("float32", torch.float32),
            ("float64", torch.float64),
            ("bfloat16", torch.bfloat16),
            ("int8", torch.int8),
        ],
    )
    def test_effective_dtype_matches_configured_dtype(self, dtype, expected):
        params = self.get_minimal_params(dtype=dtype)
        assert params.effective_dtype == expected

    def test_effective_dtype_none_when_dtype_unset(self):
        params = self.get_minimal_params()
        assert params.dtype is None
        assert params.effective_dtype is None
