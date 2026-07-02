"""
Tests for mode-specific parameter precedence.

These tests verify that the mode-scoped parameter system works correctly
with the proper precedence hierarchy:
1. CLI arguments (highest)
2. mode.component.param
3. mode.param
4. component.param
5. param (base defaults, lowest)

Run with: pytest tests/params/test_params_mode_precedence.py -v
"""

import pytest

from dynvision.params import InitParams, TrainingParams, TestingParams
from tests.params.utils import (
    create_temp_config,
    temporary_init_paths,
    temporary_training_paths,
    temporary_testing_paths,
)


class TestInitModeOverrides:
    """Test init mode-specific parameter overrides."""

    def test_init_mode_data_override(self):
        """Test that init.data.* overrides work correctly."""
        config_data = {
            "use_ffcv": True,  # Level 5: base
            "data.use_ffcv": True,  # Level 4: component
            "init.use_ffcv": False,  # Level 3: mode (should win for init)
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_init_paths() as init_paths:
                params = InitParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **init_paths,
                        "init_with_pretrained": False,
                    },
                )

            assert (
                params.data.use_ffcv is False
            ), "init.use_ffcv should override base and component"
        finally:
            config_file.unlink()

    def test_init_mode_component_specific(self):
        """Test that init.data.* is more specific than init.*."""
        config_data = {
            "num_workers": 8,  # Level 5: base
            "init.num_workers": 4,  # Level 3: mode
            "init.data.num_workers": 0,  # Level 2: mode+component (should win)
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_init_paths() as init_paths:
                params = InitParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **init_paths,
                        "init_with_pretrained": False,
                    },
                )

            assert (
                params.data.num_workers == 0
            ), "init.data.num_workers should be most specific"
        finally:
            config_file.unlink()


class TestTrainingModeOverrides:
    """Test training mode-specific parameter overrides."""

    def test_train_mode_data_override(self):
        """Test that train.data.* overrides work correctly."""
        config_data = {
            "shuffle": False,  # Level 5: base
            "data.shuffle": False,  # Level 4: component
            "train.shuffle": True,  # Level 3: mode (should win for training)
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_training_paths() as training_paths:
                params = TrainingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs=dict(training_paths),
                )

            assert params.data.shuffle is True, "train.shuffle should override base"
        finally:
            config_file.unlink()


class TestTestingModeOverrides:
    """Test testing mode-specific parameter overrides."""

    def test_test_mode_batch_size_precedence(self):
        """Test full precedence hierarchy with batch_size."""
        config_data = {
            "batch_size": 128,  # Level 5: base
            "data.batch_size": 256,  # Level 4: component
            "test.batch_size": 512,  # Level 3: mode
            "test.data.batch_size": 1024,  # Level 2: mode+component (should win)
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **testing_paths,
                        "verbose": False,
                    },
                )

            assert (
                params.data.batch_size == 1024
            ), "test.data.batch_size should have highest priority"
        finally:
            config_file.unlink()

    def test_test_mode_trainer_devices(self):
        """Test mode+component override for trainer."""
        config_data = {
            "devices": 8,  # Level 5: base
            "trainer.devices": 4,  # Level 4: component
            "test.devices": 2,  # Level 3: mode
            "test.trainer.devices": 1,  # Level 2: mode+component (should win)
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **testing_paths,
                        "verbose": False,
                    },
                )

            assert (
                params.trainer.devices == 1
            ), "test.trainer.devices should be most specific"
        finally:
            config_file.unlink()

    def test_test_mode_use_ffcv_override(self):
        """Test that test.data.use_ffcv overrides base."""
        config_data = {
            "use_ffcv": True,  # Level 5: base (should win for base)
            "test.data.use_ffcv": False,  # Level 2: mode+component (should win for test)
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **testing_paths,
                        "verbose": False,
                    },
                )

            assert (
                params.data.use_ffcv is False
            ), "test.data.use_ffcv should override base"
        finally:
            config_file.unlink()


class TestWrongModeIgnored:
    """Test that parameters for other modes are ignored."""

    def test_train_params_ignored_by_init(self):
        """Test that train.* parameters are ignored by InitParams."""
        config_data = {
            "use_ffcv": True,  # Level 5: base (should win for init)
            "train.use_ffcv": False,  # Wrong mode (ignored by init)
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_init_paths() as init_paths:
                params = InitParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **init_paths,
                        "init_with_pretrained": False,
                    },
                )

            # train.use_ffcv should be ignored since mode is "init"
            assert (
                params.data.use_ffcv is True
            ), "Wrong mode override should be ignored"
        finally:
            config_file.unlink()

    def test_init_params_ignored_by_test(self):
        """Test that init.* parameters are ignored by TestingParams."""
        config_data = {
            "num_workers": 1,  # Level 5: base (should win for test; 1 is always <= CPU count)
            "init.num_workers": 0,  # Wrong mode (ignored by test)
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **testing_paths,
                        "verbose": False,
                    },
                )

            # init.num_workers should be ignored since mode is "test"
            assert (
                params.data.num_workers == 1
            ), "Wrong mode override should be ignored"
        finally:
            config_file.unlink()


class TestUnscopedParameterRouting:
    """Test that unscoped parameters are still routed correctly."""

    def test_unscoped_model_param(self):
        """Test that unscoped model parameters route correctly."""
        config_data = {
            "n_classes": 100,  # Unscoped, should route to model
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **testing_paths,
                        "verbose": False,
                    },
                )

            assert (
                params.model.n_classes == 100
            ), "Unscoped params should still route to components"
        finally:
            config_file.unlink()


class TestCLIOverride:
    """Test that CLI overrides work at all scope levels."""

    def test_cli_overrides_mode_component(self):
        """Test that CLI args override even mode+component config."""
        config_data = {
            "batch_size": 128,  # Level 5: base
            "test.data.batch_size": 1024,  # Level 2: mode+component
        }

        config_file = create_temp_config(config_data)
        try:
            # CLI override should win
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **testing_paths,
                        "verbose": False,
                        "test.data.batch_size": 2048,  # CLI override (Level 1)
                    },
                )

            assert (
                params.data.batch_size == 2048
            ), "CLI should override all config levels"
        finally:
            config_file.unlink()


class TestBackwardCompatibility:
    """Test that existing unscoped parameters still work."""

    def test_unscoped_params_work(self):
        """Test that traditional unscoped parameters continue to work."""
        config_data = {
            "batch_size": 256,
            "shuffle": False,
            "num_workers": 4,
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **testing_paths,
                        "verbose": False,
                    },
                )

            assert params.data.batch_size == 256, "Unscoped batch_size should work"
            assert params.data.shuffle is False, "Unscoped shuffle should work"
            assert params.data.num_workers == 4, "Unscoped num_workers should work"
        finally:
            config_file.unlink()


class TestModeShortcutActivation:
    """Ensure short-form mode toggles (e.g., debug=True) activate modes."""

    def test_debug_shortcut_from_config(self):
        """Setting debug=True in config should enable the debug mode payload."""

        config_data = {
            "debug": True,
        }

        config_file = create_temp_config(config_data)
        try:
            with temporary_training_paths() as training_paths:
                params = TrainingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs=dict(training_paths),
                )

            assert (
                params.trainer.check_val_every_n_epoch == 1
            ), "Debug mode payload should tighten validation cadence"
            assert (
                params.trainer.accumulate_grad_batches == 1
            ), "Debug mode payload should override grad accumulation"
        finally:
            config_file.unlink()

    def test_debug_shortcut_from_cli_override(self):
        """Providing debug via CLI/override kwargs should also enable the mode."""

        config_file = create_temp_config({})
        try:
            with temporary_training_paths() as training_paths:
                params = TrainingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={
                        **training_paths,
                        "debug": True,
                    },
                )

            assert (
                params.trainer.enable_progress_bar is True
            ), "Debug mode payload should flip progress bar"
            assert (
                params.trainer.log_every_n_steps == 1
            ), "Debug mode payload should tighten logging frequency"
        finally:
            config_file.unlink()


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
