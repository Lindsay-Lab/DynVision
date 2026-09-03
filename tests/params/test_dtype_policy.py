"""Tests for the central dtype policy and the dtype-coordination regression fix.

These encode the bug that caused the cluster crash: unscoped ``precision`` in a
config was routed to ``trainer.precision`` only (via a TestingParams alias),
leaving ``data.precision`` None and silently deriving float16 data while the
model stayed float32.
"""

import logging
import warnings

import pytest
import torch

from dynvision.utils.dtype_policy import (
    DEFAULT_DTYPE,
    DtypePolicy,
    normalize_precision,
    resolve_dtype,
)
from dynvision.params import TestingParams, TrainingParams
from tests.params.utils import (
    create_temp_config,
    temporary_testing_paths,
    temporary_training_paths,
)


class TestResolveDtype:
    @pytest.mark.parametrize(
        "precision,expected",
        [
            ("32", torch.float32),
            ("32-true", torch.float32),
            (32, torch.float32),
            ("16", torch.float16),
            ("16-true", torch.float16),
            (16, torch.float16),
            ("64", torch.float64),
            ("64-true", torch.float64),
            (64, torch.float64),
            ("bf16", torch.bfloat16),
            ("bf16-true", torch.bfloat16),
            # Mixed precision keeps fp32 master weights.
            ("16-mixed", torch.float32),
            ("bf16-mixed", torch.float32),
        ],
    )
    def test_known_precisions(self, precision, expected):
        assert resolve_dtype(precision) == expected

    @pytest.mark.parametrize("unknown", [None, "", "nonsense", 8, True, False])
    def test_unknown_precision_warns_and_defaults_to_float32(self, unknown, caplog):
        with caplog.at_level(logging.WARNING):
            result = resolve_dtype(unknown)
        assert result == torch.float32
        assert result == DEFAULT_DTYPE
        assert any("Unknown precision" in r.message for r in caplog.records)

    def test_no_warning_when_suppressed(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = resolve_dtype("bogus", warn_fallback=False)
        assert result == torch.float32

    def test_normalize_precision_torch_dtype(self):
        assert normalize_precision(torch.float32) == "32"


class TestDtypePolicy:
    def test_from_precision_defaults_all_to_trainer(self):
        policy = DtypePolicy.from_precision("32")
        assert policy.trainer_dtype == torch.float32
        assert policy.model_dtype == torch.float32
        assert policy.data_dtype == torch.float32
        assert policy.checkpoint_dtype == torch.float32

    def test_data_dtype_override(self):
        policy = DtypePolicy.from_precision("32", data_dtype=torch.float16)
        assert policy.data_dtype == torch.float16
        assert policy.trainer_dtype == torch.float32


class TestCoordinateComponentDtypes:
    def test_aligns_data_to_trainer(self):
        config = create_temp_config(
            {
                "precision": "32",
                "data": {"dtype": "float16"},
            }
        )
        try:
            with temporary_training_paths() as training_paths:
                params = TrainingParams.from_cli_and_config(
                    config_path=str(config),
                    override_kwargs=dict(training_paths),
                )
            assert params.data.dtype == torch.float32
            assert params.data.dtype == params.trainer.get_effective_dtype()
        finally:
            config.unlink()


class TestPrecisionRoutesToDataDtype:
    """Regression test for the cluster crash (unscoped precision -> data dtype)."""

    def test_testing_params_data_dtype_matches_trainer(self):
        config = create_temp_config({"precision": "32"})
        try:
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config),
                    override_kwargs={
                        **testing_paths,
                        "verbose": False,
                    },
                )
            assert params.data.dtype == torch.float32
            assert params.data.dtype == params.trainer.get_effective_dtype()
        finally:
            config.unlink()

    def test_training_params_data_dtype_matches_trainer(self):
        config = create_temp_config({"precision": "32"})
        try:
            with temporary_training_paths() as training_paths:
                params = TrainingParams.from_cli_and_config(
                    config_path=str(config),
                    override_kwargs=dict(training_paths),
                )
            assert params.data.dtype == torch.float32
            assert params.data.dtype == params.trainer.get_effective_dtype()
        finally:
            config.unlink()

    def test_mixed_precision_stays_float32(self):
        config = create_temp_config({"precision": "bf16-mixed"})
        try:
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config),
                    override_kwargs={
                        **testing_paths,
                        "verbose": False,
                    },
                )
            assert params.data.dtype == torch.float32
        finally:
            config.unlink()
