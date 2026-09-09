"""Regression tests for DtypeDeviceCoordinator.map_dtype.

Encodes the bug where map_dtype normalized string dtype spellings by
unconditionally stripping the substring "float", which turned "bfloat16" /
"torch.bfloat16" into "b16" and made resolve_dtype() fall back to the
default float32 instead of resolving bfloat16.
"""

import pytest
import torch

from dynvision.base.coordination import DtypeDeviceCoordinator


class TestMapDtype:
    @pytest.mark.parametrize(
        "spelling,expected",
        [
            ("torch.float32", torch.float32),
            ("float32", torch.float32),
            ("32", torch.float32),
            ("torch.float16", torch.float16),
            ("float16", torch.float16),
            ("16", torch.float16),
            ("torch.float64", torch.float64),
            ("float64", torch.float64),
            ("64", torch.float64),
            ("bfloat16", torch.bfloat16),
            ("torch.bfloat16", torch.bfloat16),
            ("bf16", torch.bfloat16),
            ("BFloat16", torch.bfloat16),
        ],
    )
    def test_string_spellings_resolve_correctly(self, spelling, expected):
        coordinator = DtypeDeviceCoordinator()
        assert coordinator.map_dtype(spelling) == expected

    def test_torch_dtype_passthrough(self):
        coordinator = DtypeDeviceCoordinator()
        assert coordinator.map_dtype(torch.bfloat16) == torch.bfloat16
        assert coordinator.map_dtype(torch.float16) == torch.float16

    def test_none_passthrough(self):
        coordinator = DtypeDeviceCoordinator()
        assert coordinator.map_dtype(None) is None

    def test_bfloat16_target_dtype_survives_init(self):
        """Explicit bfloat16 target dtype must not silently become float32."""
        coordinator = DtypeDeviceCoordinator(target_dtype="bfloat16")
        assert coordinator._target_dtype == torch.bfloat16
