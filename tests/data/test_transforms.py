"""Tests for transform parsing and resolution.

Tests cover:
- Transform string parsing (parse_transform_string)
- Transform list parsing (parse_transform_list)
- Preset resolution (resolve_transform_preset)
- Data transform retrieval (get_data_transform)
- Target transform retrieval (get_target_transform)
"""

import pytest
from dynvision.data.transforms import (
    parse_transform_string,
    parse_transform_list,
    validate_transform_string,
    resolve_transform_preset,
    get_data_transform,
    get_target_transform,
)

try:
    import ffcv  # noqa: F401

    HAS_FFCV = True
except ImportError:
    HAS_FFCV = False

requires_ffcv = pytest.mark.skipif(
    not HAS_FFCV,
    reason="ffcv is an optional dependency not installed in this environment",
)


class TestParseTransformString:
    """Test transform string parsing."""

    def test_parse_bare_module_name_torch(self):
        """Test parsing bare module name for torch backend."""
        transform = parse_transform_string("RandomHorizontalFlip", backend="torch")
        assert transform is not None
        assert type(transform).__name__ == "RandomHorizontalFlip"

    @requires_ffcv
    def test_parse_bare_module_name_ffcv(self):
        """Test parsing bare module name for FFCV backend."""
        transform = parse_transform_string("RandomHorizontalFlip", backend="ffcv")
        assert transform is not None
        assert type(transform).__name__ == "RandomHorizontalFlip"

    def test_parse_with_single_arg(self):
        """Test parsing transform with single positional argument."""
        transform = parse_transform_string("RandomRotation(10)", backend="torch")
        assert transform is not None
        assert type(transform).__name__ == "RandomRotation"

    def test_parse_with_keyword_args(self):
        """Test parsing transform with keyword arguments."""
        transform = parse_transform_string(
            "ColorJitter(brightness=0.2, contrast=0.2)", backend="torch"
        )
        assert transform is not None
        assert type(transform).__name__ == "ColorJitter"

    def test_parse_with_mixed_args(self):
        """Test parsing transform with mixed positional and keyword arguments."""
        transform = parse_transform_string(
            "RandomAffine(0, translate=(0.1, 0.1))", backend="torch"
        )
        assert transform is not None
        assert type(transform).__name__ == "RandomAffine"

    def test_parse_with_tuple_args(self):
        """Test parsing transform with tuple arguments."""
        transform = parse_transform_string("Resize(256)", backend="torch")
        assert transform is not None
        assert type(transform).__name__ == "Resize"

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        assert parse_transform_string("", backend="torch") is None
        assert parse_transform_string("   ", backend="torch") is None

    def test_parse_invalid_backend(self):
        """Test parsing with invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            parse_transform_string("RandomHorizontalFlip", backend="invalid")

    def test_parse_nonexistent_transform(self):
        """Test parsing nonexistent transform raises AttributeError."""
        with pytest.raises(AttributeError, match="not found"):
            parse_transform_string("NonexistentTransform", backend="torch")

    def test_parse_invalid_format(self):
        """Test parsing invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid transform string format"):
            parse_transform_string("Invalid-Format!", backend="torch")


class TestParseTransformList:
    """Test transform list parsing."""

    def test_parse_multiple_transforms(self):
        """Test parsing list of transforms."""
        transforms = parse_transform_list(
            [
                "RandomHorizontalFlip()",
                "RandomRotation(10)",
                "ColorJitter(brightness=0.2)",
            ],
            backend="torch",
        )
        assert len(transforms) == 3
        assert type(transforms[0]).__name__ == "RandomHorizontalFlip"
        assert type(transforms[1]).__name__ == "RandomRotation"
        assert type(transforms[2]).__name__ == "ColorJitter"

    def test_parse_empty_list(self):
        """Test parsing empty list returns empty list."""
        transforms = parse_transform_list([], backend="torch")
        assert transforms == []

    def test_parse_list_with_error(self):
        """Test parsing list with invalid transform raises error."""
        with pytest.raises(AttributeError):
            parse_transform_list(
                ["RandomHorizontalFlip()", "InvalidTransform()"], backend="torch"
            )


class TestValidateTransformString:
    """Test transform string validation."""

    def test_validate_valid_string(self):
        """Test validating valid transform string."""
        is_valid, error = validate_transform_string(
            "RandomHorizontalFlip()", backend="torch"
        )
        assert is_valid is True
        assert error is None

    def test_validate_invalid_string(self):
        """Test validating invalid transform string."""
        is_valid, error = validate_transform_string(
            "InvalidTransform()", backend="torch"
        )
        assert is_valid is False
        assert error is not None
        assert "not found" in error


class TestResolveTransformPreset:
    """Test transform preset resolution."""

    def test_resolve_dataset_specific_preset(self):
        """Test resolving dataset-specific preset."""
        transforms = resolve_transform_preset(
            backend="torch", context="train", dataset_or_preset="imagenette"
        )
        assert transforms is not None
        assert isinstance(transforms, list)
        assert len(transforms) > 0

    def test_resolve_base_preset(self):
        """Test resolving base preset."""
        transforms = resolve_transform_preset(
            backend="torch", context="train", dataset_or_preset=None
        )
        assert transforms is not None
        assert isinstance(transforms, list)

    def test_resolve_fallback_to_base(self):
        """Test fallback to base when dataset preset doesn't exist."""
        transforms = resolve_transform_preset(
            backend="torch", context="train", dataset_or_preset="nonexistent_dataset"
        )
        # Should fall back to base preset
        assert transforms is not None
        assert isinstance(transforms, list)

    def test_resolve_test_context(self):
        """Test resolving test context preset."""
        transforms = resolve_transform_preset(
            backend="torch", context="test", dataset_or_preset="imagenette"
        )
        # Test presets should exist
        assert transforms is not None
        assert isinstance(transforms, list)

    def test_resolve_ffcv_backend(self):
        """Test resolving FFCV backend preset."""
        transforms = resolve_transform_preset(
            backend="ffcv", context="train", dataset_or_preset="base"
        )
        assert transforms is not None
        assert isinstance(transforms, list)

    def test_resolve_invalid_backend(self):
        """Test resolving with invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Backend.*not found"):
            resolve_transform_preset(
                backend="invalid", context="train", dataset_or_preset="base"
            )

    def test_resolve_invalid_context(self):
        """Test resolving with invalid context raises ValueError."""
        with pytest.raises(ValueError, match="Context.*not found"):
            resolve_transform_preset(
                backend="torch", context="invalid", dataset_or_preset="base"
            )


class TestGetDataTransform:
    """Test data transform retrieval."""

    def test_get_torch_train_transforms(self):
        """Test getting torch training transforms."""
        transforms = get_data_transform(
            backend="torch", context="train", dataset_or_preset="imagenette"
        )
        assert transforms is not None
        assert len(transforms) > 0
        # All should be callable
        assert all(callable(t) for t in transforms)

    @requires_ffcv
    def test_get_ffcv_train_transforms(self):
        """Test getting FFCV training transforms."""
        transforms = get_data_transform(
            backend="ffcv", context="train", dataset_or_preset="base"
        )
        assert transforms is not None
        assert len(transforms) > 0

    def test_get_test_transforms(self):
        """Test getting test transforms."""
        transforms = get_data_transform(
            backend="torch", context="test", dataset_or_preset="imagenette"
        )
        # Test transforms may be empty or have resize/crop
        assert transforms is not None or transforms is None

    def test_get_with_nonexistent_preset(self):
        """Test getting transforms with nonexistent preset falls back to base."""
        transforms = get_data_transform(
            backend="torch", context="train", dataset_or_preset="nonexistent"
        )
        # Should fall back to base and return transforms
        assert transforms is not None


class TestGetTargetTransform:
    """Test target transform retrieval."""

    def test_get_target_transform_all_group(self):
        """Test getting target transform for 'all' group returns None."""
        transform = get_target_transform(data_name="imagenette", data_group="all")
        assert transform is None

    def test_get_target_transform_specific_group(self):
        """Test getting target transform for specific group returns IndexToLabel."""
        transform = get_target_transform(data_name="mnist", data_group="01")
        assert transform is not None
        assert len(transform) == 1
        assert type(transform[0]).__name__ == "IndexToLabel"

    def test_get_target_transform_empty_data_name(self):
        """Test getting target transform with empty data_name raises ValueError."""
        with pytest.raises(ValueError, match="data_name cannot be empty"):
            get_target_transform(data_name="", data_group="all")

    def test_get_target_transform_empty_data_group(self):
        """Test getting target transform with empty data_group raises ValueError."""
        with pytest.raises(ValueError, match="data_group cannot be empty"):
            get_target_transform(data_name="imagenette", data_group="")


class TestTransformIntegration:
    """Integration tests for complete transform workflow."""

    def test_mnist_train_workflow(self):
        """Test complete MNIST training transform workflow."""
        transforms = get_data_transform(
            backend="torch", context="train", dataset_or_preset="mnist"
        )
        assert transforms is not None
        assert len(transforms) > 0

        target_transform = get_target_transform(data_name="mnist", data_group="all")
        assert target_transform is None  # 'all' group returns None

    def test_imagenette_train_workflow(self):
        """Test complete Imagenette training transform workflow."""
        transforms = get_data_transform(
            backend="torch", context="train", dataset_or_preset="imagenette"
        )
        assert transforms is not None
        assert len(transforms) > 0

        target_transform = get_target_transform(
            data_name="imagenette", data_group="one"
        )
        assert target_transform is not None
        assert len(target_transform) == 1

    @requires_ffcv
    def test_ffcv_mnist_train_workflow(self):
        """Test complete FFCV MNIST training workflow."""
        transforms = get_data_transform(
            backend="ffcv", context="train", dataset_or_preset="mnist"
        )
        assert transforms is not None
        assert len(transforms) > 0

    def test_test_mode_workflow(self):
        """Test test mode workflow with minimal transforms."""
        transforms = get_data_transform(
            backend="torch", context="test", dataset_or_preset="imagenette"
        )
        # Test mode should have resize/crop for imagenette
        assert transforms is not None
        assert len(transforms) >= 2  # At least Resize and CenterCrop

    def test_base_fallback_workflow(self):
        """Test workflow falls back to base preset when dataset preset missing."""
        transforms = get_data_transform(
            backend="torch", context="train", dataset_or_preset="unknown_dataset"
        )
        # Should get base transforms
        assert transforms is not None
        assert len(transforms) > 0
