"""Tests for `recurrence_target` and `data_loader` as declared parameter fields.

Both were previously reachable only as *aliases* whose target was not a declared
field. The value was stored via ``extra="allow"`` and reached its consumer only
because the component's kwargs builder merges ``__pydantic_extra__``
(``ModelParams.get_model_kwargs``) or because the consumer read the attribute
directly (``DataModule.setup_dataloader`` reads ``config.data.data_loader``).
Nothing validated them, and neither appeared in the summary or in provenance.

Declaring them makes them first-class. Two things had to move with the
declaration:

* ``get_model_kwargs`` did *not* list ``recurrence_target``, so it reached the
  model only through the extras merge. Declaring the field without adding it
  there would have silently stopped forwarding it.
* ``None`` is now a legitimate value for ``data_loader`` (the sentinel for "not
  set"), so ``get_data_loader_class`` has to report that instead of raising
  ``TypeError`` from a substring test on ``None``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dynvision.data.dataloader import get_data_loader_class
from dynvision.params import TestingParams, TrainingParams
from dynvision.params.data_params import DataParams
from dynvision.params.model_params import ModelParams

from .utils import (
    create_temp_config,
    temporary_testing_paths,
    temporary_training_paths,
)


class TestRecurrenceTargetField:
    """`recurrence_target` is a declared, validated ModelParams field."""

    def test_is_a_declared_field(self):
        assert "recurrence_target" in ModelParams.model_fields

    def test_alias_resolves_through_the_composite(self):
        for composite in (TrainingParams, TestingParams):
            assert composite.get_aliases()["rctarget"] == "model.recurrence_target"

    def test_valid_values(self):
        for value in ("output", "input", "middle"):
            params = ModelParams(seed=0, log_level="INFO", recurrence_target=value)
            assert params.recurrence_target == value

    def test_case_is_normalised(self):
        """BaseParams normalises Literal members case-insensitively."""
        params = ModelParams(seed=0, log_level="INFO", recurrence_target="Middle")
        assert params.recurrence_target == "middle"

    def test_invalid_value_is_rejected(self):
        with pytest.raises(ValidationError):
            ModelParams(seed=0, log_level="INFO", recurrence_target="sideways")

    def test_is_forwarded_to_the_model(self):
        """The regression guard: get_model_kwargs must carry the field.

        Before it was declared, it reached the model only via the extras merge in
        get_model_kwargs. Adding the field without the kwargs entry would drop it.
        """
        params = ModelParams(
            seed=0, log_level="INFO", dt=2.0, tau=5.0, recurrence_target="middle"
        )
        assert params.get_model_kwargs()["recurrence_target"] == "middle"

    def test_is_not_stored_as_an_extra(self):
        params = ModelParams(seed=0, log_level="INFO", recurrence_target="middle")
        extras = getattr(params, "__pydantic_extra__", None) or {}
        assert "recurrence_target" not in extras

    def test_unset_is_filtered_out_so_the_model_default_applies(self):
        params = ModelParams(seed=0, log_level="INFO", dt=2.0, tau=5.0)
        assert "recurrence_target" not in params.get_model_kwargs()

    def test_is_listed_in_the_summary_sections(self):
        """Being a declared field makes it visible to the summary output."""
        connectivity = ModelParams.summary_sections["Connectivity"]
        assert "recurrence_target" in [item.source for item in connectivity]

    def test_reaches_the_model_through_a_training_config(self):
        config_file = create_temp_config({"rctarget": "input"})
        try:
            with temporary_training_paths() as training_paths:
                params = TrainingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs=dict(training_paths),
                )
            assert params.model.recurrence_target == "input"
            assert params.model.get_model_kwargs()["recurrence_target"] == "input"
        finally:
            config_file.unlink()


class TestDataLoaderField:
    """`data_loader` is a declared DataParams field."""

    def test_is_a_declared_field(self):
        assert "data_loader" in DataParams.model_fields

    def test_alias_resolves_through_testing_params(self):
        assert TestingParams.get_aliases()["data_loader"] == "data.data_loader"

    def test_reaches_the_data_component_through_a_testing_config(self):
        config_file = create_temp_config({"data_loader": "StimulusDuration"})
        try:
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={**testing_paths, "verbose": False},
                )
            assert params.data.data_loader == "StimulusDuration"
        finally:
            config_file.unlink()

    def test_is_not_stored_as_an_extra(self):
        config_file = create_temp_config({"data_loader": "StimulusNoise"})
        try:
            with temporary_testing_paths() as testing_paths:
                params = TestingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={**testing_paths, "verbose": False},
                )
            extras = getattr(params.data, "__pydantic_extra__", None) or {}
            assert "data_loader" not in extras
        finally:
            config_file.unlink()


class TestGetDataLoaderClass:
    """The loader registry lookup reports an unset name clearly."""

    def test_unset_name_raises_a_clear_error(self):
        """None is now a legitimate sentinel for `data_loader`, so say so."""
        with pytest.raises(ValueError, match="DataLoader class name is required"):
            get_data_loader_class(None)

    def test_empty_name_raises_a_clear_error(self):
        with pytest.raises(ValueError, match="DataLoader class name is required"):
            get_data_loader_class("")

    def test_error_lists_the_registry(self):
        with pytest.raises(ValueError) as excinfo:
            get_data_loader_class(None)
        assert "StandardDataLoader" in str(excinfo.value)

    def test_resolves_with_and_without_the_suffix(self):
        assert get_data_loader_class("Standard") is get_data_loader_class(
            "StandardDataLoader"
        )

    def test_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown DataLoader class"):
            get_data_loader_class("NoSuchLoader")
