"""Tests for the dt-multiple constraint on millisecond durations.

DynVision expresses delays (``t_feedforward``, ``t_recurrence``,
``t_feedback``, ``t_skip``) and the loss reaction window
(``loss_reaction_time``) in milliseconds, but integrates them on a fixed grid
of ``dt`` millisecond steps. Every one of them must therefore be a whole number
of steps.

These tests cover the three claims that make the constraint meaningful:

1. A duration off the grid is rejected with a message naming the parameter
   (previously it was rounded silently, and the conversions disagreed).
2. A duration on the grid converts to the integer step count it names, even
   when floating-point division is inexact.
3. Durations that no computation depends on do not cause a rejection.
"""

import pytest
from pydantic import ValidationError

from dynvision.models.dyrcnn import DyRCNNx2
from dynvision.params.model_params import ModelParams
from dynvision.utils.timing import (
    is_integer_multiple,
    multiple_error_message,
    to_timesteps,
)

#: Every ModelParams field that is measured in milliseconds and integrated on dt.
DT_MULTIPLE_FIELDS = (
    "t_feedforward",
    "t_recurrence",
    "t_feedback",
    "t_skip",
    "loss_reaction_time",
)


def make_params(**overrides) -> ModelParams:
    """Build ModelParams with the required BaseParams fields supplied."""
    base = {"seed": 0, "log_level": "INFO", "dt": 2.0, "tau": 10.0}
    return ModelParams(**{**base, **overrides})


def make_model(**overrides) -> DyRCNNx2:
    """Build a minimal DyRCNNx2 for model-level validation tests."""
    base = {
        "n_timesteps": 10,
        "n_classes": 10,
        "input_dims": (10, 3, 32, 32),
        "dt": 2.0,
        "tau": 10.0,
        "recurrence_type": "none",
        "shuffle_presentation_pattern": False,
    }
    return DyRCNNx2(**{**base, **overrides})


class TestIsIntegerMultiple:
    """The predicate itself, including its tolerance for float division."""

    def test_exact_multiples(self):
        assert is_integer_multiple(6.0, 3.0)
        assert is_integer_multiple(0.0, 3.0)
        assert is_integer_multiple(3.0, 3.0)
        assert is_integer_multiple(30.0, 2.0)

    def test_non_multiples(self):
        assert not is_integer_multiple(4.0, 3.0)
        assert not is_integer_multiple(5.0, 2.0)
        assert not is_integer_multiple(10.0, 3.0)

    def test_tolerates_inexact_float_division(self):
        """A bare remainder test rejects these valid pairs.

        ``0.3 % 0.1`` is 0.09999999999999998 and ``1.5 % 0.1`` is
        0.09999999999999986, so both would fail an exact modulo check.
        """
        assert (0.3 % 0.1) != 0, "precondition: the remainder test is inexact"
        assert (1.5 % 0.1) != 0, "precondition: the remainder test is inexact"
        assert is_integer_multiple(0.3, 0.1)
        assert is_integer_multiple(1.5, 0.1)
        assert is_integer_multiple(1.5, 0.3)

    def test_rejects_non_positive_dt(self):
        for bad_dt in (0.0, -2.0):
            with pytest.raises(ValueError, match="dt must be positive"):
                is_integer_multiple(4.0, bad_dt)

    def test_rejects_negative_value(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            is_integer_multiple(-4.0, 2.0)


class TestToTimesteps:
    """Conversion of a validated duration to a step count."""

    def test_exact_conversion(self):
        assert to_timesteps(6.0, 2.0) == 3
        assert to_timesteps(0.0, 2.0) == 0
        assert to_timesteps(1.5, 0.1) == 15

    def test_tolerates_inexact_float_division(self):
        """0.6 / 0.2 is 2.9999999999999996, which int() truncates to 2."""
        assert int(0.6 / 0.2) == 2, "precondition: truncation is wrong here"
        assert to_timesteps(0.6, 0.2) == 3
        assert to_timesteps(0.3, 0.1) == 3

    def test_rejects_off_grid_values_instead_of_rounding(self):
        """Rounding dt=3 with a 5 ms delay to 2 steps would run 6 ms silently."""
        with pytest.raises(ValueError, match="t_recurrence=5 ms"):
            to_timesteps(5.0, 3.0, name="t_recurrence")
        with pytest.raises(ValueError, match="Use 6 ms \\(2 step\\(s\\)\\) or 9 ms"):
            to_timesteps(7.0, 3.0)

    def test_rejects_negative_value(self):
        with pytest.raises(ValueError, match="t_skip must be non-negative"):
            to_timesteps(-2.0, 2.0, name="t_skip")

    def test_rejects_non_positive_dt(self):
        with pytest.raises(ValueError, match="dt must be positive"):
            to_timesteps(4.0, 0.0)


class TestMultipleErrorMessage:
    """The message has to let the reader fix the config without recomputing."""

    def test_names_parameter_and_both_neighbours(self):
        message = multiple_error_message("t_recurrence", 4.0, 3.0)
        assert "t_recurrence=4 ms" in message
        assert "dt=3 ms" in message
        assert "Use 3 ms (1 step(s)) or 6 ms (2 step(s))" in message


class TestModelParamsRejectsNonMultiples:
    """Validation rejects an off-grid duration for every duration field."""

    @pytest.mark.parametrize("field", DT_MULTIPLE_FIELDS)
    def test_rejects_each_field(self, field):
        with pytest.raises(ValidationError, match=f"{field}=4 ms"):
            make_params(dt=3.0, **{field: 4.0})

    def test_rejects_the_reported_case(self):
        """dt=3 with t_recurrence=4 and loss_reaction_time=4.

        This configuration used to run without error: the recurrent delay was
        truncated to 1 step (3 ms) while the reaction window was rounded up to
        2 steps (6 ms).
        """
        with pytest.raises(ValidationError) as excinfo:
            make_params(dt=3.0, t_recurrence=4.0, loss_reaction_time=4.0)
        assert "t_recurrence" in str(excinfo.value)

    @pytest.mark.parametrize("field", DT_MULTIPLE_FIELDS)
    def test_accepts_multiples(self, field):
        params = make_params(dt=3.0, **{field: 6.0})
        assert getattr(params, field) == 6.0

    def test_accepts_zero_durations(self):
        params = make_params(
            dt=3.0, t_feedforward=0.0, t_skip=0.0, loss_reaction_time=0.0
        )
        assert params.t_recurrence is None

    def test_unset_durations_are_not_rejected(self):
        """None means "use the model default", which is not known here."""
        params = make_params(dt=4.0)
        for field in DT_MULTIPLE_FIELDS:
            assert getattr(params, field) is None

    def test_rejects_non_positive_dt(self):
        for bad_dt in (0.0, -2.0):
            with pytest.raises(ValidationError, match="dt must be positive"):
                make_params(dt=bad_dt, t_recurrence=6.0)

    def test_assignment_is_revalidated(self):
        """validate_assignment=True means a later edit is caught too."""
        params = make_params(dt=3.0, t_recurrence=6.0)
        with pytest.raises(ValidationError, match="t_recurrence=4 ms"):
            params.t_recurrence = 4.0


class TestModelParamsDerivedValues:
    """Derived properties after the constraint is enforced."""

    def test_delay_properties(self):
        params = make_params(dt=3.0, t_recurrence=6.0, t_feedforward=9.0)
        assert params.delay_recurrence == 2
        assert params.delay_feedforward == 3

    def test_delay_properties_are_none_when_unset(self):
        params = make_params()
        assert params.delay_recurrence is None
        assert params.delay_feedforward is None

    def test_effective_tau_steps(self):
        assert make_params(dt=2.0, tau=10.0).effective_tau_steps == 5.0

    def test_effective_tau_steps_is_none_without_tau(self):
        assert (
            ModelParams(seed=0, log_level="INFO", dt=2.0).effective_tau_steps is None
        )


class TestGetTimingSummary:
    """The summary is documented public API and must not raise."""

    def test_reports_derived_values(self):
        summary = make_params(dt=3.0, tau=10.0, t_recurrence=6.0).get_timing_summary()
        assert summary["derived_parameters"]["delay_recurrence"] == 2
        assert summary["derived_parameters"]["effective_tau_steps"] == pytest.approx(
            10.0 / 3.0
        )
        assert summary["assessment"]["delays_are_exact_multiples"] is True

    def test_reports_every_duration(self):
        summary = make_params(
            dt=3.0, t_recurrence=6.0, t_skip=9.0, loss_reaction_time=3.0
        ).get_timing_summary()
        base = summary["base_parameters"]
        assert base["t_skip"] == 9.0
        assert base["loss_reaction_time"] == 3.0

    def test_survives_unset_inputs(self):
        """Every value None: no field may raise, and stability is unknown."""
        summary = ModelParams(seed=0, log_level="INFO").get_timing_summary()
        assert summary["derived_parameters"]["effective_tau_steps"] is None
        assert summary["assessment"]["is_stable"] is None
        assert summary["assessment"]["delays_are_exact_multiples"] is None

    def test_no_phantom_total_processing_keys(self):
        summary = make_params().get_timing_summary()
        derived = summary["derived_parameters"]
        assert "total_processing_time" not in derived
        assert "total_processing_steps" not in derived


class TestModelLevelRejection:
    """TemporalBase re-checks the effective values it actually integrates."""

    def test_rejects_the_reported_case(self):
        with pytest.raises(ValueError, match="t_recurrence=4 ms"):
            make_model(dt=3.0, t_recurrence=4.0, loss_reaction_time=4.0)

    def test_rejects_off_grid_reaction_time(self):
        with pytest.raises(ValueError, match="loss_reaction_time=4 ms"):
            make_model(dt=3.0, loss_reaction_time=4.0)

    def test_rejects_a_default_resolved_duration(self):
        """dt=4 with the model default t_recurrence=6 is off the grid.

        The params layer cannot see this, because the config leaves
        t_recurrence unset and the value is supplied by the model class.
        """
        with pytest.raises(ValueError, match="t_recurrence=6 ms"):
            make_model(dt=4.0)

    def test_inert_durations_are_not_rejected(self):
        """t_feedback is unused unless feedback is enabled."""
        model = make_model(dt=4.0, t_recurrence=8.0, t_feedback=30.0, feedback=False)
        assert model.t_feedback == 30.0

    def test_enabled_durations_are_rejected(self):
        with pytest.raises(ValueError, match="t_feedback=30 ms"):
            make_model(dt=4.0, t_recurrence=8.0, t_feedback=30.0, feedback="add")

    def test_inert_skip_delay_is_not_rejected(self):
        model = make_model(dt=4.0, t_recurrence=8.0, t_skip=30.0, skip=False)
        assert model.t_skip == 30.0

    def test_enabled_skip_delay_is_rejected(self):
        with pytest.raises(ValueError, match="t_skip=30 ms"):
            make_model(dt=4.0, t_recurrence=8.0, t_skip=30.0, skip=True)

    def test_accepts_a_valid_configuration(self):
        model = make_model(
            dt=3.0,
            tau=10.0,
            t_recurrence=6.0,
            t_feedforward=0.0,
            loss_reaction_time=3.0,
            recurrence_type="full",
        )
        assert model.loss_reaction_time == 3.0

    def test_masked_steps_match_the_configured_duration(self):
        """dt=3, loss_reaction_time=3 must mask exactly one step (3 ms)."""
        model = make_model(
            dt=3.0,
            tau=10.0,
            t_recurrence=6.0,
            loss_reaction_time=3.0,
            data_presentation_pattern="1",
            recurrence_type="full",
        )
        pattern = model._get_presentation_pattern()
        mask = model._compute_reaction_mask(pattern)
        assert mask[:1].all(), "the single 3 ms step should be masked"
        assert not mask[1:].any(), "no later step should be masked"
