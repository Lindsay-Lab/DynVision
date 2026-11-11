"""Model-side delay conversion: exact, strict, and never silently clamped.

The golden test pins that on-grid configurations are unchanged. These tests
pin what changed for the rest:

1. A delay that floating-point division used to truncate converts correctly.
2. A recurrent or feedback delay shorter than one step is rejected, at the
   params layer, at the model and at the layer, instead of being raised to one
   step without notice.
3. A layer never sizes its buffers with a default dt of its own, and a
   recurrence that its buffer cannot hold is rejected instead of switched off.
"""

import pytest
from pydantic import ValidationError

from dynvision.model_components.recurrence import RecurrentConnectedConv2d
from dynvision.models.dyrcnn import DyRCNNx2
from dynvision.params.model_params import ModelParams


def make_model(**overrides) -> DyRCNNx2:
    base = dict(
        n_timesteps=10,
        n_classes=10,
        input_dims=(10, 3, 32, 32),
        dt=2.0,
        tau=10.0,
        t_recurrence=6.0,
        recurrence_type="full",
        shuffle_presentation_pattern=False,
    )
    return DyRCNNx2(**{**base, **overrides})


def make_layer(**overrides) -> RecurrentConnectedConv2d:
    base = dict(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        dim_y=16,
        dim_x=16,
        dt=2.0,
        t_recurrence=6.0,
        recurrence_type="full",
    )
    return RecurrentConnectedConv2d(**{**base, **overrides})


def make_params(**overrides) -> ModelParams:
    base = {"seed": 0, "log_level": "INFO", "dt": 2.0, "tau": 10.0}
    return ModelParams(**{**base, **overrides})


class TestExactConversion:
    def test_delay_that_int_truncated(self):
        """int(0.6 / 0.2) == 2, so a 0.6 ms recurrence used to run as 0.4 ms."""
        layer = make_layer(dt=0.2, t_recurrence=0.6)
        assert layer.delay_recurrence == 3

    def test_skip_index_keeps_its_offset(self):
        """A skip index is steps + 1: the source stored the current step already."""
        model = make_model(skip=True, t_skip=4.0)
        assert model.delay_index_skip == 3

    def test_feedback_index_has_no_offset(self):
        model = make_model(feedback="add", t_feedback=10.0)
        assert model.delay_index_feedback == 5

    def test_disabled_connections_get_no_index(self):
        model = make_model(skip=False, feedback=False, t_skip=3.0, t_feedback=7.0)
        assert model.delay_index_skip is None
        assert model.delay_index_feedback is None


class TestMinimumOneStep:
    def test_model_rejects_recurrence_shorter_than_dt(self):
        with pytest.raises(ValueError, match="t_recurrence=0 ms is shorter than 1 step"):
            make_model(t_recurrence=0.0)

    def test_model_accepts_zero_recurrence_without_recurrence(self):
        model = make_model(t_recurrence=0.0, recurrence_type="none")
        assert model.t_recurrence == 0.0

    def test_model_rejects_feedback_shorter_than_dt(self):
        with pytest.raises(ValueError, match="t_feedback=0 ms is shorter than 1 step"):
            make_model(feedback="add", t_feedback=0.0)

    def test_params_reject_recurrence_shorter_than_dt(self):
        with pytest.raises(ValidationError, match="Use t_recurrence >= 2 ms"):
            make_params(t_recurrence=0.0, recurrence_type="full")

    def test_params_accept_one_step(self):
        assert make_params(t_recurrence=2.0, recurrence_type="full").delay_recurrence == 1

    def test_params_do_not_guess_an_unset_recurrence_type(self):
        """With recurrence_type unset the model decides; it re-checks itself."""
        assert make_params(t_recurrence=0.0).t_recurrence == 0.0

    def test_params_reject_feedback_shorter_than_dt(self):
        with pytest.raises(ValidationError, match="t_feedback=0 ms"):
            make_params(t_feedback=0.0, feedback="add")

    def test_layer_rejects_recurrence_shorter_than_dt(self):
        with pytest.raises(ValueError, match="t_recurrence=0 ms is shorter than 1 step"):
            make_layer(t_recurrence=0.0)


class TestLayerTiming:
    def test_layer_requires_dt(self):
        with pytest.raises(ValueError, match="needs dt"):
            make_layer(dt=None)

    def test_layer_rejects_off_grid_history(self):
        with pytest.raises(ValueError, match="history_length=5 ms"):
            make_layer(history_length=5.0)

    def test_layer_rejects_a_recurrence_its_buffer_cannot_hold(self):
        """get_hidden_state() would return None and switch the recurrence off."""
        with pytest.raises(ValueError, match="needs 3 stored steps"):
            make_layer(t_recurrence=6.0, history_length=2.0)
