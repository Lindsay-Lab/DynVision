"""Tests for the model's time windows in milliseconds and in steps.

``t_simulation``/``n_timesteps`` and ``t_idle``/``idle_timesteps`` each name one
quantity twice: in milliseconds at the interface and in steps for the engine.
The claims pinned here:

1. Whichever form is given derives the other with dt, exactly; a legacy step
   value keeps its step meaning and gains its ms value.
2. Contradictory or off-grid values are rejected rather than reconciled.
3. Later updates keep both forms in step, including the data-driven
   ``update_field("n_timesteps", ...)`` calls of the composites.
4. The file-name keys ``tsim``/``tidle`` route through every composite, and a
   legacy key in user input logs a deprecation warning.
5. The engine still receives step counts only.
"""

import logging

import pytest
from pydantic import ValidationError

from dynvision.models.dyrcnn import DyRCNNx2
from dynvision.params import InitParams, TestingParams, TrainingParams
from dynvision.params.model_params import ModelParams
from dynvision.utils import timing

from .utils import create_temp_config, temporary_training_paths


def make_params(**overrides) -> ModelParams:
    base = {"seed": 0, "log_level": "INFO"}
    return ModelParams(**{**base, **overrides})


class TestDerivation:
    def test_ms_values_derive_step_counts(self):
        params = make_params(dt=2.0, t_simulation=40.0, t_idle=40.0)
        assert params.n_timesteps == 20
        assert params.idle_timesteps == 20

    def test_step_counts_derive_ms_values(self):
        """A legacy tsteps=20 at dt=0.5 simulated 10 ms, and says so now."""
        params = make_params(dt=0.5, n_timesteps=20, idle_timesteps=20)
        assert params.t_simulation == 10.0
        assert params.t_idle == 10.0

    def test_the_same_ms_value_means_more_steps_at_smaller_dt(self):
        assert make_params(dt=0.5, t_simulation=40.0).n_timesteps == 80
        assert make_params(dt=2.0, t_simulation=40.0).n_timesteps == 20

    def test_agreeing_forms_are_accepted(self):
        params = make_params(dt=2.0, t_simulation=40.0, n_timesteps=20)
        assert params.n_timesteps == 20

    def test_without_dt_step_counts_stay_as_given(self):
        params = make_params(n_timesteps=20)
        assert params.n_timesteps == 20
        assert params.t_simulation is None

    def test_derived_values_are_tagged_in_provenance(self):
        params = make_params(dt=2.0, t_simulation=40.0)
        record = params._value_provenance["n_timesteps"]
        assert "derived" in str(record)


class TestRejection:
    def test_contradicting_forms_are_rejected(self):
        with pytest.raises(ValidationError, match="disagree at dt=2 ms"):
            make_params(dt=2.0, t_simulation=40.0, n_timesteps=25)

    def test_off_grid_ms_value_is_rejected(self):
        with pytest.raises(ValidationError, match="t_simulation=40 ms is not an integer multiple"):
            make_params(dt=3.0, t_simulation=40.0)

    def test_ms_value_without_dt_is_rejected(self):
        with pytest.raises(ValidationError, match="needs dt"):
            make_params(t_idle=40.0)


class TestUpdatesStayInStep:
    def test_assigning_ms_rederives_steps(self):
        params = make_params(dt=2.0, t_simulation=40.0)
        params.t_simulation = 60.0
        assert params.n_timesteps == 30

    def test_assigning_steps_rederives_ms(self):
        params = make_params(dt=2.0, t_simulation=40.0)
        params.n_timesteps = 10
        assert params.t_simulation == 20.0

    def test_rejected_assignment_leaves_both_forms_unchanged(self):
        params = make_params(dt=2.0, t_simulation=40.0)
        with pytest.raises(ValidationError):
            params.t_simulation = 41.0
        assert (params.t_simulation, params.n_timesteps) == (40.0, 20)

    def test_update_field_from_data_dims_syncs_ms(self):
        """The composites set n_timesteps from the data without revalidation."""
        params = make_params(dt=2.0, t_simulation=40.0)
        params.update_field("n_timesteps", 40, mutation_tag="derived")
        assert params.t_simulation == 80.0

    def test_update_field_of_ms_syncs_steps(self):
        params = make_params(dt=0.5, t_idle=10.0)
        params.update_field("t_idle", 20.0)
        assert params.idle_timesteps == 40


class TestEngineReceivesSteps:
    def test_model_kwargs_carry_step_counts_only(self):
        kwargs = make_params(dt=2.0, t_simulation=40.0, t_idle=8.0).get_model_kwargs()
        assert kwargs["n_timesteps"] == 20
        assert kwargs["idle_timesteps"] == 4
        assert "t_simulation" not in kwargs
        assert "t_idle" not in kwargs

    def test_a_model_built_from_ms_params_runs_the_derived_steps(self):
        params = make_params(
            dt=2.0,
            t_simulation=20.0,
            t_idle=4.0,
            n_classes=10,
            input_dims=(10, 3, 32, 32),
            recurrence_type="full",
            t_recurrence=6.0,
        )
        model = DyRCNNx2(**params.get_model_kwargs(DyRCNNx2))
        assert model.n_timesteps == 10
        assert model.idle_timesteps == 2


class TestFileNameKeys:
    @pytest.mark.parametrize("composite", (TrainingParams, TestingParams, InitParams))
    @pytest.mark.parametrize("key,target", [("tsim", "model.t_simulation"), ("tidle", "model.t_idle")])
    def test_keys_route_through_every_composite(self, composite, key, target):
        assert composite.get_aliases()[key] == target

    def _train(self, config_data, overrides):
        config_file = create_temp_config(config_data)
        try:
            with temporary_training_paths() as training_paths:
                return TrainingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={**training_paths, **overrides},
                )
        finally:
            config_file.unlink()

    def test_tsim_reaches_the_model_as_steps(self):
        params = self._train({"dt": 2}, {"tsim": 40.0, "tidle": 8.0})
        assert params.model.t_simulation == 40.0
        assert params.model.n_timesteps == 20
        assert params.model.idle_timesteps == 4

    def test_legacy_tsteps_keeps_its_step_meaning(self, caplog, monkeypatch):
        monkeypatch.setattr(timing, "_WARNED_LEGACY_KEYS", set())
        with caplog.at_level(logging.WARNING, logger="dynvision.utils.timing"):
            params = self._train({"dt": 0.5}, {"tsteps": 20})
        assert params.model.n_timesteps == 20
        assert params.model.t_simulation == 10.0
        assert "Legacy step key 'tsteps'" in caplog.text
        assert "use 'tsim'" in caplog.text

    def test_legacy_warning_is_logged_once(self, caplog, monkeypatch):
        monkeypatch.setattr(timing, "_WARNED_LEGACY_KEYS", set())
        with caplog.at_level(logging.WARNING, logger="dynvision.utils.timing"):
            timing.warn_legacy_keys(["idle", "model.idle", "tsim", "dt"])
        assert caplog.text.count("Legacy step key 'idle'") == 1
        assert "tsim" not in caplog.text.replace("'tidle'", "")
