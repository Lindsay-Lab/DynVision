"""The stimulus protocol in milliseconds, converted with the model's dt.

The data component holds the protocol of the Stimulus* loaders but not ``dt``,
which belongs to the model. ``coordinate_component_timing`` converts it in every
composite, so the loaders keep receiving step counts. The claims pinned here:

1. The ms keys (``tseq``, ``tstim``, ``tintro``, ``tinterval``/``tisi``) and the
   legacy step keys (``dsteps``, ``stim``, ``intro``, ``interval``) route to
   declared ``DataParams`` fields, not to unvalidated extras.
2. One ms protocol yields stimulus onsets and offsets at the same times in ms
   at every dt -- the property the step-defined experiments lacked.
3. A legacy step protocol keeps its step meaning and gains its ms values.
4. Off-grid, contradictory or unconvertible values are rejected.
"""

from typing import Any, Dict, List

import pytest
import torch

from dynvision.data.dataloader import get_data_loader
from dynvision.params import InitParams, TestingParams, TrainingParams

from .utils import create_temp_config, temporary_testing_paths, temporary_training_paths

LOADER_WINDOWS = ("data_timesteps", "stimulus_duration", "intro_duration", "interval_duration")


def _test(args: List[str]) -> TestingParams:
    """Resolve CLI-style arguments as the workflow's test_model rule does."""
    config_file = create_temp_config({"data_loader": "StimulusInterval"})
    try:
        with temporary_testing_paths() as testing_paths:
            return TestingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={**testing_paths, "verbose": False},
                args=args,
            )
    finally:
        config_file.unlink()


def _train(overrides: Dict[str, Any]) -> TrainingParams:
    config_file = create_temp_config({})
    try:
        with temporary_training_paths() as training_paths:
            return TrainingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={**training_paths, **overrides},
            )
    finally:
        config_file.unlink()


def _label_edges_ms(params) -> List[float]:
    """Build the interval loader from the params and return label edges in ms."""
    kwargs = params.data.get_dataloader_kwargs()
    windows = {key: kwargs[key] for key in LOADER_WINDOWS}
    dataset = [(torch.rand(3, 8, 8), index % 3) for index in range(2)]
    loader = get_data_loader(
        dataset, "StimulusInterval", batch_size=2, num_workers=0, **windows
    )
    _, labels, *_ = next(iter(loader))
    present = (labels[0] >= 0).int().tolist()
    edges = [step for step in range(1, len(present)) if present[step] != present[step - 1]]
    return [step * params.model.dt for step in edges]


class TestRouting:
    @pytest.mark.parametrize("composite", (TrainingParams, TestingParams, InitParams))
    @pytest.mark.parametrize(
        "key,target",
        [
            ("tseq", "data.t_sequence"),
            ("tstim", "data.t_stimulus"),
            ("tintro", "data.t_intro"),
            ("tinterval", "data.t_interval"),
            ("tisi", "data.t_interval"),
            ("stim", "data.stimulus_duration"),
            ("intro", "data.intro_duration"),
            ("interval", "data.interval_duration"),
        ],
    )
    def test_keys_reach_declared_data_fields(self, composite, key, target):
        assert composite.get_aliases()[key] == target

    def test_cli_values_are_fields_not_extras(self):
        params = _test(["--dt", "2", "--tseq", "40", "--tstim", "10", "--stim", "5"])
        assert params.data.t_stimulus == 10.0
        extras = getattr(params.data, "__pydantic_extra__", None) or {}
        assert not {"tseq", "tstim", "stim"} & set(extras)


class TestMillisecondProtocol:
    PROTOCOL = ["--tseq", "40", "--tintro", "2", "--tstim", "10", "--tisi", "8"]

    @pytest.mark.parametrize("dt, steps", [("2", (20, 1, 5, 4)), ("0.5", (80, 4, 20, 16))])
    def test_converted_to_steps_with_the_model_dt(self, dt, steps):
        params = _test(["--dt", dt, *self.PROTOCOL])
        data = params.data
        assert (
            data.data_timesteps,
            data.intro_duration,
            data.stimulus_duration,
            data.interval_duration,
        ) == steps

    def test_stimulus_edges_are_at_the_same_ms_for_every_dt(self):
        edges = {dt: _label_edges_ms(_test(["--dt", dt, *self.PROTOCOL])) for dt in ("2", "1", "0.5")}
        assert edges["2"] == edges["1"] == edges["0.5"] == [2.0, 12.0, 20.0, 30.0]

    def test_derived_steps_are_tagged_in_provenance(self):
        params = _test(["--dt", "2", *self.PROTOCOL])
        assert "derived" in str(params.data._value_provenance["stimulus_duration"])


class TestLegacyProtocol:
    LEGACY = ["--dsteps", "40", "--intro", "1", "--stim", "5", "--interval", "4"]

    def test_step_values_keep_their_meaning(self):
        params = _test(["--dt", "0.5", *self.LEGACY])
        assert (params.data.data_timesteps, params.data.stimulus_duration) == (40, 5)

    def test_step_values_gain_their_ms_values(self):
        params = _test(["--dt", "0.5", *self.LEGACY])
        assert params.data.t_stimulus == 2.5
        assert params.data.t_sequence == 20.0

    def test_the_same_steps_last_less_time_at_smaller_dt(self):
        """The effect that motivated the fix, now visible in the params."""
        at_2 = _label_edges_ms(_test(["--dt", "2", *self.LEGACY]))
        at_half = _label_edges_ms(_test(["--dt", "0.5", *self.LEGACY]))
        assert at_half == [edge / 4 for edge in at_2]


class TestRejection:
    def test_off_grid_stimulus(self):
        with pytest.raises(Exception, match="t_stimulus=5 ms is not an integer multiple of dt=2"):
            _test(["--dt", "2", "--tseq", "40", "--tstim", "5"])

    def test_contradicting_forms(self):
        with pytest.raises(Exception, match="t_stimulus=10 ms and stimulus_duration=4 disagree"):
            _test(["--dt", "2", "--tseq", "40", "--tstim", "10", "--stim", "4"])

    def test_empty_sequence(self):
        """t_sequence=0 would derive data_timesteps=0, which dsteps=0 cannot set."""
        with pytest.raises(Exception, match="t_sequence"):
            _test(["--dt", "2", "--tseq", "0"])

    def test_unconverted_ms_values_are_not_passed_on(self):
        """Without a composite nobody converts t_sequence; the loader must not guess."""
        data = _test(["--dt", "2", "--tseq", "40"]).data.model_copy()
        object.__setattr__(data, "data_timesteps", None)
        with pytest.raises(ValueError, match="needs the model's dt"):
            data.get_dataloader_kwargs()


class TestDefaults:
    def test_single_frames_when_no_sequence_is_given(self):
        params = _train({"dt": 2})
        assert params.data.data_timesteps == 1

    def test_training_window_follows_the_sequence_in_ms(self):
        """The coordinator runs before conflict resolution, which sees the steps."""
        params = _train({"dt": 2, "tseq": 60})
        assert params.data.data_timesteps == 30
        assert params.model.n_timesteps == 30
        assert params.model.t_simulation == 60.0
        assert params.data.t_sequence == 60.0
