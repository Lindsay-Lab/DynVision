"""Presentation-pattern chunks have one duration each, or the pattern is rejected.

``data_presentation_pattern`` (e.g. ``1011``) is stretched over the simulation
window, one chunk per entry. Floor indexing used to accept any window, so
``1011`` over 10 steps ran chunks of 3, 2, 3 and 2 steps: the stimulus and blank
periods differed in duration without notice, and differently for every dt.

Pinned here:

1. A window that divides evenly expands exactly as before.
2. A window that does not is rejected with the chunk lengths and the nearest
   valid windows in steps and ms.
3. The check fires only where the pattern is applied, so a pattern-trained
   model can still be tested on sequences of any length.
"""

import logging
import warnings

import pytest
import torch

from dynvision.models.dyrcnn import DyRCNNx2


def make_model(n_timesteps: int, pattern: str = "1011", **overrides) -> DyRCNNx2:
    base = dict(
        n_timesteps=n_timesteps,
        n_classes=10,
        input_dims=(n_timesteps, 3, 32, 32),
        dt=2.0,
        tau=10.0,
        t_recurrence=6.0,
        recurrence_type="full",
        data_presentation_pattern=pattern,
        shuffle_presentation_pattern=False,
    )
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return DyRCNNx2(**{**base, **overrides})
    finally:
        logging.disable(logging.NOTSET)


def floor_expansion(pattern: torch.Tensor, n_timesteps: int) -> torch.Tensor:
    """The previous expansion, kept as the reference for divisible windows."""
    indices = torch.arange(n_timesteps) * pattern.numel() // n_timesteps
    return pattern[torch.clamp(indices, 0, pattern.numel() - 1)]


class TestEvenChunks:
    @pytest.mark.parametrize("n_timesteps", [4, 8, 20, 40])
    def test_divisible_windows_expand_as_before(self, n_timesteps):
        model = make_model(n_timesteps)
        base = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
        assert torch.equal(
            model._get_presentation_pattern(), floor_expansion(base, n_timesteps)
        )

    def test_chunks_have_equal_length(self):
        pattern = make_model(20)._get_presentation_pattern().int().tolist()
        assert pattern == [1] * 5 + [0] * 5 + [1] * 10

    def test_single_entry_pattern_fits_any_window(self):
        model = make_model(7, pattern="1")
        assert model._get_presentation_pattern().all()


class TestUnevenChunks:
    def test_uneven_window_is_rejected_with_the_chunk_lengths(self):
        model = make_model(10)
        with pytest.raises(ValueError, match=r"the chunks would last \[3, 2, 3, 2\] steps"):
            model._get_presentation_pattern()

    def test_message_names_the_nearest_windows_in_ms(self):
        model = make_model(10)
        with pytest.raises(ValueError, match=r"Use a simulation window of 8 or 12 steps \(16 or 24 ms\)"):
            model._get_presentation_pattern()

    def test_shuffled_patterns_are_checked_too(self):
        model = make_model(10, shuffle_presentation_pattern=True)
        with pytest.raises(ValueError, match="do not divide n_timesteps=10"):
            model._get_presentation_pattern()


class TestOnlyWhereApplied:
    def test_construction_does_not_expand_the_pattern(self):
        """A 1011-trained model tested on a 30-step sequence never applies it."""
        model = make_model(30)
        assert not hasattr(model, "_cached_presentation_pattern")

    def test_sequence_input_bypasses_the_pattern(self):
        model = make_model(30)
        inputs = torch.rand(2, 30, 3, 32, 32)
        labels = torch.zeros(2, 30, dtype=torch.long)
        expanded, _ = model._expand_timesteps((inputs, labels))
        assert expanded.shape[1] == 30

    def test_static_input_applies_and_checks_the_pattern(self):
        model = make_model(10)
        inputs = torch.rand(2, 1, 3, 32, 32)
        labels = torch.zeros(2, 1, dtype=torch.long)
        with pytest.raises(ValueError, match="do not divide n_timesteps=10"):
            model._expand_timesteps((inputs, labels))


class TestTrainWindowReconciliation:
    """When the data fixes the step count, the ms window follows and is reported."""

    def _train(self, config_data, overrides):
        from ..params.utils import create_temp_config, temporary_training_paths
        from dynvision.params import TrainingParams

        config_file = create_temp_config(config_data)
        try:
            with temporary_training_paths() as training_paths:
                return TrainingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={**training_paths, **overrides},
                )
        finally:
            config_file.unlink()

    def test_configured_window_overridden_by_data_is_warned_in_ms(self, caplog):
        with caplog.at_level(logging.WARNING, logger="dynvision.params.training_params"):
            params = self._train({"dt": 2}, {"tsim": 40.0, "data_timesteps": 30})
        assert params.model.n_timesteps == 30
        assert params.model.t_simulation == 60.0
        assert "simulation window 40 ms -> 60 ms at dt=2 ms" in caplog.text
