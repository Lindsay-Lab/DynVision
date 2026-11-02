"""Tests for temporal masking in TemporalBase.

This test suite verifies that:
1. loss_reaction_time is applied correctly for all patterns (including static images)
2. data_presentation_pattern correctly masks inputs and labels
3. The combination of pattern and reaction time masking works as expected
4. Edge cases are handled properly
"""

import pytest
import torch

from dynvision.models.dyrcnn import DyRCNNx2


class TestReactionTimeMasking:
    """Test loss_reaction_time masking for various patterns."""

    @pytest.fixture
    def base_config(self):
        """Common model configuration for tests."""
        return {
            "n_timesteps": 10,
            "n_classes": 10,
            "input_dims": (10, 3, 32, 32),
            "dt": 2.0,
            "tau": 10.0,
            "recurrence_type": "none",  # Simplify for testing
            "shuffle_presentation_pattern": False,  # Deterministic for testing
        }

    def test_reaction_time_applied_with_static_image(self, base_config):
        """Test that loss_reaction_time is applied when pattern='1' (static image).

        This is the key bug fix: reaction time should mask the first N timesteps
        even when the presentation pattern is just '1' (always showing stimulus).
        """
        model = DyRCNNx2(
            **base_config,
            data_presentation_pattern="1",
            loss_reaction_time=6.0,  # Should mask 3 timesteps (6ms / 2ms dt = 3)
        )

        # Create test batch
        batch_size = 2
        inputs = torch.randn(batch_size, 1, 3, 32, 32)
        labels = torch.ones(batch_size, 1, dtype=torch.long)  # Valid labels

        # Call _expand_timesteps
        expanded_inputs, expanded_labels, *_ = model._expand_timesteps(
            (inputs, labels)
        )

        # Check shapes
        assert expanded_inputs.shape == (batch_size, 10, 3, 32, 32)
        assert expanded_labels.shape == (batch_size, 10)

        # Inputs should NOT be masked (pattern is all 1s)
        # All timesteps should have the same input
        assert torch.allclose(expanded_inputs[:, 0], expanded_inputs[:, 5])

        # Labels for first 3 timesteps should be masked (reaction time)
        reaction_steps = 3  # ceil(6.0 / 2.0) = 3
        for t in range(reaction_steps):
            assert (
                expanded_labels[:, t] == model.non_label_index
            ).all(), f"Timestep {t} should be masked for reaction time"

        # Labels after reaction time should be valid
        for t in range(reaction_steps, 10):
            assert (
                expanded_labels[:, t] != model.non_label_index
            ).all(), f"Timestep {t} should NOT be masked"

    def test_no_reaction_time_no_masking_with_static_image(self, base_config):
        """Test that no masking occurs when reaction_time=0 and pattern='1'."""
        model = DyRCNNx2(
            **base_config,
            data_presentation_pattern="1",
            loss_reaction_time=0.0,
        )

        batch_size = 2
        inputs = torch.randn(batch_size, 1, 3, 32, 32)
        labels = torch.full((batch_size, 1), 5, dtype=torch.long)

        expanded_inputs, expanded_labels, *_ = model._expand_timesteps(
            (inputs, labels)
        )

        # No labels should be masked
        assert (expanded_labels == 5).all(), "No labels should be masked"

    def test_reaction_time_with_pattern_containing_zeros(self, base_config):
        """Test reaction time masking with a pattern like '1011'.

        Pattern '1011' expands to 12 timesteps in equal chunks of 3:
        indices: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        pattern: [T, T, T, F, F, F, T, T, T, T, T, T]

        Rising edges at t=0 and t=6 (after the False gap).
        With reaction_time=4ms, dt=2ms -> 2 timesteps masked after each onset.
        The window must be a multiple of the pattern length; 10 steps would
        give uneven chunks and is rejected.
        """
        model = DyRCNNx2(
            **{**base_config, "n_timesteps": 12, "input_dims": (12, 3, 32, 32)},
            data_presentation_pattern="1011",  # On, off, on, on
            loss_reaction_time=4.0,  # Masks 2 timesteps after each onset
        )

        batch_size = 2
        inputs = torch.randn(batch_size, 1, 3, 32, 32)
        labels = torch.full((batch_size, 1), 5, dtype=torch.long)

        expanded_inputs, expanded_labels, *_ = model._expand_timesteps(
            (inputs, labels)
        )

        # Get the expanded pattern to understand expected behavior
        pattern = model._get_presentation_pattern()

        # Check that inputs are zeroed where pattern is 0
        for t in range(model.n_timesteps):
            if not pattern[t]:
                # Input should be non_input_value (default 0)
                assert (
                    expanded_inputs[:, t] == model.non_input_value
                ).all(), f"Input at t={t} should be masked (pattern=0)"

        # Labels should be masked where pattern is 0 OR within reaction window
        reaction_mask = model._compute_reaction_mask(pattern)
        pattern_mask = ~pattern
        expected_label_mask = pattern_mask | reaction_mask

        for t in range(model.n_timesteps):
            if expected_label_mask[t]:
                assert (
                    expanded_labels[:, t] == model.non_label_index
                ).all(), f"Label at t={t} should be masked"
            else:
                assert (
                    expanded_labels[:, t] == 5
                ).all(), f"Label at t={t} should NOT be masked"


class TestComputeReactionMask:
    """Test the _compute_reaction_mask method directly."""

    @pytest.fixture
    def model(self):
        """Create a model for testing mask computation."""
        return DyRCNNx2(
            n_timesteps=10,
            n_classes=10,
            input_dims=(10, 3, 32, 32),
            dt=2.0,
            loss_reaction_time=4.0,  # 2 timesteps
            recurrence_type="none",
            shuffle_presentation_pattern=False,
        )

    def test_static_image_pattern_rising_edge_at_start(self, model):
        """Test that a static image (all True) has rising edge at t=0."""
        pattern = torch.ones(10, dtype=torch.bool)
        mask = model._compute_reaction_mask(pattern)

        # First 2 timesteps should be masked (reaction_time=4ms, dt=2ms)
        assert mask[0] == True
        assert mask[1] == True
        assert mask[2] == False
        assert mask[3:].sum() == 0

    def test_pattern_with_single_onset(self, model):
        """Test pattern with stimulus onset in the middle."""
        # Pattern: off, off, on, on, on, on, on, on, on, on
        pattern = torch.tensor(
            [False, False, True, True, True, True, True, True, True, True]
        )
        mask = model._compute_reaction_mask(pattern)

        # Rising edge at t=2, so t=2 and t=3 should be masked
        assert mask[0] == False
        assert mask[1] == False
        assert mask[2] == True
        assert mask[3] == True
        assert mask[4] == False

    def test_pattern_with_multiple_onsets(self, model):
        """Test pattern with multiple stimulus onsets."""
        # Pattern: on, off, on, on, off, on, on, on, on, on
        pattern = torch.tensor(
            [True, False, True, True, False, True, True, True, True, True]
        )
        mask = model._compute_reaction_mask(pattern)

        # Rising edges at t=0, t=2, t=5
        # Reaction window is 2 timesteps
        expected = torch.tensor(
            [True, True, True, True, False, True, True, False, False, False]
        )
        assert torch.equal(mask, expected), f"Expected {expected}, got {mask}"

    def test_zero_reaction_time_returns_all_false(self):
        """Test that zero reaction time produces no masking."""
        model = DyRCNNx2(
            n_timesteps=10,
            n_classes=10,
            input_dims=(10, 3, 32, 32),
            dt=2.0,
            loss_reaction_time=0.0,
            recurrence_type="none",
            shuffle_presentation_pattern=False,
        )

        pattern = torch.ones(10, dtype=torch.bool)
        mask = model._compute_reaction_mask(pattern)

        assert mask.sum() == 0, "No masking should occur with reaction_time=0"

    def test_all_false_pattern_no_masking(self, model):
        """Test that an all-False pattern produces no reaction masking."""
        pattern = torch.zeros(10, dtype=torch.bool)
        mask = model._compute_reaction_mask(pattern)

        assert mask.sum() == 0, "No rising edges means no reaction masking"


class TestPatternMasking:
    """Test data_presentation_pattern masking of inputs."""

    @pytest.fixture
    def base_config(self):
        return {
            "n_timesteps": 10,
            "n_classes": 10,
            "input_dims": (10, 3, 32, 32),
            "dt": 2.0,
            "recurrence_type": "none",
            "loss_reaction_time": 0.0,  # Disable reaction time for these tests
            "shuffle_presentation_pattern": False,  # Deterministic for testing
        }

    def test_pattern_zeros_mask_inputs(self, base_config):
        """Test that pattern zeros cause inputs to be set to non_input_value."""
        model = DyRCNNx2(
            **{**base_config, "n_timesteps": 12, "input_dims": (12, 3, 32, 32)},
            data_presentation_pattern="1010",  # Alternating
        )

        batch_size = 2
        inputs = torch.randn(batch_size, 1, 3, 32, 32) + 10  # Offset to distinguish from 0
        labels = torch.full((batch_size, 1), 5, dtype=torch.long)

        expanded_inputs, expanded_labels, *_ = model._expand_timesteps(
            (inputs, labels)
        )

        pattern = model._get_presentation_pattern()

        for t in range(model.n_timesteps):
            if pattern[t]:
                # Input should be preserved (expanded from original)
                assert (
                    expanded_inputs[:, t].mean() > 5
                ), f"Input at t={t} should be preserved"
            else:
                # Input should be non_input_value
                assert (
                    expanded_inputs[:, t] == model.non_input_value
                ).all(), f"Input at t={t} should be zeroed"

    def test_pattern_zeros_also_mask_labels(self, base_config):
        """Test that pattern zeros cause labels to be set to non_label_index."""
        model = DyRCNNx2(
            **{**base_config, "n_timesteps": 12, "input_dims": (12, 3, 32, 32)},
            data_presentation_pattern="1100",  # First half on, second half off
        )

        batch_size = 2
        inputs = torch.randn(batch_size, 1, 3, 32, 32)
        labels = torch.full((batch_size, 1), 5, dtype=torch.long)

        expanded_inputs, expanded_labels, *_ = model._expand_timesteps(
            (inputs, labels)
        )

        pattern = model._get_presentation_pattern()

        for t in range(model.n_timesteps):
            if pattern[t]:
                assert (
                    expanded_labels[:, t] == 5
                ).all(), f"Label at t={t} should be preserved"
            else:
                assert (
                    expanded_labels[:, t] == model.non_label_index
                ).all(), f"Label at t={t} should be masked"


class TestCombinedMasking:
    """Test combination of pattern and reaction time masking."""

    def test_combined_masking_labels(self):
        """Test that labels are masked by both pattern zeros AND reaction time."""
        model = DyRCNNx2(
            n_timesteps=10,
            n_classes=10,
            input_dims=(10, 3, 32, 32),
            dt=2.0,
            data_presentation_pattern="1100111100",  # Complex pattern
            loss_reaction_time=4.0,  # 2 timesteps
            recurrence_type="none",
            shuffle_presentation_pattern=False,
        )

        batch_size = 2
        inputs = torch.randn(batch_size, 1, 3, 32, 32)
        labels = torch.full((batch_size, 1), 5, dtype=torch.long)

        expanded_inputs, expanded_labels, *_ = model._expand_timesteps(
            (inputs, labels)
        )

        # Get masks
        pattern = model._get_presentation_pattern()
        pattern_mask = ~pattern
        reaction_mask = model._compute_reaction_mask(pattern)
        combined_mask = pattern_mask | reaction_mask

        # Verify label masking matches expected
        for t in range(10):
            if combined_mask[t]:
                assert (
                    expanded_labels[:, t] == model.non_label_index
                ).all(), f"Label at t={t} should be masked"
            else:
                assert (
                    expanded_labels[:, t] == 5
                ).all(), f"Label at t={t} should NOT be masked"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_timestep_no_expansion(self):
        """Test that single timestep model doesn't expand."""
        model = DyRCNNx2(
            n_timesteps=1,
            n_classes=10,
            input_dims=(1, 3, 32, 32),
            dt=2.0,
            data_presentation_pattern="1",
            loss_reaction_time=4.0,
            recurrence_type="none",
            shuffle_presentation_pattern=False,
        )

        batch_size = 2
        inputs = torch.randn(batch_size, 1, 3, 32, 32)
        labels = torch.full((batch_size, 1), 5, dtype=torch.long)

        # Should not expand since n_timesteps=1
        expanded_inputs, expanded_labels, *_ = model._expand_timesteps(
            (inputs, labels)
        )

        assert expanded_inputs.shape == (batch_size, 1, 3, 32, 32)
        assert expanded_labels.shape == (batch_size, 1)

    def test_already_expanded_input_not_masked(self):
        """Test that already-expanded inputs are not re-masked."""
        model = DyRCNNx2(
            n_timesteps=5,
            n_classes=10,
            input_dims=(5, 3, 32, 32),
            dt=2.0,
            data_presentation_pattern="1",
            loss_reaction_time=4.0,
            recurrence_type="none",
            shuffle_presentation_pattern=False,
        )

        batch_size = 2
        # Input already has 5 timesteps
        inputs = torch.randn(batch_size, 5, 3, 32, 32) + 10
        labels = torch.full((batch_size, 5), 5, dtype=torch.long)

        # Should not expand or mask since input already has multiple timesteps
        expanded_inputs, expanded_labels, *_ = model._expand_timesteps(
            (inputs, labels)
        )

        # Inputs should be unchanged (original values preserved)
        assert torch.allclose(expanded_inputs, inputs)
        # Labels should be unchanged
        assert (expanded_labels == 5).all()

    def test_fractional_reaction_steps_rejected(self):
        """Test that a reaction time off the dt grid is rejected, not rounded.

        ``loss_reaction_time=5.0`` with ``dt=2.0`` is 2.5 integration steps. It
        used to be quietly rounded up to 3 steps (a 6 ms window), which ran a
        different reaction time than the one configured.
        """
        with pytest.raises(ValueError, match="loss_reaction_time=5 ms"):
            DyRCNNx2(
                n_timesteps=10,
                n_classes=10,
                input_dims=(10, 3, 32, 32),
                dt=2.0,
                data_presentation_pattern="1",
                loss_reaction_time=5.0,  # 5ms / 2ms = 2.5 -> not a whole step
                recurrence_type="none",
                shuffle_presentation_pattern=False,
            )

    def test_exact_reaction_steps_masked(self):
        """Test that an on-grid reaction time masks exactly that many steps."""
        model = DyRCNNx2(
            n_timesteps=10,
            n_classes=10,
            input_dims=(10, 3, 32, 32),
            dt=2.0,
            data_presentation_pattern="1",
            loss_reaction_time=6.0,  # 6ms / 2ms = 3 whole steps
            recurrence_type="none",
            shuffle_presentation_pattern=False,
        )

        pattern = torch.ones(10, dtype=torch.bool)
        mask = model._compute_reaction_mask(pattern)

        assert mask[:3].all(), "First 3 timesteps should be masked"
        assert not mask[3:].any(), "Remaining timesteps should not be masked"

    def test_non_input_value_customization(self):
        """Test that custom non_input_value is used."""
        model = DyRCNNx2(
            n_timesteps=10,
            n_classes=10,
            input_dims=(10, 3, 32, 32),
            dt=2.0,
            data_presentation_pattern="10",
            loss_reaction_time=0.0,
            non_input_value=-1.0,  # Custom value
            recurrence_type="none",
            shuffle_presentation_pattern=False,
        )

        batch_size = 2
        inputs = torch.randn(batch_size, 1, 3, 32, 32) + 10
        labels = torch.full((batch_size, 1), 5, dtype=torch.long)

        expanded_inputs, expanded_labels, *_ = model._expand_timesteps(
            (inputs, labels)
        )

        pattern = model._get_presentation_pattern()

        for t in range(10):
            if not pattern[t]:
                assert (
                    expanded_inputs[:, t] == -1.0
                ).all(), f"Masked input should use non_input_value=-1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
