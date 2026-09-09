"""Regression tests for GitHub issue #11.

``setup("fit")`` crashes in ``_initialize_connections`` whenever a model has
the default ``skip=True`` *and* a nonzero feedforward delay
(``delay_feedforward > 0``). The shape-inference forward pass run by
``_initialize_connections`` is meant to zero out the feedforward delay so
every layer sees a valid (non-``None``) tensor at t=0, but the per-layer
``layer.delay()`` call never receives that override, so the delay buffer
returns ``None`` and the skip connection crashes trying to read ``.shape``.
"""

from unittest.mock import patch

import pytest

from dynvision.models.dyrcnn import DyRCNNx2, DyRCNNx4


class TestSetupWithFeedforwardDelay:
    """setup('fit') must succeed for any valid delay configuration."""

    @pytest.mark.parametrize("model_cls", [DyRCNNx2, DyRCNNx4])
    def test_setup_fit_with_skip_and_feedforward_delay(self, model_cls):
        """Reproduces issue #11: skip=True (default) + delay_feedforward > 0."""
        model = model_cls(
            n_classes=10,
            input_dims=(10, 1, 28, 28),
            tff=2,  # dt defaults to 2.0 -> delay_feedforward = 1 > 0
        )
        assert model.skip is True
        assert model.delay_feedforward > 0

        # Should not raise AttributeError: 'NoneType' object has no attribute 'shape'
        model.setup("fit")

    def test_setup_fit_with_biological_delays(self):
        """Reproduces the realistic biological-delay config from the issue."""
        model = DyRCNNx4(
            n_classes=10,
            input_dims=(10, 1, 28, 28),
            dt=2,
            tau=8,
            tff=10,
            trc=6,
        )
        assert model.delay_feedforward == 5

        model.setup("fit")

    def test_setup_fit_without_feedforward_delay_still_works(self):
        """Baseline: the bare default (delay_feedforward == 0) must keep working."""
        model = DyRCNNx4(n_classes=10, input_dims=(10, 1, 28, 28))
        assert model.delay_feedforward == 0

        model.setup("fit")

    def test_initialize_connections_restores_state_when_forward_raises(self):
        """A failure during shape-inference must not leak override/eval state.

        ``_initialize_connections`` sets ``self._delay_feedforward_override``
        and switches to eval mode for the duration of a probe forward pass.
        If that forward pass raises, both must still be restored (see
        review discussion on #11's fix PR).
        """
        model = DyRCNNx4(n_classes=10, input_dims=(10, 1, 28, 28), tff=2)
        model.train()
        assert model.training is True

        with patch.object(DyRCNNx4, "forward", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                model.setup("fit")

        assert model._delay_feedforward_override is None
        assert model.training is True
