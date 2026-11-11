"""Tests for ``recurrence_target='middle'`` needing a mid feedforward stage.

``RecurrentConnectedConv2d`` supports three places to apply the recurrent
connection: ``'input'`` (before the feedforward convolution), ``'output'``
(after it), and ``'middle'`` (between the two feedforward stages). The third
only exists for a layer that declares ``mid_channels`` -- that is what creates
``conv2`` for the second stage.

``_calculate_recurrence_output_dims`` used to subscript ``self.kernel_size``,
``self.stride`` and ``self.padding`` unconditionally in its ``'middle'`` branch.
Those attributes are only normalised to ``(y, x)`` tuples when
``mid_channels`` is set; otherwise ``kernel_size`` stays a scalar int and the
subscript raises ``TypeError: 'int' object is not subscriptable``.

``DyRCNNx2`` and ``DyRCNNx4`` build every recurrent layer without
``mid_channels``, so ``rctarget=middle`` -- a value in
``config_experiments.yaml``'s ``rctarget`` category -- crashed on those models
rather than reporting the invalid combination. ``DyRCNNx8`` (the configured
``model_name``) sets ``mid_channels`` on all four recurrent layers and is
unaffected.
"""

import pytest

from dynvision.model_components.recurrence import RecurrentConnectedConv2d
from dynvision.models.dyrcnn import DyRCNNx2, DyRCNNx4, DyRCNNx8

BASE = dict(
    n_timesteps=20,
    n_classes=10,
    input_dims=(20, 3, 32, 32),
    dt=2.0,
    tau=5.0,
    t_recurrence=6.0,
    recurrence_type="full",
)


class TestRecurrentConnectedConv2dGuard:
    """The component rejects 'middle' without mid_channels, at construction."""

    def _make_layer(self, **overrides):
        kwargs = dict(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            dim_y=16,
            dim_x=16,
            dt=2.0,
            t_recurrence=6.0,
            recurrence_type="full",
            recurrence_target="middle",
        )
        return RecurrentConnectedConv2d(**{**kwargs, **overrides})

    def test_middle_without_mid_channels_is_rejected(self):
        with pytest.raises(ValueError, match="requires mid_channels"):
            self._make_layer(mid_channels=None)

    def test_error_names_the_alternative(self):
        with pytest.raises(ValueError) as excinfo:
            self._make_layer(mid_channels=None)
        message = str(excinfo.value)
        assert "recurrence_target='output'" in message
        assert "mid_channels" in message

    def test_middle_with_mid_channels_is_accepted(self):
        layer = self._make_layer(mid_channels=8)
        assert layer.recurrence_target == "middle"

    def test_output_without_mid_channels_is_unaffected(self):
        layer = self._make_layer(mid_channels=None, recurrence_target="output")
        assert layer.recurrence_target == "output"

    def test_input_without_mid_channels_is_unaffected(self):
        layer = self._make_layer(mid_channels=None, recurrence_target="input")
        assert layer.recurrence_target == "input"

    def test_invalid_target_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid recurrence target"):
            self._make_layer(mid_channels=8, recurrence_target="sideways")


class TestDyRCNNModels:
    """The crash that motivated the guard, per model variant."""

    @pytest.mark.parametrize("model_cls", [DyRCNNx2, DyRCNNx4])
    def test_middle_is_rejected_on_models_without_mid_channels(self, model_cls):
        """x2 and x4 have no mid_channels, so 'middle' is not representable."""
        with pytest.raises(ValueError, match="requires mid_channels"):
            model_cls(**BASE, recurrence_target="middle")

    @pytest.mark.parametrize("model_cls", [DyRCNNx2, DyRCNNx4, DyRCNNx8])
    def test_output_still_constructs(self, model_cls):
        model = model_cls(**BASE, recurrence_target="output")
        assert model.recurrence_target == "output"

    def test_middle_constructs_on_the_configured_model(self):
        """DyRCNNx8 declares mid_channels on every recurrent layer."""
        model = DyRCNNx8(**BASE, recurrence_target="middle")
        assert model.recurrence_target == "middle"

    def test_dyrcnnx8_is_the_only_variant_with_mid_channels(self):
        """Pin the premise the guard depends on.

        If a model gains or loses mid_channels, the 'middle' verdict for it moves
        with it, and this test says so rather than leaving it implicit.
        """
        with_mid = []
        for name, cls in [("x2", DyRCNNx2), ("x4", DyRCNNx4), ("x8", DyRCNNx8)]:
            model = cls(**BASE, recurrence_target="output")
            layers = [
                mod
                for _, mod in model.named_modules()
                if isinstance(mod, RecurrentConnectedConv2d)
            ]
            if layers and all(mod.mid_channels is not None for mod in layers):
                with_mid.append(name)
        assert with_mid == ["x8"]
