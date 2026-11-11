# Integration Strategies

This reference describes how recurrent, skip, and feedback signals are combined
with the feedforward signal, and where in the layer that integration happens.

## Description

Recurrent, skip, and feedback connections all produce a signal (`h`) that must be
combined with the layer's feedforward signal (`x`). The combination rule is the
*integration strategy*. Recurrence modules and the `Skip`/`Feedback` connection
modules share the same integration mechanism, implemented in
`dynvision/model_components/integration_strategy.py`.

## Available Strategies

Set via the `integration_strategy` parameter. It accepts either a string
identifier or any callable `f(x, h)`:

| Strategy | Formula | Description |
|----------|---------|-------------|
| `additive` | $x' = x + h$ | Default. Simple addition. |
| `multiplicative` | $x' = x \cdot (1 + \tanh(h))$ | Gain modulation via a tanh-squashed signal. |
| `none` (or `None`) | $x' = x$ | Pass-through; the signal `h` is ignored. |
| *callable* | $x' = f(x, h)$ | Any user-supplied callable, used directly. |

An unrecognized string raises `ValueError`.

<p align="center">
  <img src="../../assets/rcnn_architecture.png" alt="Integration shown in layer architecture" width="700"/>
</p>

## Recurrence Target

The integration **location** is controlled by `recurrence_target`
(`dynvision/model_components/recurrence.py`). It determines where the recurrent
signal is combined with the feedforward pathway:

| Target   | Description |
|----------|-------------|
| `input`  | Integrated with the input tensor to the layer *before* the convolutions. |
| `middle` | Integrated with intermediate activations *between* two convolutions (applicable only to layers with a mid-channel stage). |
| `output` | Default. Integrated with the output tensor *after* all convolutions. |

An unrecognized target raises `ValueError`.

`middle` requires the layer to declare `mid_channels`, which is what creates the
second convolution. Requesting it on a layer without a mid stage raises a
`ValueError` naming the parameter, rather than failing inside the output-size
arithmetic. `CorNetRT` and `DyRCNNx8` set `mid_channels` on their recurrent
layers; `DyRCNNx2` and `DyRCNNx4` do not, so `rctarget=middle` is rejected for
those two models. Setting it through a configuration is also validated: the
field accepts only `input`, `middle` and `output` (case-insensitively).

## Feedback Mode

Feedback connections select their integration strategy through the model's
`feedback_mode` parameter, which is passed as the `integration_strategy` of the
`Feedback` module. It accepts the same `additive` / `multiplicative` options.

## Feedforward-Only Mode

Setting `feedforward_only = True` disables the recurrent, skip, and feedback
integration during a forward pass — useful for measuring the contribution of
recurrence to performance. Internally, the operations listed in
`non_feedforward_operations` (by default `addskip` and `addfeedback`) are
skipped. See [Layer Operations](layer-operations.md).

## See Also

- [Recurrence Types](recurrence-types.md)
- [Skip & Feedback Connections](skip-feedback-connections.md)
- [Layer Operations](layer-operations.md)
