# Skip & Feedback Connections

This reference describes DynVision's generalized skip- and feedback-connection
modules and the `auto_adapt` mechanism. Both are implemented in
`dynvision/model_components/layer_connections.py`.

## Description

Skip and feedback connections describe the same process: a copy of a source
layer's activity is diverted from the immediate feedforward path and integrated
with the signal at a target layer. DynVision provides a single generalized
`ConnectionBase` module for both cases:

- **`Skip`** — adds the activity of an *earlier* layer to the output of a
  *deeper* layer.
- **`Feedback`** — adds the activity of a *deeper* layer to the input of an
  *earlier* layer.

`Skip` and `Feedback` are thin subclasses of `ConnectionBase` and differ only in
this semantic direction; they share the same implementation and integration
strategies as recurrent connections (additive / multiplicative / custom
callable). See [Integration Strategies](integration-strategies.md).

## Retrieving the Source Signal

A connection retrieves the source layer's activity in one of two ways:

- **Explicit hidden state** — call the module as `connection(x, h)`, passing the
  source activity `h` directly.
- **Source module** — initialize with a `source` module and a delay so the
  connection retrieves the correct past hidden state itself when called as
  `connection(x)`. The source must expose a `get_hidden_state(delay_index)`
  method.

The delay is set either directly through `delay_index`, or derived from
`t_connection` and `dt` as `int(t_connection / dt) + 1`. The `+1` accounts for
the fact that earlier layers have already updated their hidden state within the
current timestep.

## Shape Adaptation

When the source activity `h` does not match the target signal `x`, two
transformations are applied to `h`:

1. **Channel adaptation** — a 1×1 convolution maps `h` from its channel count to
   the target's channel count.
2. **Spatial adaptation** — `nn.Upsample` resizes the spatial dimensions of `h`
   to match `x`, using the configured `upsample_mode` (default `"nearest"`).

The `scale_factor` parameter expresses the spatial scaling from `h` to `x`
(`x_size / h_size`): values greater than 1 upsample `h`, values less than 1
downsample it, and 1 leaves it unchanged. When channels already match and
`force_conv` is `False`, no convolution is created.

## Auto-Adapt

Manually matching tensor shapes between source and target layers is tedious and
error-prone — especially during architecture exploration where layers and
operation order change frequently.

Setting `auto_adapt=True` defers shape inference to the first forward pass: the
channel and spatial transforms are constructed once the actual `x` and `h`
shapes are seen. This lets you re-order operations and swap connection targets
without pre-computing dimensions.

!!! note
    Because `auto_adapt` builds its transform lazily on the first forward pass,
    the module docstring warns that inferring shapes during training may break
    gradients and can cause checkpoint issues. For fixed architectures, prefer
    passing `in_channels`, `out_channels`, and `scale_factor` explicitly.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source` | `None` | Source module exposing `get_hidden_state(delay_index)`. |
| `delay_index` | `None` | Explicit hidden-state delay index. |
| `t_connection`, `dt` | `None` | Alternative to `delay_index`; delay derived as `int(t_connection / dt) + 1`. |
| `in_channels`, `out_channels` | `None` | Channel counts for the adaptation conv (inferred from `source` when omitted). |
| `kernel_size` | `1` | Kernel size of the adaptation convolution. |
| `stride` | `1` | Stride of the adaptation convolution. |
| `scale_factor` | `1` | Spatial scaling from `h` to `x`. |
| `bias` | `True` | Whether the adaptation convolution has a bias. |
| `integration_strategy` | `"additive"` | How `h` is combined with `x`. |
| `upsample_mode` | `"nearest"` | Interpolation mode for spatial adaptation. |
| `auto_adapt` | `False` | Infer shapes lazily on the first forward pass. |
| `force_conv` | `False` | Create the adaptation conv even when channels already match. |

## Biological Motivation

Anatomical studies show that feedback projections from higher cortical areas
(e.g. V4, IT) back to lower areas (V2, V1) are as numerous as feedforward
connections, suggesting a fundamental role in visual computation (Felleman &
Van Essen, 1991; Salin & Bullier, 1995). Skip connections bypass intermediate
processing stages, analogous to long-range cortico-cortical projections.

## See Also

- [Layer Operations](layer-operations.md) — where `addskip` and `addfeedback` fit in the pipeline
- [Integration Strategies](integration-strategies.md) — additive / multiplicative / custom
- [Recurrence Types](recurrence-types.md) — lateral recurrent connections
