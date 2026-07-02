# Layer Operations

This reference describes the configurable order of operations within each network
layer, the reserved operation names, and how operations are dispatched during the
forward pass.

## Description

DynVision models define each layer as a sequence of operations executed in a
user-configurable order (the `layer_operations` list). This design lets you
control precisely where recurrent, skip, and feedback signals are integrated
relative to feedforward convolutions and the dynamics solver — without modifying
model code.

During each timestep, `TemporalBase` iterates over `layer_names`, and for every
layer it applies each entry of `layer_operations` in order.

## Default Operation Order (DyRCNN)

The DyRCNN model family defines the following operation order
(`dynvision/models/dyrcnn.py`):

```python
self.layer_operations = [
    "layer",         # apply the (recurrent) convolutional layer
    "addext",        # add external input
    "addskip",       # add skip connection
    "addfeedback",   # add feedback connection
    "tstep",         # apply the dynamical-systems ODE solver step
    "nonlin",        # apply the nonlinearity
    "supralin",      # apply the supralinearity
    "record",        # record activations in the responses dict
    "delay",         # set and get delayed activations for the next layer
    "pool",          # apply pooling
]
```

The base default in `TemporalBase` (`dynvision/base/temporal.py`) is the same but
additionally appends a `"norm"` (normalization) operation at the end. This
default is used only if a model does not define its own `layer_operations`.

<p align="center">
  <img src="../../assets/rcnn_architecture.png" alt="Layer Operations and Architecture" width="700"/>
</p>

*Figure: Architecture schematic of the DyRCNN model family. The left side shows
the signal flow within one layer; the right side shows layer parameters and
cross-layer connections. Feedforward, recurrent, skip, and feedback delays are
indicated by pointers into hidden-state timestep slots.*

## Operation Dispatch and Naming Convention

For each operation, `TemporalBase` looks for a module or method named
`{operation}_{layer_name}` (layer-specific), and falls back to `{operation}`
(layer-unspecific). For example, `tstep_V1` is the ODE solver for layer `V1`,
while `nonlin` is a single shared nonlinearity used by every layer.

If neither a layer-specific nor a shared implementation exists for an operation,
that operation is silently skipped for that layer. This allows heterogeneous
layer configurations without extra code paths.

## Reserved Operation Names

The following operation names have dedicated handling in the forward pass:

| Name      | Behaviour |
|-----------|-----------|
| `layer`   | Applies the (recurrent) convolutional layer, passing `feedforward_only` through to it. |
| `tstep`   | Applies the numerical ODE solver step, using the layer's newest hidden state. |
| `delay`   | Writes the current activity into the hidden states and retrieves past activity with the correct feedforward delay. |
| `record`  | Stores the current activity in the responses dictionary (only when response storage is active). |

All other operations (e.g. `addext`, `addskip`, `addfeedback`, `nonlin`,
`supralin`, `pool`, `norm`) are dispatched generically by name using the
convention above.

## Conditional Operation Skipping

Two instance attributes control when operations are skipped:

- **`non_feedforward_operations`** — operations skipped during a
  `feedforward_only` forward pass. Default: `["addfeedback", "addskip"]`. This
  isolates the pure feedforward pathway when measuring the contribution of
  recurrence.

- **`operations_skipped_on_null_input`** — operations skipped when the layer
  input is `None` (a blank timestep). Default:
  `["nonlin", "supralin", "pool", "norm"]`.

Both lists can be overridden per instance or subclass.

## Adding Custom Operations

Each operation corresponds to a method or module attribute on the model. For
layer-specific operations use the naming convention
`self.<operation>_<layer_name> = <module>`; for layer-unspecific operations
(e.g. a shared nonlinearity) the `_<layer_name>` suffix can be omitted.

To insert a custom operation into the pipeline, add its name to
`layer_operations` at the desired position and define the corresponding
attribute in `_define_architecture()`.

## See Also

- [Dynamics Solvers](dynamics-solvers.md)
- [Skip & Feedback Connections](skip-feedback-connections.md)
- [Integration Strategies](integration-strategies.md)
