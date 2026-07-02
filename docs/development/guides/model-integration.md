# Model Integration Guide

> **Purpose**: Translate external PyTorch model implementations into DynVision-conformant versions that leverage the framework's temporal dynamics, recurrence mechanisms, and biological features.

This guide walks through the complete process of integrating a model from the literature (e.g., from a paper's reference implementation) into DynVision's architecture.

---

## Overview

DynVision provides a framework for temporal neural networks with:

- **Recurrent connections** within and between layers
- **Temporal dynamics** with configurable delays and time constants
- **Biological features** like skip connections, feedback, and supralinearity
- **Unified forward pass** via operation sequencing in `TemporalBase`

### When to Use This Guide

- Reimplementing a model from a paper to leverage DynVision's features
- Adding pretrained models (e.g., CorNet, ResNet) to the model zoo
- Creating DynVision-compatible versions of custom architectures
- Ensuring architectural equivalence when loading pretrained weights

---

## Phase 1: Investigation - Understand the Original

**Core Principle**: Never begin implementation before fully understanding the original architecture.

### 1.1 Trace the Complete Forward Pass

Read the original model's forward pass **line by line** and document:

```python
# Example: Original CORblock_RT forward pass
def forward(self, inp=None, state=None, batch_size=None):
    # [1] Feedforward input processing
    inp = self.conv_input(inp)      # Conv2d: channels, kernel, stride
    inp = self.norm_input(inp)      # Normalization type and params
    inp = self.nonlin_input(inp)    # Activation function

    # [2] Recurrence point
    if state is None:
        state = 0
    skip = inp + state              # ← How is state integrated?

    # [3] Second stage
    x = self.conv1(skip)            # Conv2d params
    x = self.norm1(x)               # Normalization
    x = self.nonlin1(x)             # Activation

    # [4] State management
    state = self.output(x)          # What gets stored?
    return output, state
```

**Document for each operation**:

- Input/output shapes at each step
- Parameter values (kernel size, stride, padding, bias, channels)
- Operation order (critical for exact replication)
- Where recurrence is applied (before/after which operations?)
- How state is stored and retrieved

### 1.2 Identify Layer Structure Patterns

Determine the layer organization:

```python
# Original CorNet structure
class CORnet_RT:
    def __init__(self, times=5):
        self.V1 = CORblock_RT(3, 64, kernel_size=7, stride=4)
        self.V2 = CORblock_RT(64, 128, stride=2)
        self.V4 = CORblock_RT(128, 256, stride=2)
        self.IT = CORblock_RT(256, 512, stride=2)
        self.decoder = nn.Sequential(...)
```

**Questions to answer**:

- How many layers/stages?
- Are they similar or heterogeneous?
- What are the channel progressions?
- What are the spatial dimension reductions?
- Is there a pattern that can be generalized?

### 1.3 Analyze Temporal Dynamics

For recurrent models, understand the time loop:

```python
# Original CorNet time loop
for t in range(1, self.times):
    for block in blocks[1:]:
        prev_output = outputs[prev_block]  # Feedforward input
        prev_state = states[block]          # Recurrent state
        new_output, new_state = getattr(self, block)(prev_output, prev_state)
        outputs[block] = new_output
        states[block] = new_state
```

**Document**:

- How many timesteps?
- What initializes at t=0?
- What is passed between timesteps?
- Where does feedforward input come from?
- Where does recurrent input come from?
- What delay is there between storing and retrieving state?

### 1.4 Check Data Preprocessing

Critical for loading pretrained weights:

```python
# Look for data transformations in training scripts
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # ← Check if normalization is applied!
    # transforms.Normalize(mean=[...], std=[...])  # Present or not?
])
```

**Document**:

- Input resolution expected
- Normalization parameters (mean, std) or absence thereof
- Other preprocessing (resizing, cropping, augmentation)
- Expected input range (0-1, 0-255, normalized)

### 1.5 Review Pretrained Weights (if applicable)

If loading pretrained weights:

```python
# Load and inspect
state_dict = torch.load("pretrained.pth")
print(state_dict.keys())

# Check layer names, shapes
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
```

**Document**:

- Layer naming convention
- Which layers have pretrained weights
- Which layers need to be randomly initialized (e.g., classifier for different n_classes)
- State dict structure (nested? module. prefix?)

---

## Phase 2: Mapping - Find DynVision Equivalents

### 2.1 Choose the Base Class

DynVision provides two main base classes:

| Base Class | Use When |
|------------|----------|
| `BaseModel` | Standard models with minimal temporal dynamics |
| `DyRCNN` | Models with biological features (skip, feedback, retina, tau learning) |

```python
from dynvision.base import BaseModel
# OR
from dynvision.models.dyrcnn import DyRCNN

class MyModel(BaseModel):  # or DyRCNN
    def __init__(self, ...):
        super().__init__(...)
```

**Decision factors**:

- Does the model have recurrent connections? → Consider BaseModel with RConv2d
- Need skip connections between non-adjacent layers? → DyRCNN
- Need feedback connections? → DyRCNN
- Need learnable time constants (tau)? → DyRCNN
- Simple feedforward with optional recurrence? → BaseModel

### 2.2 Map Convolutional Blocks to RConv2d

DynVision's `RConv2d` (RecurrentConnectedConv2d) can represent various block types:

#### **Single-Stage Convolution**

```python
# Original: Simple conv → norm → activation
class Block:
    def __init__(self):
        self.conv = nn.Conv2d(in_c, out_c, k, s, p)
        self.norm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

# DynVision equivalent
from dynvision.model_components import RConv2d

layer = RConv2d(
    in_channels=in_c,
    out_channels=out_c,
    kernel_size=k,
    stride=s,
    padding=p,
    mid_channels=None,  # ← Single-stage
    recurrence_type="none",  # No recurrence
)
# Then apply norm and activation via layer_operations
```

#### **Two-Stage Convolution** (e.g., CorNet blocks)

```python
# Original: conv1 → norm1 → relu1 → [recurrence] → conv2 → norm2 → relu2
class CORblock:
    def __init__(self):
        self.conv_input = nn.Conv2d(in_c, mid_c, k1, s1, p1, bias=True)
        self.norm_input = nn.GroupNorm(32, mid_c)
        self.nonlin_input = nn.ReLU()
        # [recurrence applied here]
        self.conv1 = nn.Conv2d(mid_c, out_c, k2, s2, p2, bias=False)
        self.norm1 = nn.GroupNorm(32, out_c)
        self.nonlin1 = nn.ReLU()

# DynVision equivalent
layer = RConv2d(
    in_channels=in_c,
    mid_channels=mid_c,      # ← Enables two-stage
    out_channels=out_c,
    kernel_size=(k1, k2),    # ← Tuple for two stages
    stride=(s1, s2),
    bias=(True, False),      # ← Per-stage bias
    mid_modules=nn.GroupNorm(32, mid_c),  # Applied after first conv
    recurrence_target="middle",  # Recurrence between stages
    recurrence_type="self",
    fixed_self_weight=1.0,
    recurrence_bias=False,
)
```

**Key RConv2d parameters**:

| Parameter | Purpose | Values |
|-----------|---------|--------|
| `mid_channels` | Enables two-stage conv | `None` (single) or int (two-stage) |
| `kernel_size` | Kernel size(s) | int or tuple `(k1, k2)` |
| `stride` | Stride(s) | int or tuple `(s1, s2)` |
| `bias` | Bias per stage | bool or tuple `(b1, b2)` |
| `mid_modules` | Operations after first conv | `nn.Module` (e.g., GroupNorm) |
| `recurrence_target` | Where recurrence applies | `"input"`, `"middle"`, `"output"` |
| `recurrence_type` | Type of recurrence | `"none"`, `"self"`, `"full"`, etc. |
| `t_recurrence` | Recurrence delay | float (ms) |
| `fixed_self_weight` | Fixed recurrence weight | float or None (learnable) |

### 2.3 Understand recurrence_target

**Critical**: This determines where the recurrent connection is applied.

```python
# recurrence_target="input"
# Flow: [+ recurrence] → conv → ... → conv2

# recurrence_target="middle" (CorNet case)
# Flow: conv → mid_modules → nonlin → [+ recurrence] → conv2

# recurrence_target="output"
# Flow: conv → ... → conv2 → [+ recurrence]
```

**To determine correct setting**:

1. Trace where `state` or recurrent input is added in original
2. Count operations before/after that point
3. Match to DynVision's structure

**Debugging recurrence_target issues:**

If outputs don't match despite correct weights, trace EXACTLY where state is added:

```python
# Original code - count operations before/after recurrence
# Example from CorNet-RT:
inp = self.conv_input(inp)      # [1] First conv
inp = self.norm_input(inp)      # [2] First norm
inp = self.nonlin_input(inp)    # [3] First activation
skip = inp + state              # [← RECURRENCE ADDED HERE]
x = self.conv1(skip)            # [4] Second conv
x = self.norm1(x)               # [5] Second norm
x = self.nonlin1(x)             # [6] Second activation

# Count: 3 operations before recurrence, 3 after
# This is two-stage conv with recurrence in the middle
# → recurrence_target="middle"
```

**Common mistake**: Assuming recurrence_target based on intuition rather than counting operations in the original code.

### 2.4 Map Normalization and Activation

DynVision separates operations into `layer_operations`:

```python
# Original combined in block
x = conv(x)
x = norm(x)
x = relu(x)

# DynVision: Define operations and let TemporalBase orchestrate
self.layer_operations = [
    "layer",   # Calls RConv2d
    "norm",    # Calls self.norm_LayerName if exists
    "nonlin",  # Calls self.nonlin if exists
    "record",  # Records activations
    "delay",   # Manages hidden state
]

# Define the components
self.LayerName = RConv2d(...)           # The convolution
self.norm_LayerName = nn.GroupNorm(...) # External normalization
self.nonlin = nn.ReLU(inplace=True)     # Shared activation
```

**Decision**: Normalization inside vs. outside RConv2d?

- **Inside** (`mid_modules`): Applied between conv stages in two-stage RConv2d
- **Outside** (`norm_LayerName`): Applied after RConv2d via `layer_operations`

For CorNet, we use both:

- `mid_modules=nn.GroupNorm(...)` for first-stage normalization
- `self.norm_V1 = nn.GroupNorm(...)` for second-stage normalization

### 2.5 Map State Management

DynVision handles state automatically via `ForwardRecurrenceBase`:

```python
# Original manual state management
def forward(self, inp, state=None):
    if state is None:
        state = 0
    x = process(inp)
    x = x + state  # Add recurrent input
    state = x      # Store for next timestep
    return x, state

# DynVision automatic handling
# RConv2d internally:
# - Retrieves: h = self.get_hidden_state(self.delay_recurrence)
# - Processes: h = self.recurrence(h)
# - Integrates: x = self.integrate_signal(x, h)
# - Stores: self.set_hidden_state(x)  # via "delay" operation
```

**Key methods** (you typically don't call these directly):

- `get_hidden_state(delay)`: Retrieve state from `delay` timesteps ago
- `set_hidden_state(x)`: Store current state for future retrieval
- `reset()`: Clear all hidden states (call this at start of forward pass)

### 2.6 Temporal Parameter Propagation

**Critical**: Understanding which temporal parameters to pass to RConv2d and which are handled elsewhere.

DynVision separates temporal dynamics into two distinct concerns:

1. **Within-layer dynamics** (handled by RConv2d): Recurrence delay
2. **Between-layer dynamics** (handled by delay operation in `layer_operations`): Feedforward delay

**Parameter Scope**:

```python
# Model-level temporal parameters (in __init__)
self.dt = 2.0                # Integration time step (ms)
self.t_feedforward = 2.0     # Delay BETWEEN layers (ms)
self.t_recurrence = 2.0      # Delay WITHIN layer (ms)
self.history_length = int(max(t_recurrence, t_feedforward) / dt) + 1

# Pass to RConv2d:
self.V1 = RConv2d(
    # ... other params ...
    dt=self.dt,                     # ✓ PASS: Needed for delay calculation
    t_recurrence=self.t_recurrence, # ✓ PASS: Sets delay_recurrence internally
    # t_feedforward=self.t_feedforward  # ✗ DO NOT PASS: Handled by delay operation!
    history_length=self.history_length,  # ✓ PASS: Must account for BOTH delays
)
```

**Why t_feedforward is NOT passed to RConv2d**:

Feedforward delays are handled by the `"delay"` operation in `layer_operations`, not by RConv2d:

```python
# In temporal.py _forward() method:
for operation in self.layer_operations:
    if operation == "delay":
        # 1. Store current layer output
        layer.set_hidden_state(x)

        # 2. Retrieve delayed output for next layer
        x = layer.get_hidden_state(delay=self.delay_feedforward)
        # This is where t_feedforward is used!
```

The delay operation sits BETWEEN layers and manages the feedforward delay. RConv2d only needs to know about its own recurrent delay.

**Why history_length must account for BOTH delays**:

Even though RConv2d doesn't use `t_feedforward` internally, its hidden state buffer must be large enough to store states for retrieval at both delay points:

```python
# Example with dt=2, t_feedforward=2, t_recurrence=2:
delay_feedforward = int(2.0 / 2.0) = 1  # Used by delay operation
delay_recurrence = int(2.0 / 2.0) = 1   # Used by RConv2d internally

# Both delays might need to retrieve from the same history:
history_length = int(max(2.0, 2.0) / 2.0) + 1 = 2
# Buffer needs 2 timesteps to support both delay=1 retrievals
```

**Separation of Concerns**:

```python
# RConv2d handles:
# - Recurrent connections (state from this layer's past)
# - Internal parameter: delay_recurrence = int(t_recurrence / dt)
# - Gets hidden state: h = self.get_hidden_state(self.delay_recurrence)
# - Applies recurrence: x = x + self.recurrence(h)

# Delay operation (in temporal.py) handles:
# - Feedforward delays (output to next layer)
# - Model-level parameter: delay_feedforward = int(t_feedforward / dt)
# - Sets state: layer.set_hidden_state(x)
# - Gets delayed state: x = layer.get_hidden_state(delay_feedforward)
```

**Common Mistake**: Parameter Scope Confusion

**Symptom**: Initially passing all temporal parameters to RConv2d, or not passing any.

**Wrong Approach**:
```python
# Passing everything (incorrect)
self.V1 = RConv2d(
    dt=self.dt,
    t_feedforward=self.t_feedforward,  # ✗ RConv2d doesn't handle this!
    t_recurrence=self.t_recurrence,
)

# Passing nothing (incorrect)
self.V1 = RConv2d(
    # ✗ Missing dt and t_recurrence - delay_recurrence won't be calculated!
)
```

**Correct Approach**:
```python
# Calculate shared history_length once
self.history_length = int(max(self.t_recurrence, self.t_feedforward) / self.dt) + 1

# Pass only what RConv2d needs
self.V1 = RConv2d(
    in_channels=3,
    out_channels=64,
    dt=self.dt,                     # For internal delay calculation
    t_recurrence=self.t_recurrence, # Sets delay_recurrence
    history_length=self.history_length,  # Large enough for both delays
    dim_y=self.dim_y,
    dim_x=self.dim_x,
)
```

**Debugging temporal parameter issues**:

If recurrence doesn't work as expected, check:

```python
# Add debug method to your model
def debug_temporal_params(self):
    print(f"Model level:")
    print(f"  dt: {self.dt}")
    print(f"  t_feedforward: {self.t_feedforward}")
    print(f"  t_recurrence: {self.t_recurrence}")
    print(f"  delay_feedforward: {self.delay_feedforward}")
    print(f"  history_length: {self.history_length}")
    print()

    for layer_name in self.layer_names:
        layer = getattr(self, layer_name)
        print(f"{layer_name}:")
        print(f"  dt: {layer.dt}")
        print(f"  t_recurrence: {layer.t_recurrence}")
        print(f"  delay_recurrence: {layer.delay_recurrence}")
        print(f"  history_length: {layer.history_length}")

# Expected output for CorNet-RT defaults (dt=2, t_ff=2, t_rec=2):
# Model level:
#   dt: 2.0
#   t_feedforward: 2.0
#   t_recurrence: 2.0
#   delay_feedforward: 1
#   history_length: 2
#
# V1:
#   dt: 2.0
#   t_recurrence: 2.0
#   delay_recurrence: 1  # ← Should be 1, not 0!
#   history_length: 2
```

**Key Insight**: The separation between within-layer (RConv2d) and between-layer (delay operation) temporal dynamics allows each component to focus on one responsibility, following the single responsibility principle.

---

## Phase 3: Implementation - Build the DynVision Version

### 3.1 Create the Model Class

```python
"""
MyModel: Description of the model and its purpose.

This module implements [model name] from [paper citation], adapted to
DynVision's temporal dynamics framework.

References:

- Author et al. (Year) "Paper Title"
"""

import logging
from typing import Optional, Dict, Any, Tuple, Union, List

import torch
import torch.nn as nn

from dynvision.base import BaseModel  # or DyRCNN
from dynvision.model_components import RConv2d

__all__ = ["MyModel"]

logger = logging.getLogger(__name__)


class MyModel(BaseModel):
    def __init__(
        self,
        # Standard DynVision parameters
        n_timesteps: int = 5,
        input_dims: Tuple[int, int, int] = (5, 3, 224, 224),
        n_classes: int = 1000,
        dt: float = 2,  # ms
        t_feedforward: float = 2,  # ms
        t_recurrence: float = 2,  # ms
        recurrence_type: str = "self",
        # Model-specific parameters
        init_with_pretrained: bool = True,
        **kwargs: Any,
    ) -> None:

        # Store model-specific attributes
        self.init_with_pretrained = init_with_pretrained

        # Pass to parent
        super().__init__(
            n_timesteps=n_timesteps,
            input_dims=input_dims,
            n_classes=n_classes,
            dt=dt,
            t_feedforward=t_feedforward,
            t_recurrence=t_recurrence,
            recurrence_type=recurrence_type,
            **kwargs,
        )
```

### 3.2 Define Architecture in `_define_architecture()`

This is the core method where you define the model structure:

```python
def _define_architecture(self):
    """Define model architecture following DynVision conventions."""

    # [1] Define layer names (must match attribute names!)
    self.layer_names = ["V1", "V2", "V4", "IT"]

    # [2] Define operation sequence
    self.layer_operations = [
        "layer",   # Apply the layer (RConv2d)
        "norm",    # Apply norm_LayerName if it exists
        "nonlin",  # Apply self.nonlin if it exists
        "record",  # Record activations (if store_responses=True)
        "delay",   # Manage hidden states for recurrence
    ]

    # [3] Define shared modules
    self.nonlin = nn.ReLU(inplace=True)

    # [4] Define layers
    self.V1 = RConv2d(
        in_channels=3,
        mid_channels=64,
        out_channels=64,
        kernel_size=(7, 3),
        stride=(4, 1),
        bias=(True, False),
        mid_modules=nn.GroupNorm(32, 64),
        recurrence_type=self.recurrence_type,
        fixed_self_weight=1.0,
        recurrence_bias=False,
        dim_y=self.dim_y,
        dim_x=self.dim_x,
        history_length=int(max(self.t_recurrence, self.t_feedforward) / self.dt) + 1,
    )
    self.norm_V1 = nn.GroupNorm(32, 64)

    # [5] Define classifier
    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, self.n_classes),
    )
```

**Key points**:

- `layer_names` must exactly match the attribute names you define
- `layer_operations` defines the execution order
- Operations look for attributes named `{operation}_{layer_name}` (e.g., `norm_V1`)
- Or fall back to shared operations (e.g., `self.nonlin`)
- Dimensions (`dim_y`, `dim_x`) are calculated in `_process_input_dimensions()`

### 3.3 Implement `reset()`

Clear hidden states at the start of each forward pass:

```python
def reset(self, input_shape: Optional[Tuple[int, ...]] = None) :
    """Reset all hidden states in recurrent layers."""
    for layer in [self.V1, self.V2, self.V4, self.IT]:
        layer.reset()
```

### 3.4 Handle Pretrained Weights (Optional)

If loading pretrained weights:

```python
def _init_parameters(self) -> None:
    """Initialize or load parameters."""
    if self.init_with_pretrained:
        self.load_pretrained_state_dict(
            check_mismatch_layer=["classifier.2"]  # Layers that may differ
        )
        # Only train classifier if rest is pretrained
        self.trainable_parameter_names = [
            p for p in list(self.state_dict().keys())
            if "classifier.2" in p
        ]
    else:
        # Train all parameters
        self.trainable_parameter_names = list(self.state_dict().keys())

def download_pretrained_state_dict(self):
    """Download pretrained weights."""
    url = "https://example.com/model_weights.pth"
    ckpt_data = torch.utils.model_zoo.load_url(url, map_location=self.device)
    state_dict = ckpt_data["state_dict"]
    # Remove module. prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict

def translate_pretrained_layer_names(self):
    """Map original layer names to DynVision names."""
    return {
        # Original name pattern: DynVision name pattern
        "conv_input": "conv",
        "norm_input": "mid_modules",
        "conv1": "conv2",
        "norm1.weight": "norm_LayerName.weight",
        "norm1.bias": "norm_LayerName.bias",
        "decoder.linear": "classifier.2",
    }
```

**Translation process**:

1. `download_pretrained_state_dict()`: Download and load original weights
2. `translate_pretrained_layer_names()`: Define name mappings
3. `load_pretrained_state_dict()`: Automatically applies translation (inherited from BaseModel)
4. `check_mismatch_layer`: Specify layers where shape might differ (e.g., classifier with different n_classes)

### 3.5 Verify Operation Equivalence

**Critical check**: Ensure your DynVision implementation produces the same operation sequence as the original:

```python
# Original sequence (example: CorNet block per timestep)
1. conv_input → norm_input → nonlin_input
2. + state  (recurrence)
3. conv1 → norm1 → nonlin1
4. store state

# Your DynVision sequence (must match!)
# Within RConv2d (recurrence_target="middle"):
1. self.conv → self.mid_modules → self.nonlin
2. + recurrence(h)  where h = get_hidden_state(delay_recurrence)
3. self.conv2
# Via layer_operations:
4. norm_V1  (second normalization)
5. nonlin   (second activation)
6. set_hidden_state(x)  (via "delay" operation)
```

**Method to verify**: Add print statements or use `torch.set_printoptions` and compare intermediate outputs with the original model on the same input.

### 3.6 Handling None in Bias Modules

**Critical for exact replication**: Bias modules must handle `None` input correctly during idle timesteps.

**The Issue**:

During idle timesteps before a layer activates, DynVision may pass `None` through the operation sequence. Standard PyTorch modules typically propagate `None`, but some original models expect to output bias terms even without input:

```python
# Original CordsNet behavior during idle timesteps:
def forward(self, inp=None):
    if inp is None:
        # Still return bias, not None!
        return self.bias.view(1, -1, *self.bias.shape[1:])
    return inp + self.bias
```

**Solution**: Override bias module behavior to return bias when input is None:

```python
from dynvision.model_components import AddBias

class MyModel(BaseModel):
    def _define_architecture(self):
        # ... other layers ...

        # Create bias modules
        for i in range(1, 9):
            layer_name = f"layer{i}"
            # AddBias automatically handles None by returning bias
            setattr(self, f"addbias_{layer_name}",
                   AddBias(out_channels, dim_y, dim_x))
```

**When to use this**:

Check if the original model returns non-None values during initialization before layers receive their first input. If the original has code like:

```python
if inp is None:
    return some_initial_value  # Not None
```

Then you need to ensure your bias modules (or other modules) follow the same pattern.

### 3.7 Per-Layer Temporal Parameters

**Critical for complex temporal architectures**: Not all layers need the same feedforward delays.

**The Standard Case** (covered in Section 2.6):

Most models have uniform delays:
```python
# All layers use same t_feedforward
self.t_feedforward = 1.0
# Passed to delay operation in temporal.py
```

**The Special Case** (CordsNet-like):

Some architectures need per-layer variation:

1. **Input layer with immediate skip**: Layer0 output must be immediately available
   ```python
   self.layer0 = RConv2d(
       # ... other params ...
       t_feedforward=0,  # Immediate output
   )
   ```

2. **Final layer with immediate output**: Layer8 feeds directly to classifier
   ```python
   self.layer8 = RConv2d(
       # ... other params ...
       t_feedforward=0,  # Immediate to classifier
   )
   ```

3. **Combined layer outputs**: When combining outputs from layers processed in same timestep
   ```python
   # Layer7 + Layer8 → Classifier
   # Layer8 processed after Layer7, so Layer8 is immediate
   # But Layer7's output was already written to hidden state

   def combine_outputs(self):
       layer8_out = self.layer8.get_hidden_state(delay=0)  # Immediate
       layer7_out = self.layer7.get_hidden_state(delay=1)  # From hidden state
       return layer8_out + layer7_out
   ```

**Implementation Pattern**:

For per-layer delays, add a `delay()` method to RConv2d and refactor the delay operation in `temporal.py`:

```python
# In temporal.py _forward() method:
elif operation == "delay":
    # Try layer-specific delay function first
    if hasattr(self, f"delay_{layer_name}"):
        delay_func = getattr(self, f"delay_{layer_name}")
        x = delay_func(x)
    # Then try layer's delay method
    elif hasattr(layer, "delay"):
        x = layer.delay(x, delay_feedforward=self.delay_feedforward)
    # Fall back to standard implementation
    elif hasattr(layer, "set_hidden_state"):
        layer.set_hidden_state(x)
        if self.delay_feedforward > 0:
            x = layer.get_hidden_state(self.delay_feedforward + 1)
```

**When you need this**:

- Skip connections requiring immediate input (layer0 → layer2)
- Multi-layer combinations before classifier
- Heterogeneous delay patterns in the original model

### 3.8 Hidden State Initialization

**Critical for exact replication**: Some original models return specific values (e.g., scalar 0) for uninitialized hidden states, while DynVision returns `None` by default.

**Understanding the difference**:

```python
# DynVision default behavior (ForwardRecurrenceBase.get_hidden_state):
def get_hidden_state(self, delay=None):
    if delay is None:
        delay = 0
    if delay >= len(self._hidden_states):
        return None  # ← Uninitialized state returns None
    return self._hidden_states[delay]

# Some original models (e.g., CorNet-RT):
def forward(self, inp=None, state=None):
    if state is None:
        state = 0  # ← Uninitialized state returns scalar 0
    x = inp + state  # Works with scalar 0 (additive identity)
    return x, x
```

**Why this matters**:

For models loading pretrained weights, this difference can cause subtle divergence:

1. **Scalar 0**: Acts as additive identity, cleanly integrates without special handling
2. **None**: Requires explicit checks in integration logic

If the original uses scalar 0 (or similar sentinel values), you must override DynVision's default behavior.

**Solution: Override reset() to patch state initialization**

For CorNet-RT, we needed to patch both `get_hidden_state()` and the recurrence's `forward()`:

```python
def reset(self, input_shape: Optional[Tuple[int, ...]] = None) :
    """Reset hidden states and patch to return 0 instead of None.

    Original CorNet-RT initializes state=0, not None. We patch both
    get_hidden_state and recurrence.forward to handle scalar 0.
    """
    for layer in [self.V1, self.V2, self.V4, self.IT]:
        # Standard reset
        layer.reset()

        # Patch 1: get_hidden_state returns 0 instead of None
        if hasattr(layer, 'get_hidden_state'):
            if not hasattr(layer, '_original_get_hidden_state'):
                layer._original_get_hidden_state = layer.get_hidden_state

            def get_with_zero(_layer=layer):
                def get_hidden_state(delay=None):
                    h = _layer._original_get_hidden_state(delay)
                    return 0 if h is None else h
                return get_hidden_state

            layer.get_hidden_state = get_with_zero()

        # Patch 2: SelfConnection handles scalar 0 (passes through)
        if hasattr(layer, 'recurrence') and layer.recurrence is not None:
            recurrence = layer.recurrence

            if not hasattr(recurrence, '_original_forward'):
                recurrence._original_forward = recurrence.forward

            def forward_with_zero(_rec=recurrence):
                def forward(x):
                    # Handle scalar 0 (uninitialized state)
                    if isinstance(x, (int, float)) and x == 0:
                        return x  # Return 0 as-is (additive identity)
                    return _rec._original_forward(x)
                return forward

            recurrence.forward = forward_with_zero()
```

**When to use this approach**:

Check the original model's state initialization:

```python
# Look for patterns like:
if state is None:
    state = 0           # → Need to patch to return 0
    # OR
    state = torch.zeros(...)  # → DynVision handles this fine
    # OR
    state = some_initial_value  # → May need custom initialization
```

**Alternative: Sentinel class approach** (for more complex cases):

```python
class ZeroStateSentinel:
    """Sentinel representing zero state for uninitialized hidden states.

    This allows returning 'zero' without knowing the batch size.
    Operations check isinstance and treat it as additive identity.
    """
    pass

# In integration operations:
def integrate_signal(self, x, h):
    if isinstance(h, ZeroStateSentinel) or (isinstance(h, (int, float)) and h == 0):
        return x  # h = 0, so x + 0 = x
    return x + h  # Normal integration
```

**Debugging state initialization issues**:

If outputs diverge at the first timestep despite matching weights:

```python
# Add debug logging to check state retrieval
def reset(self, input_shape: Optional[Tuple[int, ...]] = None) :
    for layer in [self.V1, self.V2, self.V4, self.IT]:
        layer.reset()

        # Temporarily add logging
        original_get = layer.get_hidden_state
        def logged_get(delay=None, _layer=layer):
            h = original_get(delay)
            print(f"{_layer.__class__.__name__} get_hidden_state({delay}): {type(h)} = {h}")
            return h
        layer.get_hidden_state = logged_get

# Run model and check what states are retrieved:
# Original model: state = 0 (scalar)
# DynVision default: state = None
# → Need to patch!
```

**Key Insight**: When reimplementing models with pretrained weights, state initialization semantics matter. A scalar 0 vs None difference can propagate through the forward pass and cause divergence, especially in the first few timesteps before real states are stored.

---

## Phase 4: Validation - Ensure Correctness

### 4.1 Test Model Creation

```python
def test_model_creation():
    """Test that model can be instantiated."""
    model = MyModel(
        input_dims=(5, 3, 224, 224),
        n_classes=1000,
    )
    model.setup("fit")

    assert hasattr(model, 'V1')
    assert hasattr(model, 'classifier')
    print("✓ Model creation successful")
```

### 4.2 Test Forward Pass

```python
def test_forward_pass():
    """Test forward pass with random input."""
    model = MyModel(input_dims=(5, 3, 224, 224))
    model.setup("fit")

    x = torch.randn(2, 5, 3, 224, 224)  # batch=2, timesteps=5
    y = model(x)

    assert y.shape == (2, 5, 1000), f"Expected (2, 5, 1000), got {y.shape}"
    assert torch.isfinite(y).all(), "Output contains NaN or Inf"
    print(f"✓ Forward pass successful: {x.shape} → {y.shape}")
```

### 4.3 Test Pretrained Weight Loading

```python
def test_pretrained_loading():
    """Test loading pretrained weights."""
    model = MyModel(
        input_dims=(5, 3, 224, 224),
        init_with_pretrained=True,
    )
    model.setup("fit")

    # Check that weights were loaded (not random initialization values)
    # You can check specific layer statistics or weight ranges
    v1_weight = model.V1.conv.weight
    assert v1_weight.abs().mean() > 0.001, "Weights appear uninitialized"
    print("✓ Pretrained weights loaded successfully")
```

### 4.4 Compare with Original (Gold Standard)

**Most important validation**: Compare outputs with the original implementation.

```python
def test_equivalence():
    """Compare DynVision implementation with original."""
    from models.original_model import OriginalModel

    # Load same pretrained weights into both
    original = OriginalModel()
    original.load_state_dict(torch.load("pretrained.pth"))
    original.eval()

    dynvision = MyModel(init_with_pretrained=True)
    dynvision.setup("fit")
    dynvision.eval()

    # Same random seed for same input
    torch.manual_seed(42)
    x = torch.randn(1, 5, 3, 224, 224)

    with torch.no_grad():
        y_original = original(x[:, 0])  # If original doesn't handle time
        y_dynvision = dynvision(x)[:, 0]  # First timestep

    # Check outputs are close
    diff = (y_original - y_dynvision).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    # Tolerances depend on precision (float32 vs float16)
    assert max_diff < 1e-4, f"Outputs differ too much: {max_diff}"
    print("✓ Outputs match original implementation")
```

**Common reasons for mismatch**:

1. **Wrong operation order**: Check `layer_operations` sequence
2. **Wrong recurrence_target**: Check where `state` is added in original
3. **Missing normalization/activation**: Check all operations are present
4. **Wrong initialization**: Ensure pretrained weights loaded correctly
5. **Numerical precision**: float16 vs float32, check with `dtype` parameter

### 4.5 Test with Different Configurations

```python
def test_configurations():
    """Test various model configurations."""
    configs = [
        {"recurrence_type": "none"},
        {"recurrence_type": "self"},
        {"recurrence_type": "full"},
        {"n_timesteps": 1},
        {"n_timesteps": 10},
        {"feedforward_only": True},
    ]

    for config in configs:
        print(f"Testing config: {config}")
        model = MyModel(
            input_dims=(5, 3, 224, 224),
            **config
        )
        model.setup("fit")

        x = torch.randn(1, 5, 3, 224, 224)
        y = model(x)

        assert torch.isfinite(y).all()
        print(f"  ✓ Passed")
```

### 4.6 Incremental Debugging Methodology

When outputs don't match despite all checks passing, use a structured debugging approach to isolate the issue.

**The Debugging Hierarchy** (Apply in order):

#### Level 0: Verify Translation Mapping Completeness

Before checking weight structure, ensure the translation mapping handles all keys:

```python
def verify_translation_mapping(model):
    """Verify translation mapping covers all pretrained keys."""
    # Get pretrained state dict
    pretrained_state = model.download_pretrained_state_dict()
    translation = model.translate_pretrained_layer_names()

    print(f"Pretrained state dict has {len(pretrained_state)} keys")
    print(f"Translation mapping has {len(translation)} entries\n")

    # Check if all pretrained keys have a translation
    untranslated = []
    for key in pretrained_state.keys():
        # Check if any translation pattern matches this key
        translated = False
        for orig_pattern in translation.keys():
            if orig_pattern in key:
                translated = True
                break
        if not translated:
            untranslated.append(key)

    if untranslated:
        print(f"❌ {len(untranslated)} keys have no translation:")
        for key in untranslated[:10]:
            print(f"  {key}")
        return False
    else:
        print(f"✅ All {len(pretrained_state)} pretrained keys have translations")
        return True
```

**Critical for models with weight_norm parametrization**:

If pretrained weights use `weight_norm`, keys look like:
```
layer1.conv.parametrizations.weight.original0
layer1.conv.parametrizations.weight.original1
```

Your translation must handle the **full path**, not just the base:
```python
# WRONG: Only maps base layer name
translate_layer_names = {
    "layer1": "v1",  # Won't match parametrized keys!
}

# RIGHT: Maps the conv part, parametrization suffix auto-matches
translate_layer_names = {
    "layer1.conv": "v1.conv",  # Matches layer1.conv.parametrizations.weight.*
}
```

The translation uses **substring replacement**, so `layer1.conv` in `layer1.conv.parametrizations.weight.original0` gets replaced to create `v1.conv.parametrizations.weight.original0`.

#### Level 1: Verify Weight Loading

Before testing outputs, ensure weights are actually loaded correctly:

```python
def compare_weights(original, reimpl):
    """Compare all pretrained parameters between models."""
    orig_state = original.state_dict()
    reimpl_state = reimpl.state_dict()

    # Get only matching layers (exclude classifier if different n_classes)
    common_keys = set(orig_state.keys()) & set(reimpl_state.keys())

    mismatches = []
    for key in sorted(common_keys):
        orig_param = orig_state[key]
        reimpl_param = reimpl_state[key]

        if orig_param.shape != reimpl_param.shape:
            mismatches.append(f"{key}: shape mismatch {orig_param.shape} vs {reimpl_param.shape}")
            continue

        diff = (orig_param - reimpl_param).abs().max().item()
        if diff > 1e-6:
            mismatches.append(f"{key}: max diff = {diff:.2e}")

    if mismatches:
        print("Weight mismatches found:")
        for m in mismatches:
            print(f"  ✗ {m}")
        return False
    else:
        print(f"✓ All {len(common_keys)} pretrained parameters match exactly")
        return True
```

**Key checks**:

- All expected parameters present?
- Shapes match?
- Values identical (within floating point precision)?
- Translation mapping correct? (Check `translate_pretrained_layer_names()`)

#### Level 1b: Verify Loaded Weight VALUES (Critical!)

**The Issue**: Level 1 checks that weights CAN be loaded (structure matches), but doesn't verify they WERE loaded. Weights might remain randomly initialized if loading silently fails.

```python
def verify_weights_actually_loaded(model_class):
    """Create fresh model with pretrained loading and verify values transferred."""
    from models.original_model import OriginalModel

    # Get original pretrained weights
    original = OriginalModel()
    original.load_state_dict(torch.load("pretrained.pth"))
    orig_state = original.state_dict()

    # Create reimpl with pretrained loading ENABLED
    reimpl = model_class(
        init_with_pretrained=True,  # ← CRITICAL: Must enable loading
    )
    reimpl.setup("fit")
    reimpl_state = reimpl.state_dict()
    translation = reimpl.translate_pretrained_layer_names()

    # Compare ACTUAL VALUES for each translated parameter
    perfect_matches = 0
    mismatches = []

    for orig_base_key, reimpl_base_pattern in translation.items():
        # Find all original keys (handles weight_norm variations)
        orig_keys = [k for k in orig_state.keys() if k.startswith(orig_base_key)]

        for orig_key in orig_keys:
            # Build expected reimpl key via translation
            expected_reimpl_key = orig_key.replace(orig_base_key, reimpl_base_pattern)

            if expected_reimpl_key not in reimpl_state:
                mismatches.append({
                    'orig_key': orig_key,
                    'expected_key': expected_reimpl_key,
                    'issue': 'Key not found in reimpl state dict'
                })
                continue

            # Compare VALUES
            orig_val = orig_state[orig_key]
            reimpl_val = reimpl_state[expected_reimpl_key]

            max_diff = (orig_val - reimpl_val).abs().max().item()

            if max_diff < 1e-6:
                perfect_matches += 1
            else:
                mismatches.append({
                    'orig_key': orig_key,
                    'reimpl_key': expected_reimpl_key,
                    'orig_mean': orig_val.mean().item(),
                    'orig_std': orig_val.std().item(),
                    'reimpl_mean': reimpl_val.mean().item(),
                    'reimpl_std': reimpl_val.std().item(),
                    'max_diff': max_diff,
                })

    total = perfect_matches + len(mismatches)
    print(f"\nWeight Value Comparison:")
    print(f"  ✓ Perfect matches: {perfect_matches}/{total}")
    print(f"  ✗ Mismatches:      {len(mismatches)}/{total}")

    if mismatches:
        print(f"\n❌ Weight values don't match! First 5 issues:")
        for i, m in enumerate(mismatches[:5], 1):
            print(f"\n{i}. {m['orig_key']}")
            if 'issue' in m:
                print(f"   {m['issue']}")
            else:
                print(f"   Original:  mean={m['orig_mean']:+.6f}, std={m['orig_std']:.6f}")
                print(f"   Reimpl:    mean={m['reimpl_mean']:+.6f}, std={m['reimpl_std']:.6f}")
                print(f"   Max diff:  {m['max_diff']:.2e}")

        print("\n🔍 Diagnosis: Weights not loaded correctly!")
        print("   Check:")
        print("   1. Is init_with_pretrained=True?")
        print("   2. Does _init_parameters() call load_pretrained_state_dict()?")
        print("   3. Does translation mapping handle parametrizations?")
        print("   4. Are there extra keys preventing strict loading?")
        return False
    else:
        print(f"✅ All {total} parameter values match - weights loaded correctly!")
        return True
```

**Why this matters**:

In CordsNet integration, we found that despite:

- Translation mapping being correct ✓
- Weight structure matching ✓
- No errors during loading ✓

ALL 53 parameters had wrong values! The weights weren't actually being loaded. This was because:

1. Reimpl had extra parameters (tau_layer*, addskip_*.source.*)
2. These extra keys were handled by `_add_missing_parameters_to_state_dict()`
3. But we needed to verify the CORE weights actually transferred

**When to use this check**:

- After implementing pretrained weight loading
- When outputs don't match despite weights "loading successfully"
- When activations are much smaller than expected
- When predictions are random despite using pretrained model

#### Level 2: Compare Layer Activations

If weights match but outputs differ, compare intermediate activations:

```python
def compare_layer_activations(original, reimpl, input_tensor):
    """Compare layer-by-layer activations with careful hook placement."""

    # Store activations
    orig_acts = {}
    reimpl_acts = {}

    # Register hooks (IMPORTANT: hooks capture pre-norm/nonlin states!)
    def make_hook(name, storage):
        def hook(module, input, output):
            storage[name] = output.detach().cpu()
        return hook

    # For original model
    original.V1.register_forward_hook(make_hook('V1', orig_acts))
    original.V2.register_forward_hook(make_hook('V2', orig_acts))

    # For DynVision (hook on RConv2d, which is pre-norm!)
    reimpl.V1.register_forward_hook(make_hook('V1', reimpl_acts))
    reimpl.V2.register_forward_hook(make_hook('V2', reimpl_acts))

    # Forward pass
    with torch.no_grad():
        _ = original(input_tensor)
        _ = reimpl(input_tensor)

    # Compare (must apply same post-processing to both!)
    for layer_name in ['V1', 'V2']:
        orig_act = orig_acts[layer_name]
        reimpl_act = reimpl_acts[layer_name]

        # CRITICAL: If hooks capture before norm/nonlin, apply them manually
        # Example for CorNet-RT:
        norm_layer = getattr(original, f'norm_{layer_name}', None)
        nonlin = getattr(original, 'nonlin', None)
        if norm_layer and nonlin:
            orig_act = nonlin(norm_layer(orig_act))
            reimpl_norm = getattr(reimpl, f'norm_{layer_name}')
            reimpl_act = reimpl.nonlin(reimpl_norm(reimpl_act))

        diff = (orig_act - reimpl_act).abs()
        print(f"{layer_name}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
```

**Hook placement pitfall**: Hooks capture outputs at the point where they're registered. If you register a hook on `model.V1` (the RConv2d), it captures the output BEFORE external norm/nonlin operations. You must apply those transformations manually for fair comparison.

#### Level 3: Timestep-by-Timestep Comparison

For recurrent models, debug one timestep at a time:

```python
def debug_timestep_by_timestep(original, reimpl, img):
    """Manually step through original model to compare with reimpl."""

    print("="*80)
    print("ORIGINAL MODEL - Timestep-by-timestep")
    print("="*80)

    # Initialize original model's manual time loop
    outputs_orig = {"inp": img}
    states_orig = {}
    blocks = ["inp", "V1", "V2", "V4", "IT"]

    # t=0 (initialization)
    print("\nTimestep 0:")
    for block in blocks[1:]:
        layer = getattr(original, block)
        this_inp = outputs_orig["inp"] if block == "V1" else None
        new_output, new_state = layer(this_inp, batch_size=img.shape[0])
        outputs_orig[block] = new_output
        states_orig[block] = new_state
        print(f"  {block} output: mean={new_output.mean():.6f}, std={new_output.std():.6f}")

    # t=1 to t=N
    for t in range(1, 5):
        print(f"\nTimestep {t}:")
        new_outputs = {"inp": img}
        for block in blocks[1:]:
            prev_block = blocks[blocks.index(block) - 1]
            prev_output = outputs_orig[prev_block]
            prev_state = states_orig[block]

            layer = getattr(original, block)
            new_output, new_state = layer(prev_output, prev_state)
            new_outputs[block] = new_output
            states_orig[block] = new_state

            print(f"  {block} output: mean={new_output.mean():.6f}, std={new_output.std():.6f}")

        outputs_orig = new_outputs

    # Compare with DynVision model's final output
    print("\n" + "="*80)
    print("DYNVISION MODEL")
    print("="*80)
    img_temporal = img.unsqueeze(1).repeat(1, 5, 1, 1, 1)
    with torch.no_grad():
        out_reimpl = reimpl(img_temporal)

    print(f"Output (last timestep): mean={out_reimpl[:, -1].mean():.6f}")
    print(f"Difference from original: {(outputs_orig['IT'] - out_reimpl[:, -1, ...]).abs().max():.6f}")
```

**What to look for**:

- Which timestep does divergence start?
- Which layer does divergence start?
- Is hidden state being set/retrieved correctly?
- Are temporal parameters (delay_recurrence, delay_feedforward) correct?

#### Level 4: Check Temporal Parameter Propagation

Use the debug method from Section 2.6:

```python
# Call debug method (from Section 2.6)
reimpl.debug_temporal_params()

# Expected vs Actual:
# - delay_recurrence should match original's recurrent delay
# - delay_feedforward should match original's feedforward delay
# - history_length should be sufficient for both
```

#### Level 5: Check State Initialization

Use the debug approach from Section 3.6:

```python
# Add logging to reset() to see what states are retrieved
# Check if original uses scalar 0 vs None
# Patch if necessary
```

**The Complete Test Suite**:

Combine all checks into a comprehensive test:

```python
def test_complete_equivalence():
    """Complete test suite for model equivalence."""

    # Setup
    original = load_pretrained_original()
    reimpl = load_pretrained_reimplemented()

    # Identical input
    torch.manual_seed(42)
    img = torch.randn(1, 3, 224, 224)

    print("Phase 1: Weight Loading")
    assert compare_weights(original, reimpl), "Weights don't match!"

    print("\nPhase 2: Layer Activations")
    compare_layer_activations(original, reimpl, img)

    print("\nPhase 3: Timestep-by-Timestep")
    debug_timestep_by_timestep(original, reimpl, img)

    print("\nPhase 4: Temporal Parameters")
    reimpl.debug_temporal_params()

    print("\nPhase 5: Final Output Comparison")
    with torch.no_grad():
        # Ensure IDENTICAL inputs
        img_temporal = img.unsqueeze(1).repeat(1, 5, 1, 1, 1)

        # Original (manual time loop)
        states = {}
        outputs = {"inp": img}
        for t in range(5):
            for block in ["V1", "V2", "V4", "IT"]:
                layer = getattr(original, block)
                prev = outputs["inp"] if (block == "V1" and t == 0) else outputs.get(prev_block, None)
                state = states.get(block, None)
                outputs[block], states[block] = layer(prev, state)
                prev_block = block
        out_orig = outputs["IT"]

        # Reimplemented
        out_reimpl = reimpl(img_temporal)[:, -1]

    # Compare predictions
    diff = (out_orig - out_reimpl).abs()
    pred_orig = out_orig.argmax(dim=1)
    pred_reimpl = out_reimpl.argmax(dim=1)

    print(f"Max output difference: {diff.max():.6f}")
    print(f"Mean output difference: {diff.mean():.6f}")
    print(f"Original prediction: {pred_orig.item()}")
    print(f"Reimpl prediction: {pred_reimpl.item()}")

    assert pred_orig == pred_reimpl, "Predictions don't match!"
    print("✓ Models produce identical predictions")
```

**Key Debugging Insights**:

1. **Test incrementally**: Don't jump to full model comparison. Verify weights → single layer → single timestep → multiple timesteps.

2. **IDENTICAL inputs**: Ensure both models receive EXACTLY the same input tensor (same seed, same preprocessing, same device).

3. **Mind the hooks**: Hooks capture at registration point. If comparing activations, ensure you're comparing apples-to-apples (same post-processing).

4. **Document operation sequences**: Write out the exact sequence for both implementations. Count operations. Verify they match.

5. **Check parameter scope**: Verify which parameters go where (e.g., dt and t_recurrence to RConv2d).

6. **State semantics matter**: Check if original uses scalar 0 vs None vs torch.zeros for uninitialized states.

---

## Phase 5: Special Considerations

### 5.1 Data Preprocessing Requirements

If the model was trained with specific preprocessing, document and configure it by overriding normalization:
```bash
snakemake <output_file_path> --config normalize=null
```

See the normalization override system implemented in:

- `dynvision/params/data_params.py` (validator handles JSON null)
- `dynvision/workflow/snake_runtime.smk` (passes normalize parameter)

### 5.2 Handling Different Input Sizes

DynVision models should be flexible about input size:

```python
def _define_architecture(self):
    # Use self.dim_y, self.dim_x from parent class
    # These are calculated from input_dims automatically

    self.V1 = RConv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=7,
        stride=4,
        dim_y=self.dim_y,  # ← Passed from input_dims
        dim_x=self.dim_x,  # ← Passed from input_dims
    )

    # Next layer dimensions calculated from previous
    self.V2 = RConv2d(
        in_channels=64,
        out_channels=128,
        dim_y=self.V1.dim_y // self.V1.stride,  # ← Propagate dimensions
        dim_x=self.V1.dim_x // self.V1.stride,
    )
```

If using adaptive pooling, final dimensions are flexible:
```python
self.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # ← Adapts to any spatial size
    nn.Flatten(),
    nn.Linear(512, self.n_classes),
)
```

### 5.3 Handling Multiple Timesteps

DynVision handles time automatically via `TemporalBase.forward()`:

```python
# User code:
x = torch.randn(batch, timesteps, channels, height, width)
y = model(x)  # Shape: (batch, timesteps, n_classes)

# Internally (you don't implement this):
# for t in range(timesteps):
#     x_t = x[:, t]
#     y_t, responses = model._forward(x_t, t=t)
#     # Automatically manages hidden states via layer.set_hidden_state()
```

Your model just needs:

- `reset()`: Clear states at start
- RConv2d layers: Automatically manage hidden states
- Proper `history_length` parameter: Based on max(t_feedforward, t_recurrence, ...)

### 5.4 Skip and Feedback Connections

For advanced models with long-range connections, use DyRCNN:

```python
from dynvision.models.dyrcnn import DyRCNN
from dynvision.model_components import Skip, Feedback

class MyModel(DyRCNN):
    def _define_architecture(self):
        self.layer_names = ["V1", "V2", "V4", "IT"]

        # ... define layers ...

        # Skip connection: V1 → V4 (bypass V2)
        if self.skip:
            self.addskip_V4 = Skip(
                source=self.V1,
                auto_adapt=True,  # Automatically handles shape mismatch
                delay_index=self.t_skip // self.dt,
            )

        # Feedback connection: V4 → V1
        if self.feedback:
            self.addfeedback_V1 = Feedback(
                source=self.V4,
                auto_adapt=True,
                delay_index=self.t_feedback // self.dt,
            )
```

The `auto_adapt=True` automatically learns the transformation to match shapes via a simple linear projection.

### 5.5 Common Pitfalls and Solutions

#### Pitfall 1: Wrong recurrence_target

**Symptom**: Model loads weights but outputs don't match original.

**Solution**: Carefully trace where recurrence is applied in original:

- Before first operation? → `recurrence_target="input"`
- Between two-stage convolutions? → `recurrence_target="middle"`
- After all operations? → `recurrence_target="output"`

#### Pitfall 2: Missing operations after RConv2d

**Symptom**: Pretrained weights load but accuracy is poor.

**Solution**: Check if operations like normalization/activation are INSIDE or OUTSIDE RConv2d:

```python
# Original has TWO normalizations:
# 1. After first conv (inside mid_modules)
# 2. After second conv (outside, via layer_operations)

# DynVision needs both:
layer = RConv2d(
    mid_modules=nn.GroupNorm(32, 64),  # First norm
    ...
)
self.norm_LayerName = nn.GroupNorm(32, 64)  # Second norm
```

#### Pitfall 3: Dimension mismatches

**Symptom**: Runtime error about tensor shape mismatch.

**Solution**:

1. Print shapes at each step during initialization
2. Ensure `dim_y`, `dim_x` are passed correctly
3. Account for all striding and pooling when calculating next layer dims

```python
# Debug dimension propagation
print(f"Input: {self.dim_y} x {self.dim_x}")
print(f"After V1 (stride={self.V1.stride}): {self.V1.dim_y} x {self.V1.dim_x}")
```

#### Pitfall 4: State not persisting across timesteps

**Symptom**: Recurrent connections don't seem to work, behavior same as feedforward.

**Solution**:

1. Ensure `reset()` is called (happens automatically in `TemporalBase.forward()`)
2. Check `history_length` is sufficient: `max(t_recurrence, t_feedforward) / dt + 1`
3. Verify `delay_recurrence` matches original model's delay
4. Ensure "delay" operation is in `layer_operations`

---

## Phase 5.5: Lessons Learned - Common Mistakes in Practice

> **Note**: These are real examples from CorNet/CordsNet integration where incorrect assumptions were corrected through careful investigation.

### Mistake 1: Claiming Architecture is Wrong Without Full Tracing

**What Happened**: Initial analysis claimed the CorNet reimplementation was architecturally incorrect because operations appeared to be in the wrong order.

**The User's Challenge**: "Reevaluate... I believe the current implementation accurately replicates the original implementation. Pinpoint the difference if I'm mistaken."

**What Was Actually True**: After careful line-by-line tracing of BOTH implementations, they were equivalent. Operations could be split between inside RConv2d (via `mid_modules`) and outside (via `layer_operations`), and both paths led to the same operation order.

**Lesson Learned**:
> **Never claim something is wrong before tracing it completely.** When comparing implementations, trace both end-to-end and document the exact operation sequence at each step. Don't rely on intuition about what "should" happen.

**How to Avoid**:
```python
# WRONG: "This looks incorrect because recurrence should be here"
# RIGHT: Trace both implementations step-by-step:

# Original (line by line):
# 1. x = conv_input(x)        # Conv 3→64, k=7, s=4
# 2. x = norm_input(x)        # GroupNorm(32, 64)
# 3. x = nonlin_input(x)      # ReLU
# 4. x = x + state            # Add recurrent state ← HERE
# 5. x = conv1(x)             # Conv 64→64, k=3, s=1
# 6. x = norm1(x)             # GroupNorm(32, 64)
# 7. x = nonlin1(x)           # ReLU
# 8. state = x                # Store for next timestep

# DynVision (RConv2d with recurrence_target="middle"):
# Inside RConv2d:
# 1. x = self.conv(x)         # Conv 3→64, k=7, s=4
# 2. x = self.mid_modules(x)  # GroupNorm(32, 64)
# 3. x = self.nonlin(x)       # ReLU (internal to RConv2d)
# 4. x = x + recurrence(h)    # Add recurrent state ← HERE (same point!)
# 5. x = self.conv2(x)        # Conv 64→64, k=3, s=1
# Via layer_operations:
# 6. x = norm_V1(x)           # GroupNorm(32, 64)
# 7. x = nonlin(x)            # ReLU
# 8. layer.set_hidden_state(x) # Store via "delay" operation

# Conclusion: Sequences are IDENTICAL
```

### Mistake 2: Building New Infrastructure Instead of Using Existing Systems

**What Happened**: When implementing normalization override for CorNet (which was trained without normalization), proposed solution involved:

- New utility module in transforms.py
- Helper functions for model requirements
- Updates to 5+ files
- Model attribute system
- ~200 lines of new code

**The User's Guidance**: "I just guided you to a much simpler and straightforward solution, that had minimal implementation effort and used the existing structures and functionalities."

**What the Simple Solution Was**:

- Update Pydantic validator to handle JSON null: 3 lines
- Update Snakemake rule to check for normalize config override: 4 lines
- Total: ~10 lines of actual changes
- Usage: `snakemake test_model --config normalize=null`

**Lesson Learned**:
> **Always check if existing systems can handle the requirement before building new infrastructure.** Follow the solution hierarchy: Configuration → Parameter → Extension → New Code. Most requirements can be solved at earlier levels.

**The Solution Hierarchy (Always Apply in Order)**:

1. **Configuration only**: Can changing config values solve this?
   ```bash
   # Example: Override normalization
   snakemake --config normalize=null
   ```

2. **Parameter modification**: Can existing parameters accept new values?
   ```python
   # Example: Make validator handle None
   if v is None:
       return None
   ```

3. **Extend existing code**: Can current functions/classes be enhanced?
   ```python
   # Example: Check for config override in existing rule
   params:
       normalize = lambda w: config.get('normalize', None) if 'normalize' in config else default
   ```

4. **New focused utility**: Is new, isolated functionality needed?
   ```python
   # Only if steps 1-3 can't solve it
   ```

5. **New abstraction**: Is a fundamentally new concept required?
   ```python
   # Rarely needed - usually indicates you haven't searched enough
   ```

**Question to Ask Yourself**: "What's the smallest change that solves this specific problem?"

### Mistake 3: Misunderstanding Skip Connection Flow

**What Happened**: Initially thought each layer received multiple skip inputs that needed to be explicitly combined (e.g., `addskip1_layer2`, `addskip2_layer2`).

**The User's Correction**: Pointed out to review how activity flows through layers and that each layer only needs ONE `addskip` operation.

**What Was Actually True**:

- Previous layer's output is already in `x` from the `delay` operation
- Each layer adds ONE skip from a non-adjacent layer
- The skip is added TO the current `x`, not combined with multiple sources

**Lesson Learned**:
> **Trace the data flow through `layer_operations` to understand what's already in `x` at each step.** Don't assume you need to explicitly fetch multiple sources.

**Correct Flow**:
```python
# After layer1 completes its operations:
# ... layer1 operations ...
x = relu(x)                              # nonlin
layer1.set_hidden_state(x)               # delay: store
x = layer1.get_hidden_state(delay_ff)    # delay: retrieve

# Now x = layer1 output (already there!)

# Layer2 begins:
x = addskip_layer2(x)                    # x += learned_skip(inp)
                                         # Now x = layer1_out + skip(inp)
x = layer2(x)                            # RConv2d on combined input
x = addbias_layer2(x)                    # Add spatial bias
x = relu(x)                              # Nonlin
```

**Key Insight**: Each operation in `layer_operations` modifies `x` in sequence. By the time you reach `"addskip"`, `x` already contains the previous layer's output from the `"delay"` operation.

### Mistake 4: Not Checking What Framework Components Already Provide

**What Happened**: Proposed creating `_make_stateful_sequential()` to wrap input processing layers with state management.

**The User's Question**: "Why do you need the _make_stateful_sequential function? What required functionality is not covered by the hidden state management in recurrence.py?"

**What Was Actually True**:

- RConv2d already has `get_hidden_state()`, `set_hidden_state()`, `reset()`
- Can use `RConv2d` with `recurrence_type="none"` for layer0
- No custom wrapper needed

**Lesson Learned**:
> **Before implementing helper functions, check if the framework's core components already provide that functionality.** Read the source code of base classes and components before building utilities.

**Correct Approach**:
```python
# WRONG: Build custom wrapper
class StatefulWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self._hidden_states = []
    def get_hidden_state(self, delay=0): ...
    def set_hidden_state(self, x): ...
    def reset(self, input_shape: Optional[Tuple[int, ...]] = None) : ...

self.layer0 = StatefulWrapper(nn.Conv2d(...))

# RIGHT: Use RConv2d with recurrence_type="none"
self.layer0 = RConv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=7,
    stride=2,
    recurrence_type="none",  # ← No recurrence, but still has state management!
    history_length=max(self.t_feedforward, self.t_skip),
)
```

**How to Check**: Before implementing, search codebase:
```bash
# Search for existing methods
grep -r "get_hidden_state" dynvision/model_components/
grep -r "set_hidden_state" dynvision/model_components/
grep -r "class.*Base" dynvision/base/
```

### Mistake 5: Not Understanding Recurrence Configuration Deeply

**What Happened**: Initially didn't fully understand the distinction between feedforward and recurrent convolutions in CordsNet, and how RConv2d parameters map to them.

**The User's Guidance**: "Review how the recurrent connection (an extra convolution) is configured in recurrence.py, then review the recurrent connections 'area_area' in the original cordsnet."

**What Was Actually True**:
```python
# Original CordsNet:
# area_conv: Recurrent connection (channels[i+1] → channels[i+1], k=3, s=1)
# area_area: Feedforward connection (channels[i] → channels[i+1], k=3, s=stride)

# RConv2d mapping:
RConv2d(
    # Feedforward conv (area_area)
    in_channels=channels[i],      # From previous layer
    out_channels=channels[i+1],   # To this layer
    kernel_size=3,
    stride=strides[i],            # Can downsample

    # Recurrence conv (area_conv)
    recurrence_type="full",       # 3x3 convolution
    recurrence_kernel_size=3,     # Same as feedforward
    recurrence_target="output",   # On layer output
    # Recurrence is always: channels[i+1]→channels[i+1], stride=1
)
```

**Lesson Learned**:
> **Understand the exact mapping between original parameters and DynVision parameters.** Read both the original code AND the framework component source to see how parameters translate.

**Verification Checklist**:

- [ ] Traced original: What are the input/output channels for feedforward?
- [ ] Traced original: What are the input/output channels for recurrence?
- [ ] Traced original: What are the kernel sizes?
- [ ] Traced original: What are the strides?
- [ ] Checked RConv2d: How do I specify each of these?
- [ ] Verified: Do the parameter values match exactly?

### Mistake 6: Not Understanding Parameter Scope and Responsibility Separation

**What Happened**: During CorNet-RT reimplementation, initially passed NO temporal parameters to RConv2d, then ALL temporal parameters (dt, t_feedforward, t_recurrence), causing confusion about what belonged where.

**The User's Correction**: "Actually, I corrected you that t_feedforward does not need to be passed to the RConv2D module because delays between layers are handled in the delay step. However, the delays can influence the history_length argument."

**What Was Actually True**:

DynVision separates temporal concerns:

- **RConv2d responsibility**: Within-layer recurrence delay
- **Delay operation responsibility**: Between-layer feedforward delay

```python
# WRONG Approach 1: Pass nothing
self.V1 = RConv2d(
    in_channels=3,
    out_channels=64,
    # ✗ Missing dt, t_recurrence → delay_recurrence = 0 (incorrect!)
)

# WRONG Approach 2: Pass everything
self.V1 = RConv2d(
    in_channels=3,
    out_channels=64,
    dt=self.dt,
    t_feedforward=self.t_feedforward,  # ✗ RConv2d doesn't use this!
    t_recurrence=self.t_recurrence,
)

# CORRECT Approach: Pass only what each component needs
# In __init__:
self.history_length = int(max(self.t_recurrence, self.t_feedforward) / self.dt) + 1

# Pass to RConv2d:
self.V1 = RConv2d(
    in_channels=3,
    out_channels=64,
    dt=self.dt,                       # ✓ For calculating delay_recurrence
    t_recurrence=self.t_recurrence,   # ✓ Sets delay_recurrence internally
    history_length=self.history_length,  # ✓ Large enough for both delays
    # t_feedforward NOT passed - handled by delay operation!
)
```

**Why This Matters**:

1. **Passing nothing**: `delay_recurrence` defaults to 0, meaning recurrence is instantaneous instead of delayed
2. **Passing t_feedforward unnecessarily**: Creates confusion about what RConv2d is responsible for
3. **Missing history_length consideration**: Buffer too small if you only consider recurrence delay

**The Correct Mental Model**:

```
Time t=0    Time t=1    Time t=2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Layer 1:
  [process] ──→ [recurrence]  (RConv2d handles this)
                ↓
              [store in buffer]
                ↓
              [delay operation retrieves with delay_feedforward]
                ↓
Layer 2:
  [process] ──→ [recurrence]  (RConv2d handles this)
```

**Lesson Learned**:
> **Understand the separation of concerns between components.** RConv2d handles recurrence WITHIN a layer. The delay operation (in `layer_operations`) handles delays BETWEEN layers. Don't pass parameters to components that aren't responsible for that aspect of temporal dynamics.

**How to Verify**:

After initialization, check computed values match expectations:

```python
# Add this debug method to your model (from Section 2.6)
def debug_temporal_params(self):
    print(f"Model level:")
    print(f"  dt: {self.dt}")
    print(f"  t_feedforward: {self.t_feedforward}")
    print(f"  t_recurrence: {self.t_recurrence}")
    print(f"  delay_feedforward: {self.delay_feedforward}")  # Model level
    print(f"  history_length: {self.history_length}")
    print()

    for layer_name in self.layer_names:
        layer = getattr(self, layer_name)
        print(f"{layer_name}:")
        print(f"  dt: {layer.dt}")
        print(f"  t_recurrence: {layer.t_recurrence}")
        print(f"  delay_recurrence: {layer.delay_recurrence}")  # RConv2d level
        print(f"  history_length: {layer.history_length}")
        print(f"  Expected delay_recurrence: {int(self.t_recurrence / self.dt)}")
        assert layer.delay_recurrence == int(self.t_recurrence / self.dt), \
            f"delay_recurrence mismatch! Got {layer.delay_recurrence}, expected {int(self.t_recurrence / self.dt)}"
```

**When You'd Notice This Bug**:

- Recurrent model behaves like feedforward (delay_recurrence=0)
- Outputs don't match original despite correct weights
- First timestep works, but subsequent timesteps diverge (wrong delays)
- Buffer overflow errors (history_length too small)

---

### Mistake 7: Bias Module None-Handling and Combined Layer Timing

**What Happened**: During final CordsNet debugging, two subtle bugs remained:

1. **Bias modules returning None**: When input was None (during idle timesteps before layer activation), bias modules propagated None instead of returning bias values
2. **Combined layer timing**: When combining layer7+layer8 outputs for the classifier, used wrong delay indices

**The User's Discovery**: "I fixed the final bugs. One was that the bias modules returned none when input was none, but in the original the layer output is just the biases if there is yet no input. The second was that although the last two layers were combined before the classifier, they also needed the right delay indices, layer8 needed a t_feedforward=0 and the fetching of layer7 output an index of 1, because layer7's output was already written into its hidden state after layer8 is processed."

**What Was Actually True**:

1. **Original CordsNet bias behavior**:
```python
# Original code:
def forward(self, inp=None):
    if inp is None:
        # Return bias even without input!
        return self.bias.view(1, -1, *self.bias.shape[1:])
    return inp + self.bias
```

2. **Combined layer timing**: When multiple layers are processed in the same forward pass and their outputs combined:
      - Layer processed LATER (layer8): Needs `t_feedforward=0` (immediate)
      - Layer processed EARLIER (layer7): Needs `delay_index=1` (from hidden state)

**Why This Matters**:

Without correct None handling, idle timesteps produce None cascades instead of bias-only activations. This breaks the temporal buildup before actual input arrives.

Without correct delay indices, combined outputs come from wrong timesteps, causing subtle activation mismatches.

**Lesson Learned**:
> **Check edge cases in temporal flow**: None-handling during idle timesteps, and relative timing when combining outputs from layers processed in the same iteration.

**How to Check**:

1. **Bias None-handling**: Test model with idle_timesteps > 0 and verify early layers output bias values, not None

```python
# Test None propagation
model.reset()
x = None
for layer_name in model.layer_names:
    layer = getattr(model, layer_name)
    x = layer(x)
    if hasattr(model, f"addbias_{layer_name}"):
        bias_module = getattr(model, f"addbias_{layer_name}")
        x = bias_module(x)
        assert x is not None, f"Bias module returned None for layer {layer_name}"
```

2. **Combined layer timing**: When combining multiple layer outputs, trace WHEN each writes to hidden state:

```python
# Combined output example (layer7 + layer8 → classifier)
# Forward pass order: ... → layer7 → layer8 → classifier

# After layer7 completes:
layer7.set_hidden_state(layer7_out)  # ← Written to hidden state at end of layer7

# Layer8 begins and completes:
layer8_out = layer8(...)  # ← Still in current variable, not yet stored

# Classifier combines:
def combine_for_classifier(self):
    # layer8: Just computed, use immediate (delay=0)
    layer8_out = self.layer8.get_hidden_state(delay=0)  # or use direct variable

    # layer7: Already written to hidden state before layer8 started
    # Need delay=1 to retrieve it
    layer7_out = self.layer7.get_hidden_state(delay=1)

    return layer8_out + layer7_out
```

**Symptoms of this bug**:

- Activations much smaller than expected (None → 0 in some operations)
- Top-5 predictions close but not identical (4/5 match)
- Small numerical differences in final output logits
- Issues appear only with idle_timesteps > 0 or multi-layer combinations

### Meta-Lesson: The Investigation-First Mindset

All these mistakes share a common root cause: **Acting before fully investigating**.

**The Wrong Approach**:

1. See a requirement → immediately design a solution
2. Encounter unfamiliar code → guess at its behavior
3. Find complexity → build custom infrastructure
4. See a problem → assume existing code is wrong

**The Right Approach** (From AI Style Guide):

1. **Investigate thoroughly**: Trace existing code, search for patterns, read docs
2. **Understand constraints**: What exists? What can be reused? What must not change?
3. **Apply solution hierarchy**: Config → Parameter → Extension → New code
4. **Validate assumptions**: Never claim something is wrong without proof

**Before implementing anything, ask**:

- What does the existing code actually do? (trace it completely)
- Does similar functionality already exist? (search the codebase)
- Can existing systems handle this? (check configuration, parameters, extensions)
- What's the simplest solution that works? (not the most general or elegant)

---

## Phase 6: Documentation and Testing

### 6.1 Write Docstrings

```python
class MyModel(BaseModel):
    """
    MyModel: [One-line description]

    [Detailed description of what this model implements, its purpose,
    and key features]

    This implementation adapts [Original Model] from [Citation] to
    DynVision's temporal framework, enabling:
    - Recurrent processing with configurable delays
    - Temporal dynamics with learnable time constants
    - Integration with DynVision's training and evaluation pipeline

    Architecture:
        - Layer 1: [Description]
        - Layer 2: [Description]
        - ...

    Args:
        n_timesteps: Number of timesteps to process
        input_dims: Input dimensions (timesteps, channels, height, width)
        n_classes: Number of output classes
        dt: Integration time step (ms)
        t_recurrence: Recurrent connection delay (ms)
        recurrence_type: Type of recurrent connections ("self", "full", "none")
        init_with_pretrained: Whether to load pretrained weights

    References:
        - Author et al. (Year) "Paper Title" Journal/Conference
        - Original implementation: [URL if available]

    Example:
        >>> model = MyModel(
        ...     input_dims=(5, 3, 224, 224),
        ...     n_classes=1000,
        ...     init_with_pretrained=True,
        ... )
        >>> model.setup("fit")
        >>> x = torch.randn(2, 5, 3, 224, 224)
        >>> y = model(x)  # Shape: (2, 5, 1000)
    """
```

### 6.2 Create Test Function

```python
def test_mymodel(
    input_shape: Tuple[int, ...] = (5, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> None:
    """Test MyModel implementation.

    Tests include:
    - Model creation and setup
    - Forward pass with random input
    - Stability checks with extreme values
    - Parameter verification

    Args:
        input_shape: Input tensor shape
        device: Device to run test on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing MyModel on {device}")

    # Create and setup model
    model = MyModel(input_dims=input_shape)
    model.setup("fit")
    print("✓ Model creation successful")

    # Test forward pass
    x = torch.randn(1, *input_shape, device=device)
    y = model(x)
    print(f"✓ Forward pass successful: {x.shape} -> {y.shape}")

    # Test with pretrained weights
    if model.init_with_pretrained:
        try:
            model = MyModel(input_dims=input_shape, init_with_pretrained=True)
            model.setup("fit")
            print("✓ Pretrained weights loaded successfully")
        except Exception as e:
            print(f"⚠ Pretrained loading failed: {e}")

    # Log trainable parameters
    trainable_params = [
        f"{name} [{tuple(param.shape)}]"
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    print("Trainable Parameters:\n\t%s" % "\n\t".join(trainable_params))

    print("All tests passed!")


if __name__ == "__main__":
    test_mymodel()
```

### 6.3 Add to Model Zoo

1. **Import in `__init__.py`**:
```python
# dynvision/models/__init__.py
from .mymodel import MyModel

__all__ = [..., "MyModel"]
```

2. **Update documentation**:
```markdown
# docs/reference/models.md

## MyModel

[Description]

**Usage:**
\`\`\`python
from dynvision.models import MyModel

model = MyModel(
    input_dims=(5, 3, 224, 224),
    n_classes=1000,
    init_with_pretrained=True,
)
\`\`\`

**Parameters:**

- ...

**References:**

- ...
```

---

## Quick Reference: DynVision Architecture Patterns

### Pattern 1: Simple Feedforward with Optional Recurrence

```python
class SimpleModel(BaseModel):
    def _define_architecture(self):
        self.layer_names = ["conv1", "conv2"]
        self.layer_operations = ["layer", "nonlin", "pool", "record", "delay"]

        self.conv1 = RConv2d(3, 64, 3, stride=1, recurrence_type="self")
        self.conv2 = RConv2d(64, 128, 3, stride=2, recurrence_type="self")
        self.nonlin = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Linear(128, 10)
```

### Pattern 2: Two-Stage Blocks (CorNet-style)

```python
class TwoStageModel(BaseModel):
    def _define_architecture(self):
        self.layer_names = ["block1"]
        self.layer_operations = ["layer", "norm", "nonlin", "record", "delay"]

        self.block1 = RConv2d(
            in_channels=3,
            mid_channels=64,
            out_channels=64,
            kernel_size=(7, 3),
            stride=(4, 1),
            mid_modules=nn.GroupNorm(32, 64),
            recurrence_target="middle",
        )
        self.norm_block1 = nn.GroupNorm(32, 64)
        self.nonlin = nn.ReLU()
```

### Pattern 3: Biological Features (DyRCNN-style)

```python
class BiologicalModel(DyRCNN):
    def _define_architecture(self):
        self.layer_names = ["V1", "V4"]
        self.layer_operations = [
            "layer", "addskip", "addfeedback", "tstep",
            "nonlin", "record", "delay"
        ]

        self.V1 = RConv2d(3, 64, 3, recurrence_type="full", **params)
        self.tstep_V1 = EulerStep(dt=self.dt, tau=self.tau)

        self.V4 = RConv2d(64, 128, 3, **params)
        self.tstep_V4 = EulerStep(dt=self.dt, tau=self.tau)

        if self.skip:
            self.addskip_V4 = Skip(source=self.V1, auto_adapt=True)
        if self.feedback:
            self.addfeedback_V1 = Feedback(source=self.V4, auto_adapt=True)
```

---

## Summary Checklist

Before considering integration complete:

**Investigation Phase**:

- [ ] Traced original forward pass line by line
- [ ] Documented all layer parameters (channels, kernels, strides, bias)
- [ ] Identified where recurrence is applied
- [ ] Checked data preprocessing requirements
- [ ] Inspected pretrained weights structure (if applicable)

**Mapping Phase**:

- [ ] Chose appropriate base class (BaseModel vs DyRCNN)
- [ ] Mapped convolutional blocks to RConv2d parameters
- [ ] Determined correct `recurrence_target` value
- [ ] Identified operations inside vs. outside RConv2d
- [ ] Planned `layer_operations` sequence

**Implementation Phase**:

- [ ] Defined `layer_names` matching attribute names
- [ ] Implemented `_define_architecture()` with all layers
- [ ] Implemented `reset()` for hidden state management
- [ ] Added pretrained weight loading (if applicable)
- [ ] Created layer name translation mapping (if applicable)

**Validation Phase**:

- [ ] Tested model creation and initialization
- [ ] Tested forward pass with various inputs
- [ ] Compared outputs with original implementation
- [ ] Tested with different configurations
- [ ] Verified pretrained weights load correctly

**Lessons Learned Check** (See Phase 5.5):

- [ ] Did I trace the original code completely before claiming anything was wrong?
- [ ] Did I check existing systems before building new infrastructure?
- [ ] Did I understand the data flow through `layer_operations`?
- [ ] Did I check what framework components already provide?
- [ ] Did I verify parameter mappings between original and DynVision?

**Documentation Phase**:

- [ ] Written comprehensive docstrings
- [ ] Created test function
- [ ] Added to model zoo (`__init__.py`)
- [ ] Updated documentation
- [ ] Noted any special preprocessing requirements

---

## Further Reading

- **DynVision Architecture**: [`docs/development/guides/claude-guide.md`](claude-guide.md)
- **AI Development Guide**: [`docs/development/guides/ai-style-guide.md`](ai-style-guide.md)
- **Base Classes**: `dynvision/base/temporal.py` ([view on GitHub](https://github.com/Lindsay-Lab/DynVision/blob/main/dynvision/base/temporal.py))
- **Recurrence Components**: `dynvision/model_components/recurrence.py` ([view on GitHub](https://github.com/Lindsay-Lab/DynVision/blob/main/dynvision/model_components/recurrence.py))
- **Reference Implementations**:
  - Simple: `dynvision/models/cornet_rt.py` ([view on GitHub](https://github.com/Lindsay-Lab/DynVision/blob/main/dynvision/models/cornet_rt.py))
  - Complex: `dynvision/models/dyrcnn.py` ([view on GitHub](https://github.com/Lindsay-Lab/DynVision/blob/main/dynvision/models/dyrcnn.py))

---

*"Before building new, understand what exists. Before coding, trace the original. Before claiming equivalence, validate with data."*
