# Recurrence Types in DynVision

This document describes the different types of recurrent connections available in DynVision and their biological relevance.

## Introduction to Recurrence

Recurrent connections are abundant in the primate visual system. In the ventral visual stream, lateral recurrent connections exist amongst neurons within a visual cortical region, and feedback connections go from higher areas (like V4) back to lower ones (such as V1).

DynVision implements several types of recurrent connections, each with different computational properties and biological interpretations.

## Available Recurrence Types

### 1. Self Recurrence

Self recurrence is the simplest form of recurrence, where a unit connects only to itself.

![Self Recurrence](../assets/self_recurrence.png)

**Implementation Details**:
- A layer's output tensor from a previous time step is multiplied with a weight and added to the current time step.
- Implemented in `SelfConnection` class.

**Biological Relevance**:
- Models the persistence of neural activity over time.
- Can be interpreted as a simplified form of neural adaptation.

**Usage Example**:
```python
from dynvision.models import DyRCNNx4

model = DyRCNNx4(
    recurrence_type="self",
    # other parameters
)
```

**Computational Efficiency**:
- Very efficient, as it only requires a scalar multiplication.
- Low parameter count, making it suitable for quick experiments.

### 2. Full Recurrence

In full recurrence, a unit gets input from all units within a nearby spatial region across all channels.

![Full Recurrence](../assets/full_recurrence.png)

**Implementation Details**:
- Implemented by applying a kernel convolution on the layer's output tensor and adding the outcome back to the same layer.
- Uses a full convolutional operation with a kernel size that determines the spatial extent of the recurrence.
- Implemented in `FullConnection` class.

**Biological Relevance**:
- Models dense local connectivity within cortical areas.
- Captures both iso-feature and cross-feature interactions.

**Usage Example**:
```python
from dynvision.models import DyRCNNx4

model = DyRCNNx4(
    recurrence_type="full",
    # other parameters
)
```

**Computational Efficiency**:
- More computationally expensive than self recurrence.
- Has O(C²K²) parameters, where C is the number of channels and K is the kernel size.

### 3. Depthwise Separable Recurrence

Depthwise separable recurrence applies a depthwise (spatial dimension) and then a pointwise (feature dimension) convolution instead of a full one.

![Depthwise Separable Recurrence](../assets/depthwise_recurrence.png)

#### 3.1 Depthpointwise Recurrence

**Implementation Details**:
- First applies a depthwise convolution (separate convolution for each channel).
- Then applies a pointwise convolution (1x1 convolution across channels).
- Implemented in `DepthPointwiseConnection` class.

**Biological Relevance**:
- Models the idea that neurons first integrate information from the same feature type across space, then integrate across features.

**Usage Example**:
```python
from dynvision.models import DyRCNNx4

model = DyRCNNx4(
    recurrence_type="depthpointwise",
    # other parameters
)
```

#### 3.2 Pointdepthwise Recurrence

**Implementation Details**:
- Inverts the order of operations: first pointwise, then depthwise.
- Implemented in `PointDepthwiseConnection` class.

**Biological Relevance**:
- Models the idea that neurons in visual cortex might first integrate across features at a single location, then spread that integration spatially.

**Usage Example**:
```python
from dynvision.models import DyRCNNx4

model = DyRCNNx4(
    recurrence_type="pointdepthwise",
    # other parameters
)
```

**Computational Efficiency**:
- More efficient than full recurrence, with O(C² + CK²) parameters instead of O(C²K²).
- A good compromise between computational efficiency and representational power.

### 4. Local Recurrence

Local recurrence captures the 2-D topology of visual cortices by arranging units on a 2-D grid inspired by cortical organization.

![Local Recurrence](../assets/local_recurrence.png)

**Implementation Details**:
- Units in a layer are systematically arranged on a 2-D grid inspired by cortical organization.
- A convolution with kernel size > 1 is applied to this grid.
- Input to each unit is a combination of cortically-local feature and space information.
- Implemented in `LocalLateralConnection` class.

**Biological Relevance**:
- Respects the topographic organization of visual cortex (e.g., orientation pinwheels in V1).
- Models how features that are close in feature space (e.g., similar orientations) interact more strongly.

**Usage Example**:
```python
from dynvision.models import DyRCNNx4

model = DyRCNNx4(
    recurrence_type="local",
    # other parameters
)
```

### 5. Local Depthwise Recurrence

An extension of local recurrence that adds patchy long-range connections for more complex interactions.

**Implementation Details**:
- Combines local recurrence with an additional depthwise convolution.
- Models patchy long-range connections between units with different feature preferences but the same receptive field.
- Implemented in `LocalSeparableConnection` class.

**Biological Relevance**:
- Models the patchy long-range lateral connections observed in visual cortex.
- These connections typically link cells with similar feature preferences across space.

**Usage Example**:
```python
from dynvision.models import DyRCNNx4

model = DyRCNNx4(
    recurrence_type="localdepthwise",
    # other parameters
)
```

**Computational Efficiency**:
- Most computationally intensive of all recurrence types.
- Provides the most biologically realistic connectivity patterns.

## Biological Plausibility Comparison

| Recurrence Type | Biological Plausibility | Computational Efficiency | Parameter Count |
|-----------------|-------------------------|--------------------------|-----------------|
| Self | Low | Very High | Very Low |
| Full | Medium | Low | High |
| Depthpointwise | Medium-High | Medium | Medium |
| Pointdepthwise | Medium-High | Medium | Medium |
| Local | High | Low | High |
| Localdepthwise | Very High | Very Low | Very High |

## Implementation in RecurrentConnectedConv2d

All recurrence types are implemented within the `RecurrentConnectedConv2d` class, which serves as the main building block for recurrent layers in DynVision models.

Key parameters when configuring recurrence include:

- `recurrence_type`: Type of recurrence to use ("self", "full", "depthpointwise", "pointdepthwise", "local", "localdepthwise")
- `dt`: Time step for simulation (in milliseconds)
- `tau`: Time constant for neural dynamics (in milliseconds)
- `t_recurrence`: Delay for recurrent connections (in milliseconds)
- `delay_recurrence`: Number of time steps to delay recurrent input
- `integration_strategy`: How recurrent input is combined with feedforward input ("additive" or "multiplicative")

## Advanced Usage

### Combining Multiple Recurrence Types

While not directly supported through a single parameter, you can create models with multiple recurrence types by manually configuring the layers:

```python
import torch
from dynvision.model_components import RecurrentConnectedConv2d

# Create a layer with self recurrence
layer1 = RecurrentConnectedConv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    recurrence_type="self"
)

# Create a layer with full recurrence
layer2 = RecurrentConnectedConv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    recurrence_type="full"
)
```

### Custom Recurrence Types

You can implement custom recurrence types by creating a new recurrence class and integrating it with `RecurrentConnectedConv2d`:

```python
# in recurrence.py

# Define a custom recurrence type
class CustomRecurrence(RecurrenceBase):
    def __init__(self, in_channels, kernel_size, bias=False):
        super().__init__()
        # Implement your custom recurrence
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=bias
        )
    
    def forward(self, x):
        return self.conv(x)
```

```python 
class RecurrentConnectedConv2d(ConvolutionalRecurrenceBase):
    $$$
    def _setup_recurrence(self) -> None:
    $$$

        # Map recurrence types to their implementations
        recurrence_types = {
            "self": lambda: SelfConnection(fixed_weight=self.fixed_self_weight),
            "custom" lambda: CustomRecurrence(in_channels=self.out_channels, **recurrence_params
            )
            # ... other recurrence types
```

## Performance Considerations

Recurrence types significantly impact both memory usage and computation time:

1. **Memory Usage**:
   - Self recurrence has minimal memory overhead
   - Full and local recurrence types require storing intermediate activations
   - Depthwise separable recurrence types offer a middle ground

2. **Computation Time**:
   - Self recurrence is fastest
   - Local and localdepthwise recurrence are slowest
   - For large models, consider using pointdepthwise or depthpointwise recurrence

3. **Scaling with Image Size**:
   - Full recurrence scales quadratically with image size
   - Depthwise separable types scale more efficiently

## References

For more details on the biological inspiration behind these recurrence types, see:

1. van Bergen & Kriegeskorte (2020). Going in circles is the way forward: The role of recurrence in visual inference.
2. Liang & Hu (2015). Recurrent convolutional neural network for object recognition.
3. Stettler et al. (2002). Lateral Connectivity and Contextual Interactions in Macaque Primary Visual Cortex.
4. Ohki et al. (2006). Highly ordered arrangement of single neurons in orientation pinwheels.
5. Spoerer, C. J., McClure, P., & Kriegeskorte, N. (2017). Recurrent Convolutional Neural Networks: A Better Model of Biological Object Recognition
6. Qian, X., et al. (2024). Local lateral connectivity is sufficient for replicating cortex-like topographical organization in deep neural networks 

