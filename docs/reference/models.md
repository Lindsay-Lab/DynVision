# Model Architectures and Components

This reference documentation provides an overview of the model architectures available in DynVision and how they utilize different components to implement biologically-inspired vision models.

<p align="center">
  <img src="../../assets/reference_models.png" alt="Reference models (CORnet-RT, CordsNet, DyRCNNx8)" width="700"/>
</p>

*Figure: Equivalent reimplementation of CORnet‑RT and CordsNet in DynVision,
alongside the native DyRCNNx8. The reimplementations are numerically equivalent
to the originals while achieving up to 52% speedup.*

## Available Model Architectures

DynVision provides several model architectures, each serving different research purposes:

### 1. DyRCNN Family

The DyRCNN (Dynamic Recurrent CNN) family implements biologically-inspired vision models with continuous-time dynamics and recurrent connections.

#### DyRCNNx4

A four-layer architecture inspired by the ventral visual stream:

```python
from dynvision.models import DyRCNNx4

model = DyRCNNx4(
    n_classes=10,                  # Number of output classes
    input_dims=(20, 3, 64, 64),   # (timesteps, channels, height, width)
    recurrence_type="full",      # Type of recurrent connectivity
    dt=2.0,                      # Integration time step (ms)
    tau=5.0                      # Neural time constant (ms) — default
)
```

**Layer Organization**:

- V1: Early visual processing with local feature extraction
- V2: Intermediate feature processing
- V4: Higher-order feature integration
- IT: Object recognition

Each layer implements:

- Feedforward convolution
- Recurrent connections
- Nonlinear activation
- Optional pooling

For details on recurrence types, see the [Recurrence Types Reference](recurrence-types.md).

### 2. Standard Architectures

DynVision includes implementations of standard architectures, enhanced with temporal dynamics:

#### ResNet with Dynamics

```python
from dynvision.models import ResNet

model = ResNet(
    n_classes=1000,
    input_dims=(20, 3, 224, 224),
    version="50",                # ResNet version (18, 34, 50, 101)
    dynamics_solver="euler",     # Type of dynamics solver
    dt=2.0,
    tau=5.0
)
```

#### AlexNet with Dynamics

```python
from dynvision.models import AlexNet

model = AlexNet(
    n_classes=1000,
    input_dims=(20, 3, 224, 224),
    dynamics_solver="euler",
    dt=2.0,
    tau=5.0
)
```

### 3. Research Models

#### CordsNet

```python
from dynvision.models import CordsNet

model = CordsNet(
    n_classes=1000,
    input_dims=(20, 3, 224, 224),
    topographic=True,           # Enable topographic organization
    dt=2.0,
    tau=5.0
)
```

## Model Comparison

DynVision enables systematic comparison of different recurrent architectures. The figure below compares training performance across reference models and DynVision variants trained on ImageNet:

<p align="center">
  <img src="../../assets/reference_models.png" alt="Reference Model Comparison" width="700"/>
</p>

*Figure: Training comparison of reference RCNN implementations (CorNet-RT, CordsNet) and DynVision's DyRCNNx8 across ImageNet training. DynVision achieves faster training while supporting more biologically detailed architectures.*

## Component Integration

DynVision models are built from several key components, each documented in detail in their respective reference files:

### 1. Neural Dynamics

All models use continuous-time dynamics solvers to evolve neural activity:

```python
from dynvision.model_components import EulerStep

# Example from DyRCNNx4
self.V1_dynamics = EulerStep(dt=self.dt, tau=self.tau_V1)
```

See [Dynamics Solvers Reference](dynamics-solvers.md) for implementation details.

### 2. Recurrent Connections

Models can use various types of recurrent connectivity:

```python
from dynvision.model_components import RecurrentConnectedConv2d

# Example from DyRCNNx4
self.V1_recurrent = RecurrentConnectedConv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    recurrence_type="full",
    dt=self.dt,
    tau=self.tau
)
```

See [Recurrence Types Reference](recurrence-types.md) for available patterns.

### 3. Layer Connections

Models can implement skip and feedback connections between layers:

```python
from dynvision.model_components import Skip, Feedback

# Example skip connection
self.V1_V4_skip = Skip(
    in_channels=64,
    out_channels=256,
    scale_factor=4
)

# Example feedback connection
self.V4_V1_feedback = Feedback(
    in_channels=256,
    out_channels=64,
    scale_factor=0.25
)
```

### 4. Input Processing

Models typically start with a retina/LGN processing stage:

```python
from dynvision.model_components import RetinaLGN

self.retina = RetinaLGN(
    in_channels=3,
    out_channels=64,
    kernel_size=7,
    stride=2
)
```

## Model Configuration

Models are configured through a hierarchical configuration system:

```yaml
# Example from config_defaults.yaml
model:
  name: DyRCNNx4
  args:
    dt: 2.0
    tau: 10.0
    recurrence_type: full
    store_responses: true
```

See [Configuration Reference](configuration.md) for details.

## Creating Custom Models

To create custom models, inherit from appropriate base classes:

```python
from dynvision.model_components import LightningBase

class CustomModel(LightningBase):
    def __init__(self, n_classes=1000, input_dims=(20, 3, 224, 224), **kwargs):
        super().__init__(n_classes=n_classes, input_dims=input_dims, **kwargs)
        self._define_architecture()
    
    def _define_architecture(self):
        # Define model architecture using components
        pass
```

See the [Custom Models Guide](../user-guide/custom-models.md) for detailed instructions.

## Best Practices

1. **Model Selection**:
      - Use DyRCNN models for biological vision research
      - Use enhanced standard architectures for comparison with literature
      - Use research models for specific hypotheses

2. **Component Configuration**:
      - Match time constants to biological values (typically 5-20ms)
      - Choose recurrence types based on computational budget
      - Use appropriate nonlinearities for biological plausibility

3. **Performance Optimization**:
      - Use simpler recurrence types for large-scale training
      - Consider mixed precision training
      - Adjust batch sizes based on available memory

## References

1. Heeger, D. J., & Mackey, W. E. (2019). Oscillatory recurrent gated neural integrator circuits (ORGaNICs).
2. Spoerer, C. J., McClure, P., & Kriegeskorte, N. (2017). Recurrent Convolutional Neural Networks: A Better Model of Biological Object Recognition.
3. Nayebi, A., et al. (2018). Task-Driven Convolutional Recurrent Models of the Visual System.
4. Kubilius, J., et al. (2019). CORnet: Modeling the Neural Mechanisms of Core Object Recognition.