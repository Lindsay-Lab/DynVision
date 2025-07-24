# Creating Custom Models

This guide walks you through the process of creating custom neural network models in DynVision. You'll learn how to implement your own architectures while leveraging DynVision's biological vision components.

[Placeholder for diagram showing the relationship between model components]

> **Note:** This guide focuses on practical implementation. For technical details, see:
> - [Models Reference](../reference/models.md) for model architectures
> - [Model Components Reference](../reference/model-components.md) for base classes
> - [Organization Reference](../reference/organization.md) for system structure

## Prerequisites

Before creating a custom model, ensure you:
- Understand basic PyTorch and PyTorch Lightning concepts
- Are familiar with DynVision's [design philosophy](../explanation/design-philosophy.md)
- Have reviewed the [example models](../reference/models.md)
- Are familiar with the available [model components](../reference/model-components.md)
- Understand the [configuration system](../reference/configuration.md)

## Quick Start

Let's create a simple model to understand the basics:

```python
from dynvision.model_components import LightningBase
import torch.nn as nn

class SimpleModel(LightningBase):
    def __init__(
        self,
        input_dims=(20, 3, 32, 32),  # (Time, Channel, Height, Width)
        **kwargs
    ):
        super().__init__(input_dims=input_dims, **kwargs)
        self._define_architecture()
    
    def _define_architecture(self):
        """Define the model architecture."""
        # Define layers
        self.conv1 = nn.Conv2d(self.n_channels, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.classifier = nn.Linear(32, self.n_classes)
    
    def forward(self, x):
        """Forward pass."""
        # Base class handles input dimensions automatically
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return self.classifier(x.mean(dim=[2, 3]))
```

This simple model demonstrates:
- Inheriting from `LightningBase`
- Handling input dimensions
- Basic layer definition
- Forward pass implementation

For details on base class features, see [Model Components Reference](../reference/model-components.md).

## Adding Features

After creating a basic model, you can add more advanced features:

### 1. Response Tracking

Monitor model activations during training:

```python
class AnalyzableModel(LightningBase):
    def __init__(self, **kwargs):
        super().__init__(store_responses=True, **kwargs)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch[0])
        
        # Access responses
        responses = self.get_responses()
        
        # Convert to DataFrame for analysis
        if batch_idx % 100 == 0:
            df = self.get_dataframe()
            self.log_table("responses", dataframe=df)
```

See [Visualization Guide](../user-guide/visualization.md) for analysis tools.

### 2. State Management

For recurrent models, use `RecurrentConnectedConv2d` to handle states:

```python
from dynvision.model_components import RecurrentConnectedConv2d

class RecurrentModel(LightningBase):
    def _define_architecture(self):
        # Recurrent layer handles state automatically
        self.conv1 = RecurrentConnectedConv2d(
            in_channels=self.n_channels,
            out_channels=32,
            kernel_size=3,
            recurrence_type="full"
        )
    
    def reset(self):
        """Reset between sequences."""
        for layer in self.modules():
            if hasattr(layer, 'reset'):
                layer.reset()
```

For state management details, see [Model Components Reference](../reference/model-components.md#state-management).

### 3. Biological Features

Add biological components as needed:

```python
from dynvision.model_components import EulerStep, RecurrentConnectedConv2d

class BiologicalModel(LightningBase):
    def _define_architecture(self):
        # Add dynamics
        self.tstep = EulerStep(dt=self.dt, tau=self.tau)
        
        # Add recurrent connectivity
        self.conv1 = RecurrentConnectedConv2d(
            in_channels=self.n_channels,
            out_channels=32,
            kernel_size=3,
            recurrence_type="full"
        )
```

For details on biological components, see:
- [Dynamics Solvers](../reference/dynamics-solvers.md)
- [Recurrence Types](../reference/recurrence-types.md)
- [Biological Plausibility](../explanation/biological-plausibility.md)

## Advanced Features

After mastering the basics, you can enhance your models with advanced features:

### Training Configurations

[TODO: 
    - reference here the existing flexibility to adapt the various optimizer parameters in the config file
    - note that in particular the pytorch lightning callback function can be overwritten and edited (link to their docs)
    - give an actually useful example code snippet show a customization
]


For training configuration details, see [Training Guide](../user-guide/workflows.md#training).


### Layer Operations

Define custom operation sequences:

```python
def _define_architecture(self):
    """Define layer operations sequence."""
    self.layer_operations = [
        "layer",    # Apply convolution
        "tstep",    # Apply dynamics
        "nonlin",   # Apply nonlinearity
        "record",   # Store responses
        "pool"      # Apply pooling
    ]
```

See [Layer Operations Reference](../reference/organization.md#layer-operations) for available operations.

## Parameter Initialization

Proper initialization is critical for model convergence:

```python
def _init_parameters(self):
    """Initialize model parameters."""
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            # Initialize convolutional layers
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu'
            )
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize batch normalization
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    # Custom initialization for recurrent weights
    if hasattr(self, 'recurrent_weights'):
        nn.init.orthogonal_(self.recurrent_weights)
```

## Testing Example

Comprehensive testing ensures model reliability:

```python
def test_model():
    """Test model functionality."""
    model = MyModel(input_dims=(20, 3, 32, 32))
    
    # Test input handling
    x = torch.randn(1, 3, 32, 32)  # Single timestep
    y = model(x)
    assert y.shape == (1, 20, 10)  # Should expand timesteps
    
    # Test state management
    model.reset()
    y_new = model(x)
    assert not torch.allclose(y, y_new)  # Should be different due to dynamics
    
    # Test response tracking
    _ = model(x, store_responses=True)
    responses = model.get_responses()
    assert len(responses) > 0
```

## Troubleshooting Guide

[TODO: explain how to use debug mode]

## Integration with Workflows

1. Place your model in `dynvision/models/`:
```python
# dynvision/models/my_model.py
from dynvision.model_components import LightningBase

class MyModel(LightningBase):
    """Your custom model implementation."""
    pass
```

2. Register in `__init__.py`:
```python
# dynvision/models/__init__.py
from .my_model import MyModel

__all__ = [..., 'MyModel']
```

3. Configure in YAML:
```yaml
# config_experiments.yaml
model:
  name: MyModel
  args:
    input_dims: [20, 3, 224, 224]
    dt: 2.0
    tau: 8.0
    store_responses: true
```

4. Use with Snakemake workflows:
```bash
# Run experiment with your model
snakemake --config model_name=MyModel experiment=contrast
```

For more details on workflows, see:
- [Workflow Guide](../user-guide/workflows.md)
- [Configuration Reference](../reference/configuration.md)

## Best Practices

1. **Code Organization**
   - Keep related components together
   - Use descriptive names
   - Document parameter choices

2. **Performance**
   - Profile your model
   - Use appropriate batch sizes
   - Consider hardware constraints

3. **Debugging**
   - Add logging statements
   - Monitor gradients
   - Test incrementally

## Next Steps

- Study [example models](../reference/models.md) for inspiration
- Check [evaluation guides](../user-guide/evaluation.md) for testing

## Related Resources

- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [BrainScore]
- ...