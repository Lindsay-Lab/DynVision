# Base Module Reference

The `dynvision.base` module provides the foundational classes for building biologically-inspired neural network models with PyTorch Lightning integration. This module implements a modular architecture that separates core neural network functionality, training infrastructure, storage management, and debugging capabilities.

<p align="center">
  <img src="../../assets/rcnn_architecture.png" alt="DynVision RCNN Architecture" width="700"/>
</p>

*Figure: Architecture of the DyRCNN model family. Each layer receives feedforward input with configurable delays and recurrent input from its own past activity, with optional skip and feedback connections from other layers.*

## Overview

The base module is organized around five core components that can be used independently or combined through inheritance:

- **Device/Dtype Coordination**: Ensures consistent data types and device placement across model components
- **Core Neural Network**: Implements fundamental neural computation and architecture management
- **Training Framework**: Provides PyTorch Lightning integration for model training and evaluation
- **Storage Management**: Handles response storage and output management during training and testing
- **Monitoring**: Offers comprehensive debugging, logging, and performance monitoring capabilities

## Module Components

### BaseModel

The primary class that combines all functionality for typical use cases.

```python
class BaseModel(
    TemporalBase,
    LightningBase, 
    StorageBufferMixin,
    MonitoringMixin,
    DtypeDeviceCoordinatorMixin
)
```

**Purpose**: Complete neural network framework with training, storage, monitoring, and device coordination.

**Parameters**:

- `input_dims` (Tuple[int]): Input tensor dimensions as (timesteps, channels, height, width). Default: `(20, 3, 224, 224)`
- `n_classes` (Optional[int]): Number of output classes. Can be inferred from data if not specified
- `n_timesteps` (int): Number of temporal processing steps. Default: `1`
- `dt` (float): Integration time step in milliseconds. Default: `2.0`
- `tau` (float): Neural time constant in milliseconds. Default: `5.0`
- `learning_rate` (float): Base learning rate for training. Default: `0.001`
- `store_responses` (int): Number of responses to store during training. Default: `0`
- `**kwargs`: Additional parameters passed to component classes

**Key Methods**:

- `_define_architecture()`: Abstract method to implement model architecture
- `forward(x)`: Forward pass through the network
- `training_step(batch, batch_idx)`: PyTorch Lightning training step
- `configure_optimizers()`: Optimizer and scheduler configuration

**Example**:
```python
class MyModel(BaseModel):
    def _define_architecture(self):
        self.layer_names = ['conv1', 'conv2', 'classifier']
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.classifier = nn.Linear(128, self.n_classes)

model = MyModel(
    input_dims=(20, 3, 64, 64),
    n_classes=10,
    learning_rate=0.001
)
```

### TemporalBase

Core neural network functionality for biologically-inspired models.

```python
class TemporalBase(nn.Module)
```

**Purpose**: Implements fundamental neural network computation including forward passes, temporal dynamics, and parameter management.

**Key Attributes**:

- `input_dims` (Tuple[int]): Processed input dimensions
- `n_classes` (int): Number of output classes
- `n_timesteps` (int): Number of temporal processing steps
- `dt` (float): Integration time step in milliseconds
- `tau` (float): Neural time constant in milliseconds
- `layer_names` (List[str]): Names of network layers in processing order
- `data_presentation_pattern` (Union[str, List[int]]): Pattern controlling stimulus/null presentation
- `shuffle_presentation_pattern` (bool): Whether to shuffle pattern per batch
- `loss_reaction_time` (float): Reaction time window in milliseconds for loss masking
- `n_residual_timesteps` (int): Residual timesteps for signal propagation

**Temporal Presentation Parameters**:

TemporalBase now handles sophisticated temporal presentation patterns and reaction time masking:

- **`data_presentation_pattern`**: Controls which timesteps receive stimulus (1) vs null input (0)
  - String format: `"1011"` or `"1"` (auto-expanded)
  - List format: `[1, 0, 1, 1]`
  - Pattern length must divide evenly into `n_timesteps`
  - See [Temporal Data Presentation Guide](../user-guide/temporal-data-presentation.md) for details

- **`shuffle_presentation_pattern`**: When `True`, randomly permutes pattern per batch
  - Shuffles base pattern before resampling to `n_timesteps`
  - Maintains chunk durations after shuffling
  - Provides temporal variability during training

- **`loss_reaction_time`**: Masks labels after stimulus onset (in milliseconds)
  - Converts to timesteps: `ceil(loss_reaction_time / dt)`
  - Applied per stimulus chunk (not just first timestep)
  - Warnings issued when reaction window exceeds chunk duration
  - See [Loss Functions Reference](losses.md#temporal-masking-and-presentation-patterns)

- **`non_label_index`**: Label value for masked timesteps (default: -1)
- **`non_input_value`**: Input value for null timesteps (default: 0.0)

**Loss Configuration**:

TemporalBase owns loss criterion initialization through `_init_loss()`:

- Supports multiple loss functions with configurable weights
- Automatically registers hooks for ActivityLoss
- Handles `ignore_index` configuration for temporal masking
- See [Loss Functions Reference](losses.md) for available losses

**Core Methods**:

#### `_define_architecture() -> None`
Abstract method that must be implemented by subclasses to define the network architecture.

**Raises**: `NotImplementedError` if not implemented in subclass

#### `forward(x_0: torch.Tensor, store_responses: bool = False, feedforward_only: bool = False) -> torch.Tensor`
Forward pass through the network over all timesteps.

**Parameters**:

- `x_0` (torch.Tensor): Input tensor with shape (batch, timesteps, channels, height, width)
- `store_responses` (bool): Whether to store intermediate responses. Default: `False`
- `feedforward_only` (bool): Whether to disable recurrent connections. Default: `False`

**Returns**: `torch.Tensor` with shape (batch, timesteps, n_classes)

#### `set_residual_timesteps(n_timesteps: Optional[int] = None, max_timesteps: int = 100) -> None`
Set or automatically determine the number of residual timesteps required for signal propagation.

**Parameters**:

- `n_timesteps` (Optional[int]): Explicit number of residual timesteps
- `max_timesteps` (int): Maximum timesteps to check during automatic determination

**Example**:
```python
class CoreModel(TemporalBase, DtypeDeviceCoordinator):
    def _define_architecture(self):
        self.layer_names = ['layer1', 'layer2']
        self.layer1 = nn.Conv2d(3, 64, 3)
        self.layer2 = nn.Conv2d(64, 10, 3)
        self.classifier = nn.AdaptiveAvgPool2d(1)

model = CoreModel(input_dims=(10, 3, 32, 32))
```

### LightningBase

PyTorch Lightning integration for training and evaluation.

```python
class LightningBase(pl.LightningModule)
```

**Purpose**: Provides PyTorch Lightning training framework integration including loss computation, optimization, and training loops.

**Key Parameters**:

- `optimizer` (str): Optimizer class name from `torch.optim`. Default: `"Adam"`
- `learning_rate` (float): Base learning rate. Default: `0.001`
- `scheduler` (str): Learning rate scheduler name. Default: `"StepLR"`
- `criterion_params` (List[Tuple[str, Dict]]): Loss function specifications
- `log_every_n_steps` (int): Frequency of parameter logging. Default: `50`

**Core Training Methods**:

#### `training_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor`
Single training step implementation.

**Parameters**:

- `batch` (Tuple): Input data and labels
- `batch_idx` (int): Batch index

**Returns**: `torch.Tensor` containing the loss value

#### `validation_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor`
Single validation step implementation.

**Parameters**:

- `batch` (Tuple): Input data and labels
- `batch_idx` (int): Batch index

**Returns**: `torch.Tensor` containing the loss value

#### `test_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor`
Single test step implementation.

**Parameters**:

- `batch` (Tuple): Input data and labels
- `batch_idx` (int): Batch index

**Returns**: `torch.Tensor` containing the loss value

#### `configure_optimizers() -> Dict[str, Any]`
Configure optimizers and learning rate schedulers with parameter grouping.

**Returns**: Dictionary containing optimizer and scheduler configurations

#### `backward(loss: torch.Tensor, optimizer: Any = None, optimizer_idx: int = 0) -> None`
Perform backward pass with optional retain_graph for gradient computation.

**Parameters**:

- `loss` (torch.Tensor): Loss value to backpropagate
- `optimizer` (Any): Optimizer instance (provided by Lightning)
- `optimizer_idx` (int): Optimizer index for multi-optimizer setups

**PyTorch Lightning Hooks**:

LightningBase implements several PyTorch Lightning lifecycle hooks:

- `on_fit_start()`: Called at the beginning of fit
- `on_train_start()`: Called at the beginning of training
- `on_train_end()`: Called at the end of training
- `on_validation_start()`: Called at the beginning of validation
- `on_validation_epoch_end()`: Called at the end of validation epoch
- `on_test_start()`: Called at the beginning of testing
- `on_train_batch_start(batch, batch_idx, dataloader_idx)`: Called before each training batch
- `on_validation_batch_start(batch, batch_idx, dataloader_idx)`: Called before each validation batch
- `on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)`: Called after each training batch
- `on_before_optimizer_step(optimizer, optimizer_idx)`: Called before optimizer step

**Example**:
```python
class TrainingModel(TemporalBase, LightningBase):
    def _define_architecture(self):
        # Define architecture
        pass

model = TrainingModel(
    optimizer="SGD",
    learning_rate=0.01,
    scheduler="CosineAnnealingLR"
)
```

### StorageBuffer and StorageBufferMixin

Response and output storage management.

```python
class StorageBuffer:
    """Core storage functionality"""

class StorageBufferMixin(StorageBuffer):
    """Storage with PyTorch Lightning hooks"""
```

**Purpose**: Manages storage of neural responses, outputs, and metadata during training and evaluation.

**Key Methods**:

#### `get_responses() -> Dict[str, torch.Tensor]`
Retrieve stored neural responses.

**Returns**: Dictionary mapping layer names to response tensors

#### `get_dataframe(layer_name: str = "classifier") -> pd.DataFrame`
Generate a pandas DataFrame with classifier responses and metadata.

**Parameters**:

- `layer_name` (str): Name of classifier layer

**Returns**: `pd.DataFrame` with columns for responses, labels, predictions, and metadata

### Monitoring and MonitoringMixin

Debugging and performance monitoring utilities.

```python
class Monitoring:
    """Core monitoring functionality"""

class MonitoringMixin(Monitoring):
    """Monitoring with PyTorch Lightning hooks"""
```

**Purpose**: Provides comprehensive debugging, logging, and performance monitoring capabilities.

**Key Methods**:

#### `log_param_stats(section: str = "params", metrics: List[str] = ["min", "max", "norm"]) -> None`
Log statistics of model parameters.

**Parameters**:

- `section` (str): Section name for logging
- `metrics` (List[str]): Statistics to compute and log

#### `_check_weights(raise_error: bool = False) -> None`
Check model weights for NaN/Inf values and dtype consistency.

**Parameters**:

- `raise_error` (bool): Whether to raise exception on detection of issues

### DtypeDeviceCoordinator and DtypeDeviceCoordinatorMixin

Device and dtype coordination across model components.

```python
class DtypeDeviceCoordinator:
    """Core coordination functionality"""

class DtypeDeviceCoordinatorMixin(DtypeDeviceCoordinator):
    """Coordination with PyTorch Lightning hooks"""
```

**Purpose**: Ensures consistent data types and device placement across all model components with persistent state.

**Key Methods**:

#### `create_aligned_tensor(*args, **kwargs) -> torch.Tensor`
Create tensors with correct dtype and device for the coordination network.

**Parameters**:

- `size` (Tuple[int]): Tensor dimensions
- `creation_method` (str): Tensor creation method ("randn", "zeros", "ones"). Default: `"randn"`
- `**kwargs`: Additional tensor creation parameters

**Returns**: `torch.Tensor` with appropriate dtype and device

#### `propagate_dtype_sync() -> None`
Synchronize dtype and device across all coordinated components.

## Alternative Compositions

For advanced users who need specific functionality combinations:

### CoreModel
```python
class CoreModel(TemporalBase, DtypeDeviceCoordinator):
    """Neural network with device coordination only"""
```

### MonitoredModel  
```python
class MonitoredModel(TemporalBase, Monitoring, DtypeDeviceCoordinator):
    """Neural network with monitoring but no Lightning integration"""
```

### LightningOnlyModel
```python
class LightningOnlyModel(TemporalBase, LightningBase, MonitoringMixin):
    """Training framework without automatic storage"""
```

## Usage Patterns

### Basic Model Development
```python
from dynvision.base import BaseModel

class MyNeuralNetwork(BaseModel):
    def _define_architecture(self):
        self.layer_names = ['input', 'hidden', 'output']
        self.input = nn.Conv2d(3, 64, 3)
        self.hidden = nn.Conv2d(64, 128, 3)
        self.output = nn.Conv2d(128, self.n_classes, 1)
        self.classifier = nn.AdaptiveAvgPool2d(1)

# Instantiate with full functionality
model = MyNeuralNetwork(
    input_dims=(20, 3, 64, 64),
    n_classes=10,
    learning_rate=0.001,
    store_responses=1000
)
```

### Research-Focused Development
```python
from dynvision.base import CoreModel, Monitoring

class ResearchModel(CoreModel, Monitoring):
    def _define_architecture(self):
        # Minimal setup for experimentation
        pass

# No Lightning overhead, just core functionality
model = ResearchModel(input_dims=(5, 1, 28, 28))
```

### Custom Lightning Integration
```python
from dynvision.base import TemporalBase, LightningBase

class CustomTrainingModel(TemporalBase, LightningBase):
    def _define_architecture(self):
        # Custom architecture
        pass
    
    def training_step(self, batch, batch_idx):
        # Custom training logic
        return super().training_step(batch, batch_idx)
```

## Implementation Notes

### Method Resolution Order (MRO)
The inheritance order in `BaseModel` ensures proper method resolution:

1. `TemporalBase` provides core neural network methods
2. `LightningBase` can call DynVision methods in training steps
3. Storage and monitoring mixins add Lightning hooks
4. Device coordination ensures consistency across all components

### Parameter Flow
- Core neural network parameters are handled by `TemporalBase`
- Training configuration is managed by `LightningBase` 
- Component-specific parameters are stored by respective classes
- All classes accept `**kwargs` for flexible parameter passing

### Memory Management
- Storage components automatically manage memory usage
- Device coordination ensures optimal tensor placement
- Monitoring tools help identify memory leaks and performance issues

## Related Documentation

- [Getting Started Guide](../getting-started.md) - Basic usage tutorial
- [Custom Models Guide](../user-guide/custom-models.md) - Creating custom architectures
- [Model Testing Guide](../user-guide/model-testing.md) - Testing and evaluation
- [Temporal Data Presentation](../user-guide/temporal-data-presentation.md) - Temporal features and patterns
- [Loss Functions Reference](losses.md) - Loss configuration and behavior
- [Configuration Reference](configuration.md) - Configuration system documentation