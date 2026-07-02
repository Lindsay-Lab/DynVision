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
    
    def reset(self, input_shape: Optional[Tuple[int, ...]] = None) :
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

DynVision provides extensive flexibility for training customization through configuration files and PyTorch Lightning callbacks.

#### Optimizer Configuration

Configure optimizers and their parameters in `config_defaults.yaml`:

```yaml
# Basic optimizer settings
optimizer: Adam  # Options: Adam, AdamW, SGD, RMSprop, Adagrad
learning_rate: 0.0002

# Optimizer-specific parameters
optimizer_kwargs:
  weight_decay: 0.01  # For regularization
  betas: [0.9, 0.999]  # Adam momentum parameters
  eps: 1.0e-08         # Numerical stability

# Advanced: mode-specific optimizer configs
optimizer_configs:
  train:
    learning_rate: 0.001
  finetune:
    learning_rate: 0.0001
    weight_decay: 0.001
```

See [Optimizers and Schedulers Reference](../reference/optimizers-schedulers.md) for complete options.

#### Learning Rate Scheduling

Configure learning rate schedules to improve training:

```yaml
# Scheduler type
scheduler: CosineAnnealingLR  # or LinearWarmupCosineAnnealingLR, StepLR, etc.

# Scheduler parameters
scheduler_kwargs:
  T_max: 100        # For CosineAnnealingLR
  eta_min: 0.00001  # Minimum learning rate

# Alternative: Warmup + Cosine schedule
# scheduler: LinearWarmupCosineAnnealingLR
# scheduler_kwargs:
#   warmup_epochs: 10
#   max_epochs: 100
#   warmup_start_lr: 0.0
#   eta_min: 0.00001
```

#### Custom PyTorch Lightning Callbacks

Override or add custom callbacks for advanced training control:

```python
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

class MyCustomCallback(Callback):
    """Custom callback for specialized training behavior."""

    def on_train_epoch_end(self, trainer, pl_module):
        """Execute custom logic at end of each epoch."""
        # Example: Log custom metrics
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        pl_module.log('learning_rate', current_lr)

        # Example: Custom early stopping logic
        if pl_module.current_epoch > 50:
            val_loss = trainer.callback_metrics.get('val_loss')
            if val_loss and val_loss < 0.01:
                print("Custom stopping criterion met!")
                trainer.should_stop = True

# Add callback to your model initialization
def train_with_custom_callbacks():
    from dynvision.base import BaseModel

    model = MyModel(input_dims=(20, 3, 32, 32))

    # Define custom callbacks
    callbacks = [
        MyCustomCallback(),
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            filename='best-{epoch:02d}-{val_loss:.2f}'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ]

    # Train with custom callbacks
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        accelerator='gpu',
        devices=1
    )
    trainer.fit(model, train_dataloader, val_dataloader)
```

**Available Callback Hooks:**

PyTorch Lightning provides extensive callback hooks. Common ones for custom models:

- `on_train_start/end`: Setup/teardown for training
- `on_train_epoch_start/end`: Per-epoch logic
- `on_train_batch_start/end`: Per-batch logic
- `on_validation_epoch_end`: Custom validation metrics
- `on_save_checkpoint`: Modify checkpoint contents
- `on_load_checkpoint`: Custom checkpoint loading

See [PyTorch Lightning Callbacks Documentation](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) for complete list.

#### Advanced Training Options

Additional configuration options for specialized training:

```yaml
# Mixed precision training (faster on modern GPUs)
precision: "bf16-mixed"  # or "16-mixed" for older GPUs

# Gradient clipping (prevent exploding gradients)
gradient_clip_val: 1.0
gradient_clip_algorithm: 'norm'

# Gradient accumulation (effective batch size increase)
accumulate_grad_batches: 4

# Validation frequency
check_val_every_n_epoch: 5

# Checkpoint monitoring
monitor: "val_loss"  # Metric to track for checkpoints
save_top_k: 3        # Keep top 3 checkpoints
```

For complete training configuration details, see [Workflows Guide](../user-guide/workflows.md).


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

See [Codebase Organization Reference](../reference/organization.md) for module structure.

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

### Using Debug Mode

Debug mode provides faster iteration during development by reducing batch sizes and increasing logging frequency.

#### Enabling Debug Mode

**Method 1: Via Configuration**

```yaml
# In config_modes.yaml or config_defaults.yaml
use_debug_mode: True
```

**Method 2: Via Command Line**

```bash
# Snakemake workflow
snakemake train_model --config use_debug_mode=True model_name=MyModel

# Direct Python script
python runtime/train_model.py --use_debug_mode True
```

**Method 3: Automatic Activation**

Debug mode activates automatically when:

- `log_level: "DEBUG"` is set, OR
- `epochs <= 5` (short training runs)

#### Debug Mode Settings

When debug mode is active, the following parameters override defaults:

```yaml
debug_mode:
  batch_size: 3                    # Small batches for quick iteration
  check_val_every_n_epoch: 1       # Validate every epoch
  log_every_n_steps: 1             # Log every batch
  accumulate_grad_batches: 1       # No gradient accumulation
  enable_progress_bar: True        # Show progress
```

#### Common Issues and Solutions

**Problem: Model outputs NaN**

```python
# Check for NaN in model outputs
def forward(self, x):
    output = super().forward(x)

    # Debug: Check for NaN
    if torch.isnan(output).any():
        print(f"NaN detected in output!")
        print(f"Input stats: min={x.min()}, max={x.max()}, mean={x.mean()}")
        print(f"Output stats: {output[~torch.isnan(output)].describe()}")
        raise ValueError("NaN in forward pass")

    return output
```

**Solutions:**

- Reduce learning rate
- Add gradient clipping: `gradient_clip_val: 1.0`
- Check weight initialization
- Enable debug logging: `log_level: "DEBUG"`

**Problem: Recurrent connections not working**

```python
# Debug: Verify recurrent weights are being used
def _define_architecture(self):
    super()._define_architecture()

    # Check recurrent parameters exist
    for name, module in self.named_modules():
        if hasattr(module, 'recurrence'):
            print(f"Layer {name} has recurrence: {module.recurrence}")
            if hasattr(module.recurrence, 'weight'):
                print(f"  Weight shape: {module.recurrence.weight.shape}")
                print(f"  Weight range: [{module.recurrence.weight.min():.4f}, "
                      f"{module.recurrence.weight.max():.4f}]")
```

**Solutions:**

- Verify `n_timesteps > 1`
- Check temporal parameters: `dt`, `tau`, `t_recurrence`
- Ensure recurrence type is not `"none"`

**Problem: Training extremely slow**

**Diagnostic:**
```python
import time

def training_step(self, batch, batch_idx):
    start = time.time()

    # ... training logic ...

    elapsed = time.time() - start
    self.log('batch_time', elapsed)

    if elapsed > 1.0:  # Longer than 1 second
        print(f"Slow batch {batch_idx}: {elapsed:.2f}s")

    return loss
```

**Solutions:**

- Enable FFCV: `use_ffcv: True`
- Increase `num_workers: 8`
- Use mixed precision: `precision: "bf16-mixed"`
- Check for unnecessary synchronization points
- Profile with PyTorch profiler

**Problem: Out of memory errors**

```python
# Debug: Monitor memory usage
import torch

def on_train_batch_start(self, batch, batch_idx):
    if batch_idx % 100 == 0:
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
```

**Solutions:**

- Reduce `batch_size`
- Reduce `n_timesteps`
- Enable gradient accumulation
- Disable response storage during training: `store_responses: 0`
- Use mixed precision training

### Debugging Tools

#### Enable PyTorch Anomaly Detection

Catch NaN/Inf gradients immediately:

```python
import torch

# Enable before training
torch.autograd.set_detect_anomaly(True)

# Train model
trainer.fit(model)
```

#### Verbose Logging

Enable detailed logging to track execution:

```yaml
# In config
log_level: "DEBUG"
verbose: True
```

#### Response Inspection

Store and inspect intermediate activations:

```python
# Enable response storage
model.eval()
with torch.no_grad():
    output = model(test_input, store_responses=True)

# Inspect responses
responses = model.get_responses()
for layer_name, response in responses.items():
    print(f"{layer_name}: shape={response.shape}, "
          f"mean={response.mean():.4f}, std={response.std():.4f}")

    # Check for issues
    if torch.isnan(response).any():
        print(f"  WARNING: NaN detected in {layer_name}")
    if (response == 0).all():
        print(f"  WARNING: All zeros in {layer_name}")
```

### Performance Profiling

Use PyTorch profiler to identify bottlenecks:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(test_batch)

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")
# View in chrome://tracing
```

For additional troubleshooting help, see:

- [General Troubleshooting Guide](troubleshooting.md)
- [Cluster Integration Issues](cluster-integration.md#troubleshooting)
- [Loss Function Debugging](../reference/losses.md#troubleshooting)

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
- Check [model testing guide](model-testing.md) for evaluation approaches

## Related Resources

- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [BrainScore]
- ...