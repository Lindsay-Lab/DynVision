# Model Components Reference

This reference documentation details DynVision's model components, their interfaces, and technical capabilities.

## Base Classes

### LightningBase

Core base class providing PyTorch Lightning integration and essential model features:

#### Input Processing

Handles various input formats automatically:
```python
def _adjust_input_dimensions(self, x):
    """
    Adjust input tensor dimensions to standard format.
    
    Supported input formats:
    - (dim_y, dim_x)
    - (batch_size, dim_y, dim_x)
    - (batch_size, channels, dim_y, dim_x)
    - (batch_size, timesteps, channels, dim_y, dim_x)
    
    Returns:
        torch.Tensor: Shape (batch, timesteps, channels, height, width)
    """
```

#### Timestep Management

Handles temporal processing across timesteps:

```python
def _determine_residual_timesteps(self):
    """
    Determine required residual timesteps for recurrent processing.
    
    Returns:
        int: Number of residual timesteps needed
    """

def _extend_residual_timesteps(self, batch):
    """
    Add residual timesteps to input batch.
    
    Args:
        batch: Input batch
    Returns:
        tuple: Batch with extended timesteps
    """
```

#### Response Management

Built-in response tracking and analysis:

```python
def get_responses(self):
    """Get stored model responses."""
    
def get_dataframe(self):
    """Get classifier responses as pandas DataFrame."""
    
def _update_responses(self, response_dict, t=None):
    """Update stored responses."""
```

#### State Management

State handling for recurrent models:

```python
def set_hidden_state(self, state, t=None):
    """Set hidden state for timestep t."""
    
def get_hidden_state(self, t):
    """Get hidden state from timestep t."""
    
def reset(self):
    """Reset all stateful components."""
```

#### Layer Operations

Customizable operation sequences:

```python
# Available operations:
layer_operations = [
    "layer",       # Main layer computation
    "addskip",     # Add skip connections
    "addfeedback", # Add feedback connections
    "tstep",       # Apply dynamics step
    "nonlin",      # Apply nonlinearity
    "supralin",    # Apply supralinearity
    "record",      # Store responses
    "delay",       # Handle delayed activations
    "pool",        # Apply pooling
    "norm"         # Apply normalization
]
```

Each operation can be customized or disabled per layer.

#### Training Integration

PyTorch Lightning integration features:

```python
def training_step(self, batch, batch_idx):
    """Custom training with response tracking."""
    
def configure_optimizers(self):
    """Configure optimizers and learning rate schedules."""
    
def log_param_stats(self, section="params"):
    """Log parameter statistics during training."""
```

### DyRCNN

Specialized base class for recurrent convolutional networks:

```python
class DyRCNN(LightningBase):
    """
    Base class for dynamic recurrent CNNs.
    
    Adds:
    - Automatic recurrent connectivity
    - Neural dynamics integration
    - Biological plausibility features
    """
```

## Neural Components

### RecurrentConnectedConv2d

Convolutional layer with recurrent connections:

```python
class RecurrentConnectedConv2d(nn.Module):
    """
    Convolutional layer with recurrent connectivity.
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int): Kernel size
        recurrence_type (str): Type of recurrence
        dt (float): Integration timestep
        tau (float): Neural time constant
    """
```

Supported recurrence types:
- "full": Full connectivity
- "local": Local connectivity
- "none": No recurrence

### EulerStep

Neural dynamics solver using Euler integration:

```python
class EulerStep:
    """
    Euler integration step for neural dynamics.
    
    Args:
        dt (float): Integration timestep
        tau (float): Neural time constant
    """
    
    def forward(self, x, h):
        """
        Compute one integration step.
        
        Args:
            x: Input current
            h: Hidden state
        Returns:
            Updated hidden state
        """
```

### Skip and Feedback

Inter-layer connections:

```python
class Skip(nn.Module):
    """
    Skip connection between layers.
    
    Args:
        source (nn.Module): Source layer
        auto_adapt (bool): Auto-adapt dimensions
    """

class Feedback(nn.Module):
    """
    Feedback connection between layers.
    
    Args:
        source (nn.Module): Source layer
        auto_adapt (bool): Auto-adapt dimensions
    """
```

## Initialization

Parameter initialization utilities:

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
```

## Debugging Tools

Built-in debugging utilities:

```python
def _check_weights(self, raise_error=False):
    """Check weights for numerical issues."""
    
def _check_gradients(self, raise_error=False):
    """Check gradients for numerical issues."""
    
def _check_responses(self, raise_error=False):
    """Check responses for numerical issues."""
```

## Implementation Notes

1. **State Management**
   - Use `set_hidden_state`/`get_hidden_state` for explicit state control
   - Implement `reset()` for all stateful components
   - Clear states between sequences

2. **Response Tracking**
   - Enable with `store_responses=True`
   - Access via `get_responses()`
   - Convert to DataFrame with `get_dataframe()`

3. **Layer Operations**
   - Define sequence in `layer_operations`
   - Operations execute in order for each layer
   - Skip operations with empty implementation

4. **Performance**
   - Use gradient checkpointing for memory efficiency
   - Enable response storage only when needed

For usage examples, see the [Custom Models Guide](../user-guide/custom-models.md).