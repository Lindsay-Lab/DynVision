# Building a Custom Model

This guide shows you how to create a custom recurrent convolutional neural network model in DynVision. We'll walk through the process step-by-step, from defining the architecture to training and evaluating the model.

## Prerequisites

- DynVision installed
- Basic understanding of PyTorch and convolutional neural networks
- Familiarity with recurrent neural networks

## Overview

Creating a custom model in DynVision involves these key steps:

1. Subclassing `LightningBase`
2. Defining the layer architecture
3. Implementing recurrent connections
4. Setting up dynamics
5. Configuring the training pipeline

Let's build a model called `CustomRCNN` that implements a three-layer network with flexible recurrent connectivity.

## Step 1: Basic Model Structure

Here's the basic structure of our custom model:

```python
import torch
import torch.nn as nn

from dynvision.model_components import LightningBase, RecurrentConnectedConv2d, EulerStep
from dynvision.utils import alias_kwargs, str_to_bool

class CustomRCNN(LightningBase):
    @alias_kwargs(
        tff="t_feedforward",
        trc="t_recurrence",
        rctype="recurrence_type",
        solver="dynamics_solver"
    )
    def __init__(
        self,
        n_classes=10,                  # Number of output classes
        input_dims=(20, 3, 224, 224),  # (timesteps, channels, height, width)
        dt=1.0,                        # Time step size in ms
        tau=10.0,                      # Time constant in ms
        t_feedforward=1.0,             # Feedforward delay in ms
        t_recurrence=1.0,              # Recurrent delay in ms
        recurrence_type="full",        # Type of recurrent connections
        dynamics_solver="euler",       # ODE solver type
        bias=True,                     # Whether to use bias
        **kwargs
    ) -> None:
        # Initialize base class
        super().__init__(
            n_classes=n_classes,
            input_dims=input_dims,
            t_recurrence=float(t_recurrence),
            t_feedforward=float(t_feedforward),
            tau=float(tau),
            dt=float(dt),
            bias=str_to_bool(bias),
            recurrence_type=recurrence_type,
            dynamics_solver=dynamics_solver,
            **kwargs
        )
        
        # Calculate delays in timesteps from ms
        self.delay_ff = int(t_feedforward / dt)
        self.delay_rc = int(t_recurrence / dt)
        
        # Define architecture and initialize parameters
        self._define_architecture()
        self._init_parameters()
    
    def _init_parameters(self):
        # Initialize model parameters (optional loading of pretrained weights)
        # For now, we'll just use default initialization
        pass
    
    def reset(self):
        # Reset all stateful components of the model
        for layer_name in self.layer_names:
            # Reset recurrent layers
            if hasattr(self, layer_name):
                getattr(self, layer_name).reset()
            
            # Reset temporal dynamics
            if hasattr(self, f"tstep_{layer_name}"):
                getattr(self, f"tstep_{layer_name}").reset()
```

## Step 2: Defining the Architecture

Now let's implement the `_define_architecture` method to set up our network architecture:

```python
def _define_architecture(self):
    # Define layer names and operations
    self.layer_names = ["V1", "V2", "V3"]
    self.layer_operations = [
        "conv",      # Apply convolution
        "nonlin",    # Apply nonlinearity
        "recurrent", # Apply recurrent connections
        "tstep",     # Apply dynamics step (ODE integration)
        "pool",      # Apply pooling
        "norm",      # Apply normalization
        "record",    # Record activations
        "delay"      # Set/get delayed activations
    ]
    
    # V1 layer (early visual processing)
    self.V1_conv = nn.Conv2d(
        in_channels=self.n_channels,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=self.bias
    )
    self.V1_nonlin = nn.ReLU(inplace=False)
    self.V1_recurrent = RecurrentConnectedConv2d(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        recurrence_type=self.recurrence_type,
        dt=self.dt,
        tau=self.tau,
        history_length=self.delay_ff,
        delay_recurrence=self.delay_rc,
        dynamics_solver=self.dynamics_solver,
        bias=self.bias,
        device=self.device
    )
    self.V1_tstep = EulerStep(dt=self.dt, tau=self.tau)
    self.V1_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.V1_norm = nn.BatchNorm2d(64)
    
    # V2 layer (intermediate visual processing)
    self.V2_conv = nn.Conv2d(
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=self.bias
    )
    self.V2_nonlin = nn.ReLU(inplace=False)
    self.V2_recurrent = RecurrentConnectedConv2d(
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        recurrence_type=self.recurrence_type,
        dt=self.dt,
        tau=self.tau,
        history_length=self.delay_ff,
        delay_recurrence=self.delay_rc,
        dynamics_solver=self.dynamics_solver,
        bias=self.bias,
        device=self.device
    )
    self.V2_tstep = EulerStep(dt=self.dt, tau=self.tau)
    self.V2_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.V2_norm = nn.BatchNorm2d(128)
    
    # V3 layer (higher visual processing)
    self.V3_conv = nn.Conv2d(
        in_channels=128,
        out_channels=256,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=self.bias
    )
    self.V3_nonlin = nn.ReLU(inplace=False)
    self.V3_recurrent = RecurrentConnectedConv2d(
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        recurrence_type=self.recurrence_type,
        dt=self.dt,
        tau=self.tau,
        history_length=self.delay_ff,
        delay_recurrence=self.delay_rc,
        dynamics_solver=self.dynamics_solver,
        bias=self.bias,
        device=self.device
    )
    self.V3_tstep = EulerStep(dt=self.dt, tau=self.tau)
    self.V3_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.V3_norm = nn.BatchNorm2d(256)
    
    # Classifier (decision layer)
    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, self.n_classes)
    )
```

The model has three visual processing layers (V1, V2, V3), each with convolution, nonlinearity, recurrent connections, dynamics integration, pooling, and normalization.

## Step 3: Adding Skip Connections (Optional)

Let's enhance our model with skip connections to improve gradient flow:

```python
def _define_architecture(self):
    # Original code as before...
    
    # Add skip connections
    self.layer_operations.insert(1, "addskip")  # Add after conv
    
    # Skip connection from V1 to V3
    self.skip_V1_V3 = nn.Sequential(
        nn.Conv2d(64, 256, kernel_size=1, stride=4, bias=False),
        nn.BatchNorm2d(256)
    )
    
    # Skip connection from V2 to V3
    self.skip_V2_V3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(256)
    )
```

## Step 4: Custom Forward Method (Optional)

If you need more control over how signals flow through your model, you can override the `_forward` method:

```python
def _forward(self, x_0, t=None, **kwargs):
    # Store responses if requested
    responses = {} if self.store_responses else None
    
    # Process input
    x_0 = self._expand_input_channels(x_0)
    
    # Process through layers using the base class implementation
    # This will apply operations defined in layer_operations in order
    # to each layer in layer_names
    x, responses = self._process_through_layers(x_0, t, responses)
    
    # Apply classifier
    if responses is not None:
        responses["classifier"] = self.classifier(x)
    
    return self.classifier(x), responses
```

However, for most cases, you don't need to override `_forward` as the base class implementation will automatically apply the operations defined in `layer_operations` to each layer in `layer_names`.

## Step 5: Adding Custom Skip Connection Logic

If you want to customize how skip connections are handled:

```python
def V3_addskip(self, x, responses=None):
    # Get previous activations (from history)
    v1_act = self.get_response_from_history('V1', -1)
    v2_act = self.get_response_from_history('V2', -1)
    
    # Apply skip connections if activations exist
    if v1_act is not None:
        x = x + self.skip_V1_V3(v1_act)
    if v2_act is not None:
        x = x + self.skip_V2_V3(v2_act)
    
    return x
```

## Step 6: Putting It All Together

Here's the complete model implementation with all features:

```python
import torch
import torch.nn as nn

from dynvision.model_components import LightningBase, RecurrentConnectedConv2d, EulerStep
from dynvision.utils import alias_kwargs, str_to_bool

class CustomRCNN(LightningBase):
    @alias_kwargs(
        tff="t_feedforward",
        trc="t_recurrence",
        rctype="recurrence_type",
        solver="dynamics_solver"
    )
    def __init__(
        self,
        n_classes=10,
        input_dims=(20, 3, 224, 224),
        dt=1.0,
        tau=10.0,
        t_feedforward=1.0,
        t_recurrence=1.0,
        recurrence_type="full",
        dynamics_solver="euler",
        bias=True,
        use_skip_connections=True,
        **kwargs
    ) -> None:
        super().__init__(
            n_classes=n_classes,
            input_dims=input_dims,
            t_recurrence=float(t_recurrence),
            t_feedforward=float(t_feedforward),
            tau=float(tau),
            dt=float(dt),
            bias=str_to_bool(bias),
            recurrence_type=recurrence_type,
            dynamics_solver=dynamics_solver,
            **kwargs
        )
        
        self.delay_ff = int(t_feedforward / dt)
        self.delay_rc = int(t_recurrence / dt)
        self.use_skip_connections = str_to_bool(use_skip_connections)
        
        self._define_architecture()
        self._init_parameters()
    
    def _init_parameters(self):
        # Use kaiming initialization for convolutional layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def _define_architecture(self):
        # Define layer names and operations
        self.layer_names = ["V1", "V2", "V3"]
        
        # Define operations pipeline
        self.layer_operations = [
            "conv",      # Apply convolution
            "nonlin",    # Apply nonlinearity
        ]
        
        # Add skip connections if enabled
        if self.use_skip_connections:
            self.layer_operations.append("addskip")
        
        # Add remaining operations
        self.layer_operations.extend([
            "recurrent", # Apply recurrent connections
            "tstep",     # Apply dynamics step
            "pool",      # Apply pooling
            "norm",      # Apply normalization
            "record",    # Record activations
            "delay"      # Handle delayed activations
        ])
        
        # V1 layer
        self.V1_conv = nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=self.bias
        )
        self.V1_nonlin = nn.ReLU(inplace=False)
        self.V1_recurrent = RecurrentConnectedConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.delay_ff,
            delay_recurrence=self.delay_rc,
            dynamics_solver=self.dynamics_solver,
            bias=self.bias,
            device=self.device
        )
        self.V1_tstep = EulerStep(dt=self.dt, tau=self.tau)
        self.V1_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.V1_norm = nn.BatchNorm2d(64)
        
        # V2 layer
        self.V2_conv = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias
        )
        self.V2_nonlin = nn.ReLU(inplace=False)
        self.V2_recurrent = RecurrentConnectedConv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.delay_ff,
            delay_recurrence=self.delay_rc,
            dynamics_solver=self.dynamics_solver,
            bias=self.bias,
            device=self.device
        )
        self.V2_tstep = EulerStep(dt=self.dt, tau=self.tau)
        self.V2_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.V2_norm = nn.BatchNorm2d(128)
        
        # V3 layer
        self.V3_conv = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias
        )
        self.V3_nonlin = nn.ReLU(inplace=False)
        self.V3_recurrent = RecurrentConnectedConv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.delay_ff,
            delay_recurrence=self.delay_rc,
            dynamics_solver=self.dynamics_solver,
            bias=self.bias,
            device=self.device
        )
        self.V3_tstep = EulerStep(dt=self.dt, tau=self.tau)
        self.V3_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.V3_norm = nn.BatchNorm2d(256)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, self.n_classes)
        )
        
        # Skip connections
        if self.use_skip_connections:
            # V1 to V3 skip connection
            self.skip_V1_V3 = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1, stride=4, bias=False),
                nn.BatchNorm2d(256)
            )
            
            # V2 to V3 skip connection
            self.skip_V2_V3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256)
            )
    
    def V3_addskip(self, x, responses=None):
        # Custom skip connection logic for V3 layer
        v1_act = self.get_response_from_history('V1', -1)
        v2_act = self.get_response_from_history('V2', -1)
        
        if v1_act is not None:
            x = x + self.skip_V1_V3(v1_act)
        if v2_act is not None:
            x = x + self.skip_V2_V3(v2_act)
        
        return x
    
    def reset(self):
        # Reset all stateful components
        for layer_name in self.layer_names:
            # Reset recurrent layers
            for op in ['recurrent', 'tstep']:
                if hasattr(self, f"{layer_name}_{op}"):
                    getattr(self, f"{layer_name}_{op}").reset()
        
        # Clear cached responses
        self.clear_response_history()
```

## Step 7: Using the Custom Model

Let's see how to use our custom model for training:

```python
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from dynvision.data.dataloader import StimulusDurationDataLoader

# Create model
model = CustomRCNN(
    n_classes=10,
    input_dims=(20, 3, 32, 32),  # CIFAR10 image size
    recurrence_type='full',
    dt=2.0,
    tau=10.0,
    t_feedforward=2.0,
    t_recurrence=2.0,
    use_skip_connections=True,
    store_responses=True
)

# Load CIFAR10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

# Create temporal data loaders
train_loader = StimulusDurationDataLoader(
    train_dataset,
    batch_size=32,
    n_timesteps=20,
    stimulus_duration=10,
    intro_duration=2,
    num_workers=4,
    shuffle=True
)

test_loader = StimulusDurationDataLoader(
    test_dataset,
    batch_size=32,
    n_timesteps=20,
    stimulus_duration=10,
    intro_duration=2,
    num_workers=4,
    shuffle=False
)

# Configure trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator='auto',
    devices=1,
    logger=pl.loggers.TensorBoardLogger('logs/'),
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor='val_accuracy',
            filename='custom-rcnn-{epoch:02d}-{val_accuracy:.2f}',
            save_top_k=3,
            mode='max'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]
)

# Train the model
trainer.fit(model, train_loader, test_loader)
```

## Step 8: Analyzing Temporal Dynamics

After training, let's analyze the model's temporal dynamics:

```python
import matplotlib.pyplot as plt
import numpy as np

# Get a batch of test data
dataiter = iter(test_loader)
inputs, targets, *_ = next(dataiter)

# Run forward pass
model.eval()
with torch.no_grad():
    outputs, responses = model(inputs)

# Plot layer activations over time for a single sample
sample_idx = 0
plt.figure(figsize=(15, 10))

for i, layer_name in enumerate(model.layer_names):
    layer_response = responses[layer_name][sample_idx]
    
    # Average over channels and spatial dimensions
    mean_activity = layer_response.mean(dim=(1, 2, 3)).cpu().numpy()
    
    plt.subplot(3, 1, i+1)
    plt.plot(mean_activity)
    plt.title(f'Layer {layer_name} Response')
    plt.ylabel('Mean Activation')
    plt.xlabel('Time Step')
    
    # Mark stimulus presentation period
    plt.axvspan(2, 12, alpha=0.2, color='gray')
    plt.axvline(2, linestyle='--', color='r', label='Stimulus On')
    plt.axvline(12, linestyle='--', color='b', label='Stimulus Off')
    plt.legend()

plt.tight_layout()
plt.savefig('temporal_dynamics.png')
plt.show()

# Plot response latency across layers
def calculate_latency(responses, threshold=0.5):
    latencies = {}
    
    for layer_name in model.layer_names:
        # Average over batch, channels, and spatial dimensions
        mean_activity = responses[layer_name].mean(dim=(0, 2, 3, 4)).cpu().numpy()
        
        # Normalize
        norm_activity = (mean_activity - mean_activity.min()) / (mean_activity.max() - mean_activity.min())
        
        # Find first time point exceeding threshold
        threshold_crossings = np.where(norm_activity > threshold)[0]
        if len(threshold_crossings) > 0:
            latencies[layer_name] = threshold_crossings[0]
        else:
            latencies[layer_name] = None
    
    return latencies

latencies = calculate_latency(responses)

plt.figure(figsize=(8, 5))
plt.bar(list(latencies.keys()), [lat for lat in latencies.values() if lat is not None])
plt.title('Response Latency by Layer')
plt.ylabel('Time Steps to Threshold')
plt.savefig('response_latency.png')
plt.show()
```

## Extending Further

You can extend this model in many ways:

### Adding Feedback Connections

```python
# Add feedback connections from V3 to V1
self.feedback_V3_V1 = nn.Sequential(
    nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4, bias=False),
    nn.BatchNorm2d(64)
)

# Add to layer operations
self.layer_operations.insert(3, "addfeedback")  # After nonlin, before recurrent

# Implement custom method
def V1_addfeedback(self, x, responses=None):
    v3_act = self.get_response_from_history('V3', -1)
    if v3_act is not None:
        x = x + self.feedback_V3_V1(v3_act)
    return x
```

### Custom Dynamics Solver

```python
from torch.autograd import Function

class CustomRK4Step(Function):
    @staticmethod
    def forward(ctx, x, driving_input, dt, tau):
        ctx.save_for_backward(x, driving_input)
        ctx.dt = dt
        ctx.tau = tau
        
        # RK4 step
        k1 = (dt/tau) * (driving_input - x)
        k2 = (dt/tau) * (driving_input - (x + 0.5*k1))
        k3 = (dt/tau) * (driving_input - (x + 0.5*k2))
        k4 = (dt/tau) * (driving_input - (x + k3))
        
        return x + (k1 + 2*k2 + 2*k3 + k4) / 6

    @staticmethod
    def backward(ctx, grad_output):
        x, driving_input = ctx.saved_tensors
        dt, tau = ctx.dt, ctx.tau
        
        # Simplified backward pass
        dx = -dt/tau
        ddriving = dt/tau
        
        grad_x = grad_output * (1 + dx)
        grad_driving = grad_output * ddriving
        
        return grad_x, grad_driving, None, None

class CustomRK4(nn.Module):
    def __init__(self, dt=1.0, tau=10.0):
        super().__init__()
        self.dt = dt
        self.tau = tau

    def forward(self, x, driving_input):
        return CustomRK4Step.apply(x, driving_input, self.dt, self.tau)
        
    def reset(self):
        pass
```

## Complete Examples

You can find complete examples of custom models in the `examples/` directory of the DynVision repository:

1. `examples/custom_models/simple_rcnn.py` - A simple recurrent CNN
2. `examples/custom_models/adaptive_rcnn.py` - RCNN with adaptive recurrence
3. `examples/custom_models/multi_area_model.py` - More complex model with multiple areas

## Best Practices

When building custom models:

1. **Start Simple**: Begin with a small model and gradually add complexity
2. **Validate Each Step**: Test each component separately before combining them
3. **Debugging**: Use `print()` statements or logging to track tensor shapes and values
4. **Parameter Sharing**: Consider sharing parameters between feedforward and recurrent paths
5. **Stable Initialization**: Initialize recurrent weights carefully to avoid instability

## Next Steps

Now that you've created a custom model, you might want to:

- Experiment with different recurrence types
- Try various temporal stimuli (contrast, interval, etc.)
- Analyze the emergent representational properties
- Compare your model's dynamics to neural recordings

See the [Visualization Guide](../user-guide/visualization.md) for details on how to analyze and visualize your model's behavior.
